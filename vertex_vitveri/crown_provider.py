from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


def load_abcrown_class(abcrown_path):
    abcrown_path = Path(abcrown_path).resolve()
    if not abcrown_path.is_file():
        raise FileNotFoundError(f"abcrown.py does not exist: {abcrown_path}")
    root = abcrown_path.parents[1]
    for path in (abcrown_path.parent, root, root / "auto_LiRPA"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    spec = importlib.util.spec_from_file_location("abcrown", abcrown_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ABCROWN


def generate_vnnlib(image, true_label, predicted_label, epsilon, num_classes):
    values = image.detach().cpu().numpy().reshape(-1)
    lines = [f"; true label = {true_label}, predicted = {predicted_label}", f"; epsilon = {epsilon}", ""]
    lines.extend(f"(declare-const X_{index} Real)" for index in range(len(values)))
    lines.extend(f"(declare-const Y_{index} Real)" for index in range(num_classes))
    lines.append("")
    for index, value in enumerate(values):
        lines.append(f"(assert (>= X_{index} {max(-1.0, value - epsilon)}))")
        lines.append(f"(assert (<= X_{index} {min(1.0, value + epsilon)}))")
    constraints = [
        f"(and (>= Y_{target} Y_{predicted_label}))"
        for target in range(num_classes)
        if target != predicted_label
    ]
    lines.extend(("", f"(assert (or {' '.join(constraints)}))"))
    return "\n".join(lines)


def _node_name(node):
    return str(getattr(node, "name", ""))


def _node_type(node):
    return type(node).__name__


def _input_names(node):
    return [_node_name(value) for value in getattr(node, "inputs", [])]


def _is_nonlinear(node):
    return bool(getattr(node, "perturbed", False)) and bool(getattr(node, "requires_input_bounds", []))


def _softmax_score_node(exp_node, nodes):
    shifted = nodes.get((_input_names(exp_node) or [None])[0])
    return (_input_names(shifted) or [None])[0] if shifted is not None else None


def discover_attention_layers(net, expected_depth):
    all_nodes = list(net.nodes())
    nodes = {_node_name(node): node for node in all_nodes}
    layers = []
    current = None
    for node in (value for value in all_nodes if _is_nonlinear(value)):
        inputs = _input_names(node)
        if _node_type(node) == "BoundMatMul" and len(inputs) == 2 and not inputs[0].endswith("/mul"):
            if current is not None:
                layers.append(current)
            current = {
                "q": inputs[0],
                "k": inputs[1],
                "score_scaled": _node_name(node),
                "score_scaled_scale": "by_dim",
                "score_fallback_final_node": _node_name(node),
                "score_fallback_scale": "by_dim",
                "v": None,
                "softmax": None,
                "relu_input": None,
            }
        elif current is not None and _node_type(node) == "BoundExp":
            score = _softmax_score_node(node, nodes)
            if score and score != current["score_fallback_final_node"]:
                current["score_scaled"] = score
                current["score_scaled_scale"] = "none"
        elif current is not None and _node_type(node) == "BoundMatMul" and len(inputs) == 2 and inputs[0].endswith("/mul"):
            current["softmax"], current["v"] = inputs
        elif current is not None and _node_type(node) == "BoundRelu" and len(inputs) == 1:
            current["relu_input"] = inputs[0]
    if current is not None:
        layers.append(current)
    if len(layers) != expected_depth:
        raise ValueError(f"Discovered {len(layers)} attention layers, expected {expected_depth}")
    required = ("q", "k", "v", "softmax", "relu_input", "score_fallback_final_node")
    for index, layer in enumerate(layers):
        missing = [key for key in required if not layer.get(key)]
        if missing:
            raise ValueError(f"Attention layer {index} is missing nodes: {', '.join(missing)}")
    return layers


def _score_bounds(bounds, net, image, model, layer):
    lower = bounds["lower_bounds"]
    upper = bounds["upper_bounds"]
    node = layer["score_scaled"]
    if node in lower:
        score_l, score_u = lower[node], upper[node]
        scale_mode = layer["score_scaled_scale"]
    elif layer.get("score_unscaled") in lower:
        node = layer["score_unscaled"]
        score_l, score_u = lower[node], upper[node]
        scale_mode = layer["score_unscaled_scale"]
    else:
        node = layer["score_fallback_final_node"]
        score_l, score_u = net.compute_bounds(
            x=(image,),
            method="backward",
            final_node_name=node,
            reuse_alpha=False,
        )
        scale_mode = layer["score_fallback_scale"]
    if scale_mode == "by_dim":
        scale = 1.0 / math.sqrt(model.dim // model.heads)
        score_l, score_u = score_l * scale, score_u * scale
    return score_l, score_u


def _target_vector(values, predicted_label, num_classes, device, dtype):
    import torch

    result = torch.full((num_classes,), float("inf"), device=device, dtype=dtype)
    if values is None:
        return result
    flattened = values.detach().to(device=device, dtype=dtype).reshape(-1)
    targets = [target for target in range(num_classes) if target != predicted_label]
    if flattened.numel() != len(targets):
        raise ValueError(f"Expected {len(targets)} target margins, got {flattened.numel()}")
    result[targets] = flattened
    return result


@dataclass
class CrownBounds:
    initial_target_lowers: object
    alpha_target_lowers: object
    score_lower: object
    score_upper: object
    value_lower: object
    value_upper: object
    relu_lower: object
    relu_upper: object
    attention_layers: list


def _checked_bounds(lower, upper, node_name):
    if node_name not in lower or node_name not in upper:
        raise KeyError(f"ABCROWN did not store bounds for node {node_name}")
    node_lower = lower[node_name]
    node_upper = upper[node_name]
    if bool((node_lower > node_upper).any().item()):
        raise ValueError(f"ABCROWN returned inverted bounds for node {node_name}")
    return node_lower, node_upper


def run_crown_bounds(config, model, image, true_label, predicted_label, image_index, device):
    abcrown_entry = config.resolve_external_path(config.verification["abcrown_entry"])
    abcrown_config = config.resolve_external_path(config.verification["abcrown_config"])
    abcrown_class = load_abcrown_class(abcrown_entry)
    unique = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=f"vertex_{image_index}_{unique}_") as directory:
        directory = Path(directory)
        vnnlib = directory / "property.vnnlib"
        instances = directory / "instances.csv"
        vnnlib.write_text(
            generate_vnnlib(
                image,
                true_label,
                predicted_label,
                config.epsilon,
                config.model["num_classes"],
            ),
            encoding="utf-8",
        )
        instances.write_text(f"{vnnlib}\n", encoding="utf-8")
        verifier = abcrown_class(
            args=[
                "--config",
                str(abcrown_config),
                "--csv_name",
                str(instances),
                "--epsilon",
                str(config.epsilon),
                "--device",
                device,
                "--start",
                "0",
                "--end",
                "1",
            ]
        )
        verifier.main()
    if not hasattr(verifier, "last_computed_bounds"):
        raise RuntimeError("ABCROWN did not expose last_computed_bounds")

    bounds = verifier.last_computed_bounds
    net = verifier.net
    layers = discover_attention_layers(net, config.model["depth"])
    final = layers[-1]
    score_l, score_u = _score_bounds(bounds, net, image, model, final)
    lower = bounds["lower_bounds"]
    upper = bounds["upper_bounds"]
    value_l, value_u = _checked_bounds(lower, upper, final["v"])
    relu_l, relu_u = _checked_bounds(lower, upper, final["relu_input"])
    if bool((score_l > score_u).any().item()):
        raise ValueError("ABCROWN returned inverted attention-score bounds")
    dtype = image.dtype
    return CrownBounds(
        initial_target_lowers=_target_vector(
            getattr(verifier, "last_computed_init_crown_lb", None),
            predicted_label,
            config.model["num_classes"],
            image.device,
            dtype,
        ),
        alpha_target_lowers=_target_vector(
            getattr(verifier, "last_computed_alpha_crown_lb", None),
            predicted_label,
            config.model["num_classes"],
            image.device,
            dtype,
        ),
        score_lower=score_l.detach().to(image.device, dtype=dtype),
        score_upper=score_u.detach().to(image.device, dtype=dtype),
        value_lower=value_l.detach().to(image.device, dtype=dtype),
        value_upper=value_u.detach().to(image.device, dtype=dtype),
        relu_lower=relu_l.detach().to(image.device, dtype=dtype),
        relu_upper=relu_u.detach().to(image.device, dtype=dtype),
        attention_layers=layers,
    )
