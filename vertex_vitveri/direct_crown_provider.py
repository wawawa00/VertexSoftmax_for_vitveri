from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from .crown_provider import (
    CrownBounds,
    _checked_bounds,
    _score_bounds,
    _target_vector,
    discover_attention_layers,
    generate_vnnlib,
    load_abcrown_class,
)
from .direct_crown_hook import CAPTURE_ATTRIBUTE, capture_intermediate_bounds


def _bounded_module_class():
    from auto_LiRPA import BoundedModule

    return BoundedModule


def _direct_node_selector(expected_depth):
    selected_layers = []

    def select(net):
        layers = discover_attention_layers(net, expected_depth)
        selected_layers[:] = layers
        final = layers[-1]
        return final["block_input"], final["attention_input"]

    return selected_layers, select


def run_direct_crown_bounds(config, model, image, true_label, predicted_label, image_index, device):
    """Run ABCROWN while capturing final-block inputs on its original CROWN pass."""
    abcrown_entry = config.resolve_external_path(config.verification["abcrown_entry"])
    abcrown_config = config.resolve_external_path(config.verification["abcrown_config"])
    abcrown_class = load_abcrown_class(abcrown_entry)
    selected_layers, selector = _direct_node_selector(config.model["depth"])
    unique = uuid.uuid4().hex

    with tempfile.TemporaryDirectory(prefix=f"vertex_direct_{image_index}_{unique}_") as directory:
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
        with capture_intermediate_bounds(_bounded_module_class(), selector):
            verifier.main()

    if not hasattr(verifier, "last_computed_bounds"):
        raise RuntimeError("ABCROWN did not expose last_computed_bounds")
    if not hasattr(verifier.net, CAPTURE_ATTRIBUTE):
        raise RuntimeError("ABCROWN did not execute the direct intermediate-bound hook")

    layers = selected_layers or discover_attention_layers(verifier.net, config.model["depth"])
    final = layers[-1]
    direct = getattr(verifier.net, CAPTURE_ATTRIBUTE)
    z_l, z_u = direct[final["block_input"]]
    attention_l, attention_u = direct[final["attention_input"]]
    bounds = verifier.last_computed_bounds
    lower = bounds["lower_bounds"]
    upper = bounds["upper_bounds"]
    score_l, score_u = _score_bounds(bounds, verifier.net, image, model, final)
    value_l, value_u = _checked_bounds(lower, upper, final["v"])
    relu_l, relu_u = _checked_bounds(lower, upper, final["relu_input"])
    dtype = image.dtype

    result = CrownBounds(
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
        # The additive direct CLI bypasses the legacy Q and LayerNorm inverses,
        # so these compatibility fields carry the final block input directly.
        query_lower=z_l.detach().to(image.device, dtype=dtype),
        query_upper=z_u.detach().to(image.device, dtype=dtype),
        score_lower=score_l.detach().to(image.device, dtype=dtype),
        score_upper=score_u.detach().to(image.device, dtype=dtype),
        value_lower=value_l.detach().to(image.device, dtype=dtype),
        value_upper=value_u.detach().to(image.device, dtype=dtype),
        relu_lower=relu_l.detach().to(image.device, dtype=dtype),
        relu_upper=relu_u.detach().to(image.device, dtype=dtype),
        attention_layers=layers,
    )
    result.block_input_lower = z_l.detach().to(image.device, dtype=dtype)
    result.block_input_upper = z_u.detach().to(image.device, dtype=dtype)
    result.attention_input_lower = attention_l.detach().to(image.device, dtype=dtype)
    result.attention_input_upper = attention_u.detach().to(image.device, dtype=dtype)
    return result
