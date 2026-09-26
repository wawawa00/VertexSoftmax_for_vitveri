from __future__ import annotations

import time

import torch

from tiny_vit_block_benchmark import softmax_box_expectation_min


def _bounded_classes():
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except ImportError as exc:
        raise RuntimeError("auto_LiRPA is required for Vertex certification") from exc
    return BoundedModule, BoundedTensor, PerturbationLpNorm


def _patch_tokens(model, image):
    x = model.to_patch_embedding(image)
    batch, token_count, _ = x.shape
    cls = model.cls_token.expand(batch, -1, -1)
    x = torch.cat((cls, x), dim=1)
    return x + model.pos_embedding[: token_count + 1]


def _attention_forward(attention_residual, z, *, return_parts=False):
    prenorm = attention_residual.fn
    attention = prenorm.fn
    attention_input = prenorm.norm(z)
    q = _split_heads(attention.to_q(attention_input), attention.heads)
    k = _split_heads(attention.to_k(attention_input), attention.heads)
    v = _split_heads(attention.to_v(attention_input), attention.heads)
    scores = q.matmul(k.transpose(-1, -2)) * attention.scale
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    exponentials = torch.exp(shifted)
    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
    context = probabilities.matmul(v).transpose(1, 2).reshape(z.shape)
    output = attention.to_out[0] if isinstance(attention.to_out, torch.nn.Sequential) else attention.to_out
    residual = z + output(context)
    if return_parts:
        return attention_input, scores, residual
    return residual


def _feed_forward(feed_forward_residual, z):
    prenorm = feed_forward_residual.fn
    linear1, _relu, _dropout1, linear2, _dropout2 = prenorm.fn.net
    hidden = torch.relu(linear1(prenorm.norm(z)))
    return z + linear2(hidden)


def _transformer_layer(layer, z):
    attention_residual, feed_forward_residual = layer
    return _feed_forward(feed_forward_residual, _attention_forward(attention_residual, z))


def _final_block_input(model, image):
    x = _patch_tokens(model, image)
    for layer in model.transformer.layers[:-1]:
        x = _transformer_layer(layer, x)
    return x


def _final_components(model):
    attention_residual, feed_forward_residual = model.transformer.layers[-1]
    attention_prenorm = attention_residual.fn
    feed_forward_prenorm = feed_forward_residual.fn
    attention = attention_prenorm.fn
    output = attention.to_out[0] if isinstance(attention.to_out, torch.nn.Sequential) else attention.to_out
    return attention_prenorm.norm, attention, output, feed_forward_prenorm.norm, feed_forward_prenorm.fn


def _split_heads(tensor, heads):
    batch, tokens, _ = tensor.shape
    return tensor.reshape(batch, tokens, heads, -1).transpose(1, 2)


def _attention_parts(model, image):
    z = _final_block_input(model, image)
    attention_input, scores, residual = _attention_forward(
        model.transformer.layers[-1][0],
        z,
        return_parts=True,
    )
    return z, attention_input, scores, residual


def _stable_forward(model, image):
    z = _patch_tokens(model, image)
    for layer in model.transformer.layers:
        z = _transformer_layer(layer, z)
    pooled = z[:, 0]
    final_norm, classifier = model.mlp_head
    return classifier(final_norm(pooled))


class MarginModule(torch.nn.Module):
    def __init__(self, model, predicted_label):
        super().__init__()
        self.model = model
        self.predicted_label = int(predicted_label)

    def forward(self, image):
        logits = _stable_forward(self.model, image)
        predicted = logits[:, self.predicted_label : self.predicted_label + 1]
        return predicted - logits


class FinalBlockInputModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return _final_block_input(self.model, image)


class FinalAttentionInputModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return _attention_parts(self.model, image)[1]


class FinalScoreModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return _attention_parts(self.model, image)[2]


class FinalAttentionResidualModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return _attention_parts(self.model, image)[3]


def _compute_bounds(module, center, lower, upper, *, method="CROWN", bound_opts=None):
    BoundedModule, BoundedTensor, PerturbationLpNorm = _bounded_classes()
    bounded = BoundedModule(module.eval(), (center,), bound_opts=bound_opts or {"verbosity": 0})
    bounded_input = BoundedTensor(center, PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper))
    return bounded.compute_bounds(
        x=(bounded_input,),
        method=method,
        bound_lower=True,
        bound_upper=True,
    )


def crown_target_margins(model, center, lower, upper, predicted_label, method, alpha_iters):
    margins, _ = _compute_bounds(
        MarginModule(model, predicted_label),
        center,
        lower,
        upper,
        method=method,
        bound_opts={
            "verbosity": 0,
            "fixed_reducemax_index": True,
            "optimize_bound_args": {"iteration": alpha_iters, "lr_alpha": 0.1},
        },
    )
    result = margins.reshape(-1).clone()
    result[predicted_label] = float("inf")
    return result


def _no_var_affine(layer):
    dim = layer.weight.numel()
    centering = torch.eye(dim, device=layer.weight.device, dtype=layer.weight.dtype)
    centering = centering - torch.ones_like(centering) / dim
    matrix = torch.diag(layer.weight).matmul(centering)
    return matrix, layer.bias


def _suffix_coefficients(
    model,
    h_lower,
    h_upper,
    predicted_label,
    target_label,
    slope_policy,
    *,
    pre_lower=None,
    pre_upper=None,
):
    _norm1, _attention, _output, norm2, feed_forward = _final_components(model)
    final_norm, classifier = model.mlp_head
    ff_linear1, _relu, _dropout1, ff_linear2, _dropout2 = feed_forward.net

    margin = classifier.weight[predicted_label] - classifier.weight[target_label]
    margin_bias = classifier.bias[predicted_label] - classifier.bias[target_label]
    final_matrix, final_bias = _no_var_affine(final_norm)
    residual_coeff = final_matrix.t().matmul(margin)
    bias = margin_bias + margin.dot(final_bias) + residual_coeff.dot(ff_linear2.bias)

    norm2_matrix, norm2_bias = _no_var_affine(norm2)
    pre_matrix = ff_linear1.weight.matmul(norm2_matrix)
    pre_bias = ff_linear1.weight.matmul(norm2_bias) + ff_linear1.bias
    if pre_lower is None or pre_upper is None:
        h_l = h_lower[0, 0]
        h_u = h_upper[0, 0]
        pre_l = pre_bias + torch.where(pre_matrix >= 0, pre_matrix * h_l, pre_matrix * h_u).sum(dim=1)
        pre_u = pre_bias + torch.where(pre_matrix >= 0, pre_matrix * h_u, pre_matrix * h_l).sum(dim=1)
    else:
        pre_l = pre_lower.reshape(-1, pre_lower.shape[-1])[0]
        pre_u = pre_upper.reshape(-1, pre_upper.shape[-1])[0]

    hidden_coeff = ff_linear2.weight.t().matmul(residual_coeff)
    active = pre_l >= 0
    inactive = pre_u <= 0
    crossing = ~(active | inactive)
    slope = active.to(pre_l.dtype)
    intercept = torch.zeros_like(pre_l)

    positive_crossing = crossing & (hidden_coeff >= 0)
    if slope_policy == "identity":
        slope = torch.where(positive_crossing, torch.ones_like(slope), slope)
    elif slope_policy == "auto":
        slope = torch.where(positive_crossing, (pre_u > -pre_l).to(pre_l.dtype), slope)
    elif slope_policy != "zero":
        raise ValueError(f"Unknown ReLU slope policy: {slope_policy}")

    negative_crossing = crossing & (hidden_coeff < 0)
    denominator = torch.clamp(pre_u - pre_l, min=1e-12)
    slope = torch.where(negative_crossing, pre_u / denominator, slope)
    intercept = torch.where(negative_crossing, -pre_l * pre_u / denominator, intercept)

    coeff_cls = residual_coeff + (hidden_coeff[:, None] * slope[:, None] * pre_matrix).sum(dim=0)
    bias = bias + (hidden_coeff * (slope * pre_bias + intercept)).sum()
    reference = pre_lower if pre_lower is not None else h_lower
    coefficients = torch.zeros(
        model.pos_embedding.shape[0],
        model.dim,
        device=reference.device,
        dtype=reference.dtype,
    )
    coefficients[0] = coeff_cls
    return coefficients, bias


def _attention_residual_lower(
    model,
    z_lower,
    z_upper,
    value_lower,
    value_upper,
    score_lower,
    score_upper,
    coefficients,
    bias,
):
    _norm1, attention, output, _norm2, _feed_forward = _final_components(model)
    batch = z_lower.shape[0]
    total = torch.as_tensor(bias, device=z_lower.device, dtype=z_lower.dtype).reshape(1).expand(batch).clone()
    total = total + torch.where(coefficients >= 0, coefficients * z_lower, coefficients * z_upper).sum((1, 2))
    if isinstance(output, torch.nn.Identity):
        output_direction = coefficients
        output_bias = None
    else:
        output_direction = coefficients.matmul(output.weight)
        output_bias = output.bias
    if output_bias is not None:
        total = total + coefficients.matmul(output_bias).sum()
    head_dim = attention.to_v.out_features // attention.heads
    row_lowers = []
    for head in range(attention.heads):
        start = head * head_dim
        end = (head + 1) * head_dim
        head_direction = output_direction[:, start:end]
        value_weight = attention.to_v.weight[start:end]
        value_direction = head_direction.matmul(value_weight)
        value_coeff = torch.where(
            value_direction[None, :, None, :] >= 0,
            value_direction[None, :, None, :] * value_lower[:, None, :, :],
            value_direction[None, :, None, :] * value_upper[:, None, :, :],
        ).sum(dim=3)
        if attention.to_v.bias is not None:
            value_coeff = value_coeff + head_direction.matmul(attention.to_v.bias[start:end])[None, :, None]
        row_lowers.append(
            softmax_box_expectation_min(
                score_lower[:, head],
                score_upper[:, head],
                value_coeff,
            ).sum(dim=1)
        )
    return total + torch.stack(row_lowers).sum(dim=0)


def vertex_target_margins(model, center, lower, upper, predicted_label):
    common_opts = {"verbosity": 0, "fixed_reducemax_index": True}
    z_l, z_u = _compute_bounds(FinalBlockInputModule(model), center, lower, upper, bound_opts=common_opts)
    value_l, value_u = _compute_bounds(FinalAttentionInputModule(model), center, lower, upper, bound_opts=common_opts)
    score_l, score_u = _compute_bounds(FinalScoreModule(model), center, lower, upper, bound_opts=common_opts)
    h_l, h_u = _compute_bounds(FinalAttentionResidualModule(model), center, lower, upper, bound_opts=common_opts)

    num_classes = model.mlp_head[-1].out_features
    result = torch.full((num_classes,), float("inf"), device=center.device, dtype=center.dtype)
    for target in range(num_classes):
        if target == predicted_label:
            continue
        candidates = []
        for policy in ("auto", "zero", "identity"):
            coefficients, bias = _suffix_coefficients(
                model,
                h_l,
                h_u,
                predicted_label,
                target,
                policy,
            )
            candidates.append(
                _attention_residual_lower(
                    model,
                    z_l,
                    z_u,
                    value_l,
                    value_u,
                    score_l,
                    score_u,
                    coefficients,
                    bias,
                )
            )
        result[target] = torch.stack(candidates).amax().detach()
    return result


def _first_block_input_bounds(model, lower, upper):
    rearrange = model.to_patch_embedding[0]
    linear = model.to_patch_embedding[1]
    patch_l = rearrange(lower)
    patch_u = rearrange(upper)
    weight = linear.weight
    embedded_l = linear.bias + torch.where(weight >= 0, patch_l[:, :, None, :] * weight, patch_u[:, :, None, :] * weight).sum(dim=3)
    embedded_u = linear.bias + torch.where(weight >= 0, patch_u[:, :, None, :] * weight, patch_l[:, :, None, :] * weight).sum(dim=3)
    cls = model.cls_token.expand(lower.shape[0], -1, -1)
    positions = model.pos_embedding[: embedded_l.shape[1] + 1]
    z_l = torch.cat((cls, embedded_l), dim=1) + positions
    z_u = torch.cat((cls, embedded_u), dim=1) + positions
    return z_l, z_u


def _score_tensor(value, heads):
    while value.dim() > 4 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() == 2:
        return value.unsqueeze(0).unsqueeze(0)
    if value.dim() == 3:
        if value.shape[0] == heads:
            return value.unsqueeze(0)
        return value.unsqueeze(1)
    if value.dim() == 4:
        return value
    raise ValueError(f"Unsupported score-bound shape: {tuple(value.shape)}")


def _token_tensor(value):
    while value.dim() > 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() == 2:
        return value.unsqueeze(0)
    if value.dim() == 3:
        return value
    raise ValueError(f"Unsupported token-bound shape: {tuple(value.shape)}")


def _projected_value_tensor(value, heads):
    while value.dim() > 4 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() == 4:
        batch, value_heads, tokens, head_dim = value.shape
        if value_heads != heads:
            raise ValueError(f"Value bounds have {value_heads} heads, expected {heads}")
        return value.transpose(1, 2).reshape(batch, tokens, value_heads * head_dim)
    if value.dim() == 3 and value.shape[0] == heads and heads > 1:
        value_heads, tokens, head_dim = value.shape
        return value.transpose(0, 1).reshape(1, tokens, value_heads * head_dim)
    return _token_tensor(value)


def _centered_block_input_bounds(model, lower, upper):
    norm, _attention, _output, _norm2, _feed_forward = _final_components(model)
    weight = norm.weight.reshape(1, 1, -1)
    bias = norm.bias.reshape(1, 1, -1)
    if bool((weight.abs() < 1e-12).any().item()):
        raise ValueError("Cannot recover centered block input through a zero LayerNorm weight")
    first = (lower - bias) / weight
    second = (upper - bias) / weight
    return torch.minimum(first, second), torch.maximum(first, second)


def _attention_residual_lower_projected(
    model,
    z_lower,
    z_upper,
    value_lower,
    value_upper,
    score_lower,
    score_upper,
    coefficients,
    bias,
):
    _norm1, attention, output, _norm2, _feed_forward = _final_components(model)
    total = torch.as_tensor(bias, device=z_lower.device, dtype=z_lower.dtype).reshape(1)
    total = total + torch.where(coefficients >= 0, coefficients * z_lower, coefficients * z_upper).sum((1, 2))
    if isinstance(output, torch.nn.Identity):
        output_direction = coefficients
        output_bias = None
    else:
        output_direction = coefficients.matmul(output.weight)
        output_bias = output.bias
    if output_bias is not None:
        total = total + coefficients.matmul(output_bias).sum()

    head_dim = attention.to_v.out_features // attention.heads
    row_lowers = []
    for head in range(attention.heads):
        start = head * head_dim
        end = (head + 1) * head_dim
        direction = output_direction[:, start:end]
        v_l = value_lower[:, :, start:end]
        v_u = value_upper[:, :, start:end]
        value_coeff = torch.where(
            direction[None, :, None, :] >= 0,
            direction[None, :, None, :] * v_l[:, None, :, :],
            direction[None, :, None, :] * v_u[:, None, :, :],
        ).sum(dim=3)
        row_lowers.append(
            softmax_box_expectation_min(
                score_lower[:, head],
                score_upper[:, head],
                value_coeff,
            ).sum(dim=1)
        )
    return total + torch.stack(row_lowers).sum(dim=0)


def vertex_target_margins_from_crown(model, lower, upper, predicted_label, crown_bounds):
    if model.depth == 1:
        z_l, z_u = _first_block_input_bounds(model, lower, upper)
    else:
        attention_l = _token_tensor(crown_bounds.attention_input_lower)
        attention_u = _token_tensor(crown_bounds.attention_input_upper)
        z_l, z_u = _centered_block_input_bounds(model, attention_l, attention_u)
    value_l = _projected_value_tensor(crown_bounds.value_lower, model.heads)
    value_u = _projected_value_tensor(crown_bounds.value_upper, model.heads)
    score_l = _score_tensor(crown_bounds.score_lower, model.heads)
    score_u = _score_tensor(crown_bounds.score_upper, model.heads)
    pre_l = _token_tensor(crown_bounds.relu_lower)
    pre_u = _token_tensor(crown_bounds.relu_upper)
    expected_tokens = z_l.shape[1]
    for name, tensor in (("block input", z_l), ("value", value_l), ("relu", pre_l)):
        if tensor.shape[1] != expected_tokens:
            raise ValueError(f"{name} bounds have {tensor.shape[1]} tokens, expected {expected_tokens}")
    if z_l.shape[2] != model.dim:
        raise ValueError(f"block input bounds have width {z_l.shape[2]}, expected {model.dim}")
    if score_l.shape[-2:] != (expected_tokens, expected_tokens):
        raise ValueError(f"score bounds have shape {tuple(score_l.shape)}, expected (*, {expected_tokens}, {expected_tokens})")
    if score_l.shape[1] != model.heads:
        raise ValueError(f"score bounds have {score_l.shape[1]} heads, expected {model.heads}")
    if value_l.shape[2] != model.dim:
        raise ValueError(f"value bounds have width {value_l.shape[2]}, expected {model.dim}")
    if pre_l.shape[2] != model.transformer.layers[-1][1].fn.fn.net[0].out_features:
        raise ValueError("ReLU pre-activation bounds have an unexpected width")

    result = torch.full(
        (model.mlp_head[-1].out_features,),
        float("inf"),
        device=lower.device,
        dtype=lower.dtype,
    )
    for target in range(result.numel()):
        if target == predicted_label:
            continue
        candidates = []
        for policy in ("auto", "zero", "identity"):
            coefficients, bias = _suffix_coefficients(
                model,
                None,
                None,
                predicted_label,
                target,
                policy,
                pre_lower=pre_l,
                pre_upper=pre_u,
            )
            if model.depth > 1:
                coefficient_drift = coefficients.sum(dim=1).abs().max()
                if float(coefficient_drift.detach().cpu()) > 1e-5:
                    raise ValueError(
                        f"Final-block objective is not shift invariant: coefficient sum={coefficient_drift.item()}"
                    )
                coefficients = coefficients - coefficients.mean(dim=1, keepdim=True)
            candidates.append(
                _attention_residual_lower_projected(
                    model,
                    z_l,
                    z_u,
                    value_l,
                    value_u,
                    score_l,
                    score_u,
                    coefficients,
                    bias,
                )
            )
        result[target] = torch.stack(candidates).amax().detach()
    return result


def certify_image(
    model,
    image,
    epsilon,
    predicted_label,
    method,
    *,
    crown_method="CROWN",
    alpha_iters=20,
    crown_bounds=None,
):
    nonzero_dropout = [module.p for module in model.modules() if isinstance(module, torch.nn.Dropout) and module.p]
    if nonzero_dropout:
        raise ValueError("The Vertex adapter requires all dropout probabilities to be zero")
    center = image.unsqueeze(0)
    with torch.no_grad():
        forward_max_abs_diff = float((model(center) - _stable_forward(model, center)).abs().max().cpu())
    if forward_max_abs_diff > 1e-6:
        raise RuntimeError(f"Adapter forward does not match vitveri model: max_abs_diff={forward_max_abs_diff}")
    lower = torch.clamp(center - epsilon, min=-1.0, max=1.0)
    upper = torch.clamp(center + epsilon, min=-1.0, max=1.0)
    started = time.perf_counter()

    crown = None
    vertex = None
    if crown_bounds is None:
        raise ValueError("crown_bounds from the duplicated ABCROWN provider are required")
    if method in {"CROWN", "crown_objective_vertex_hybrid"}:
        crown = (
            crown_bounds.alpha_target_lowers
            if crown_method == "alpha-CROWN"
            else crown_bounds.initial_target_lowers
        )
    if method in {"objective_vertex_crown", "crown_objective_vertex_hybrid"}:
        vertex = vertex_target_margins_from_crown(model, lower, upper, predicted_label, crown_bounds)

    if method == "CROWN":
        selected = crown
    elif method == "objective_vertex_crown":
        selected = vertex
    else:
        selected = torch.maximum(crown, vertex)

    def serialize(values):
        return None if values is None else [float(value) for value in values.detach().cpu()]

    finite = torch.cat((selected[:predicted_label], selected[predicted_label + 1 :]))
    return {
        "crown_target_lowers": serialize(crown),
        "vertex_target_lowers": serialize(vertex),
        "target_lowers": serialize(selected),
        "minimum_lower": float(finite.min().detach().cpu()),
        "certified": bool((finite > 0).all().item()),
        "elapsed_sec": time.perf_counter() - started,
        "forward_max_abs_diff": forward_max_abs_diff,
    }
