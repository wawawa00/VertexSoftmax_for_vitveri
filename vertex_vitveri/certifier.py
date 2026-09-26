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
    return model.dropout(x + model.pos_embedding[: token_count + 1])


def _final_block_input(model, image):
    x = _patch_tokens(model, image)
    for attention, feed_forward in model.transformer.layers[:-1]:
        x = attention(x)
        x = feed_forward(x)
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
    norm1, attention, output, _norm2, _feed_forward = _final_components(model)
    attention_input = norm1(z)
    q = _split_heads(attention.to_q(attention_input), attention.heads)
    k = _split_heads(attention.to_k(attention_input), attention.heads)
    v = _split_heads(attention.to_v(attention_input), attention.heads)
    scores = q.matmul(k.transpose(-1, -2)) * attention.scale
    probabilities = torch.softmax(scores, dim=-1)
    context = probabilities.matmul(v).transpose(1, 2).reshape(z.shape)
    residual = z + output(context)
    return z, attention_input, scores, residual


class MarginModule(torch.nn.Module):
    def __init__(self, model, predicted_label):
        super().__init__()
        self.model = model
        self.predicted_label = int(predicted_label)

    def forward(self, image):
        logits = self.model(image)
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


def _suffix_coefficients(model, h_lower, h_upper, predicted_label, target_label, slope_policy):
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
    h_l = h_lower[0, 0]
    h_u = h_upper[0, 0]
    pre_l = pre_bias + torch.where(pre_matrix >= 0, pre_matrix * h_l, pre_matrix * h_u).sum(dim=1)
    pre_u = pre_bias + torch.where(pre_matrix >= 0, pre_matrix * h_u, pre_matrix * h_l).sum(dim=1)

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
    coefficients = torch.zeros(
        h_lower.shape[1],
        h_lower.shape[2],
        device=h_lower.device,
        dtype=h_lower.dtype,
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


def certify_image(model, image, epsilon, predicted_label, method, *, crown_method="CROWN", alpha_iters=20):
    center = image.unsqueeze(0)
    lower = torch.clamp(center - epsilon, min=-1.0, max=1.0)
    upper = torch.clamp(center + epsilon, min=-1.0, max=1.0)
    started = time.perf_counter()

    crown = None
    vertex = None
    if method in {"CROWN", "crown_objective_vertex_hybrid"}:
        crown = crown_target_margins(
            model,
            center,
            lower,
            upper,
            predicted_label,
            crown_method,
            alpha_iters,
        )
    if method in {"objective_vertex_crown", "crown_objective_vertex_hybrid"}:
        vertex = vertex_target_margins(model, center, lower, upper, predicted_label)

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
    }
