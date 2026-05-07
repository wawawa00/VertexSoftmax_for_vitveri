#!/usr/bin/env python3
"""Explore certificate methods for attention verification.

The main candidate method here is `vertex_softmax`: after bounding attention
score intervals, it computes exact row-wise extrema of a linear function through
softmax over the score box. The default implementation uses the exact threshold
solver, reducing the old all-vertices search from O(2^K) to O(K log K) per row.
This preserves the softmax simplex coupling that vanilla IBP loses.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Result:
    setting: str
    method: str
    epsilon: float
    trials: int
    certified_rate: float
    attack_robust_rate: float
    mean_nominal_margin: float
    mean_cert_lower: float
    mean_attack_margin: float
    mean_gap_attack_minus_cert: float
    elapsed_sec: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_params(
    batch: int,
    d_in: int,
    d_head: int,
    device: torch.device,
    weight_scale: float,
    out_scale: float,
) -> dict[str, torch.Tensor]:
    std = weight_scale / math.sqrt(d_in)
    return {
        "wq": torch.randn(batch, d_in, d_head, device=device) * std,
        "wk": torch.randn(batch, d_in, d_head, device=device) * std,
        "wv": torch.randn(batch, d_in, d_head, device=device) * std,
        "wo": torch.randn(batch, d_head, device=device) * out_scale / math.sqrt(d_head),
    }


def batched_attention_margin(x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = 1.0 / math.sqrt(params["wq"].shape[-1])
    q = torch.einsum("bld,bdh->blh", x, params["wq"])
    k = torch.einsum("bld,bdh->blh", x, params["wk"])
    v = torch.einsum("bld,bdh->blh", x, params["wv"])
    a = torch.softmax(torch.einsum("blh,bmh->blm", q, k) * scale, dim=-1)
    h = torch.einsum("blm,bmh->blh", a, v)
    return torch.einsum("bh,bh->b", h.mean(dim=1), params["wo"])


def affine_interval(
    lower: torch.Tensor, upper: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    w_pos = torch.clamp(weight, min=0)
    w_neg = torch.clamp(weight, max=0)
    out_l = torch.einsum("bld,bdh->blh", lower, w_pos) + torch.einsum(
        "bld,bdh->blh", upper, w_neg
    )
    out_u = torch.einsum("bld,bdh->blh", upper, w_pos) + torch.einsum(
        "bld,bdh->blh", lower, w_neg
    )
    return out_l, out_u


def product_interval(
    a_l: torch.Tensor, a_u: torch.Tensor, b_l: torch.Tensor, b_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.stack((a_l * b_l, a_l * b_u, a_u * b_l, a_u * b_u), dim=0)
    return candidates.min(dim=0).values, candidates.max(dim=0).values


def qkv_score_intervals(
    x_l: torch.Tensor, x_u: torch.Tensor, params: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_l, q_u = affine_interval(x_l, x_u, params["wq"])
    k_l, k_u = affine_interval(x_l, x_u, params["wk"])
    v_l, v_u = affine_interval(x_l, x_u, params["wv"])
    score_l = torch.zeros(
        x_l.shape[0], x_l.shape[1], x_l.shape[1], device=x_l.device, dtype=x_l.dtype
    )
    score_u = torch.zeros_like(score_l)
    scale = 1.0 / math.sqrt(params["wq"].shape[-1])
    for h in range(params["wq"].shape[-1]):
        p_l, p_u = product_interval(
            q_l[:, :, h].unsqueeze(2),
            q_u[:, :, h].unsqueeze(2),
            k_l[:, :, h].unsqueeze(1),
            k_u[:, :, h].unsqueeze(1),
        )
        score_l = score_l + p_l * scale
        score_u = score_u + p_u * scale
    return q_l, q_u, k_l, k_u, v_l, v_u, score_l, score_u


def output_margin_lower(
    h_l: torch.Tensor, h_u: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]
) -> torch.Tensor:
    pooled_l = h_l.mean(dim=1)
    pooled_u = h_u.mean(dim=1)
    signed_wo = params["wo"] * y.unsqueeze(1)
    w_pos = torch.clamp(signed_wo, min=0)
    w_neg = torch.clamp(signed_wo, max=0)
    return (pooled_l * w_pos + pooled_u * w_neg).sum(dim=1)


def cert_ibp(
    x0: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor], eps: float
) -> torch.Tensor:
    x_l, x_u = x0 - eps, x0 + eps
    *_unused, v_l, v_u, score_l, score_u = qkv_score_intervals(x_l, x_u, params)
    exp_l = torch.exp(torch.clamp(score_l, min=-30.0, max=30.0))
    exp_u = torch.exp(torch.clamp(score_u, min=-30.0, max=30.0))
    sum_exp_l = exp_l.sum(dim=-1, keepdim=True)
    sum_exp_u = exp_u.sum(dim=-1, keepdim=True)
    a_l = exp_l / (exp_l + sum_exp_u - exp_u)
    a_u = exp_u / (exp_u + sum_exp_l - exp_l)
    h_l = torch.zeros_like(v_l)
    h_u = torch.zeros_like(v_u)
    for token in range(x0.shape[1]):
        p_l, p_u = product_interval(
            a_l[:, :, token].unsqueeze(-1),
            a_u[:, :, token].unsqueeze(-1),
            v_l[:, token, :].unsqueeze(1),
            v_u[:, token, :].unsqueeze(1),
        )
        h_l = h_l + p_l
        h_u = h_u + p_u
    return output_margin_lower(h_l, h_u, y, params)


def cert_simplex(
    x0: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor], eps: float
) -> torch.Tensor:
    x_l, x_u = x0 - eps, x0 + eps
    *_unused, v_l, v_u, _score_l, _score_u = qkv_score_intervals(x_l, x_u, params)
    # For any attention distribution in the simplex, each component of a convex
    # combination lies between the min lower and max upper value component.
    h_l = v_l.min(dim=1, keepdim=True).values.expand_as(v_l)
    h_u = v_u.max(dim=1, keepdim=True).values.expand_as(v_u)
    return output_margin_lower(h_l, h_u, y, params)


def softmax_lp_expectation_lower(
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    exp_l = _safe_exp(score_l)
    exp_u = _safe_exp(score_u)
    denom_l = exp_l + (exp_u.sum(dim=-1, keepdim=True) - exp_u)
    denom_u = exp_u + (exp_l.sum(dim=-1, keepdim=True) - exp_l)
    p_l = exp_l / denom_l
    p_u = exp_u / denom_u
    c = coeff[:, None, :, :].expand(score_l.shape[0], score_l.shape[1], coeff.shape[1], coeff.shape[2])
    p_l_h = p_l[:, :, :, None]
    p_u_h = p_u[:, :, :, None]
    base = (p_l_h * c).sum(dim=2)
    remaining = torch.clamp(1.0 - p_l.sum(dim=2), min=0.0)
    capacity = torch.clamp(p_u_h - p_l_h, min=0.0).expand_as(c)
    order = torch.argsort(c, dim=2)
    sorted_c = torch.gather(c, 2, order)
    sorted_capacity = torch.gather(capacity, 2, order)
    used = torch.zeros_like(base)
    extra = torch.zeros_like(base)
    for key in range(score_l.shape[-1]):
        take = torch.minimum(sorted_capacity[:, :, key, :], remaining[:, :, None] - used)
        take = torch.clamp(take, min=0.0)
        extra = extra + take * sorted_c[:, :, key, :]
        used = used + take
    return base + extra


def cert_softmax_lp(
    x0: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor], eps: float
) -> torch.Tensor:
    x_l, x_u = x0 - eps, x0 + eps
    *_unused, v_l, v_u, score_l, score_u = qkv_score_intervals(x_l, x_u, params)
    signed_wo = params["wo"] * y.unsqueeze(1)
    coeff = torch.where(
        signed_wo[:, None, :] >= 0,
        signed_wo[:, None, :] * v_l,
        signed_wo[:, None, :] * v_u,
    )
    per_query_head = softmax_lp_expectation_lower(score_l, score_u, coeff)
    return per_query_head.sum(dim=(1, 2)) / x0.shape[1]


def vertex_masks(length: int, device: torch.device) -> torch.Tensor:
    masks = list(itertools.product([False, True], repeat=length))
    return torch.tensor(masks, dtype=torch.bool, device=device)


def softmax_box_expectation_bounds_exhaustive(
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    coeff_l: torch.Tensor,
    coeff_u: torch.Tensor,
    masks: torch.Tensor,
    vertex_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # score_l/score_u: B x L_query x L_key
    # coeff_*: B x L_key x H
    batch, queries, keys = score_l.shape
    d_head = coeff_l.shape[-1]
    lower = torch.full((batch, queries, d_head), float("inf"), device=score_l.device)
    upper = torch.full((batch, queries, d_head), -float("inf"), device=score_l.device)
    for start in range(0, masks.shape[0], vertex_chunk):
        mask = masks[start : start + vertex_chunk]
        scores = torch.where(mask.view(1, 1, -1, keys), score_u.unsqueeze(2), score_l.unsqueeze(2))
        probs = torch.softmax(scores, dim=-1)
        low_vals = torch.einsum("bivj,bjh->bivh", probs, coeff_l).amin(dim=2)
        high_vals = torch.einsum("bivj,bjh->bivh", probs, coeff_u).amax(dim=2)
        lower = torch.minimum(lower, low_vals)
        upper = torch.maximum(upper, high_vals)
    return lower, upper


def softmax_box_expectation_min(
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    """Exact min of softmax(score)^T coeff over independent score boxes.

    score_l/score_u: B x Q x K
    coeff: B x K x H

    Returns B x Q x H. This is the threshold solver for the equivalent bounded
    linear-fractional problem in y_j=exp(score_j).
    """
    batch, queries, keys = score_l.shape
    heads = coeff.shape[-1]
    shift = score_u.amax(dim=-1, keepdim=True)
    y_l = torch.exp(score_l - shift)
    y_u = torch.exp(score_u - shift)

    order = coeff.argsort(dim=1)
    order_q = order[:, None, :, :].expand(batch, queries, keys, heads)
    c_sorted = coeff.gather(dim=1, index=order)[:, None, :, :]
    y_l_sorted = y_l[:, :, :, None].expand(batch, queries, keys, heads).gather(
        dim=2, index=order_q
    )
    y_u_sorted = y_u[:, :, :, None].expand(batch, queries, keys, heads).gather(
        dim=2, index=order_q
    )

    zero = torch.zeros(batch, queries, 1, heads, device=score_l.device, dtype=score_l.dtype)
    prefix_num_u = torch.cat((zero, (c_sorted * y_u_sorted).cumsum(dim=2)), dim=2)
    prefix_den_u = torch.cat((zero, y_u_sorted.cumsum(dim=2)), dim=2)
    prefix_num_l = torch.cat((zero, (c_sorted * y_l_sorted).cumsum(dim=2)), dim=2)
    prefix_den_l = torch.cat((zero, y_l_sorted.cumsum(dim=2)), dim=2)

    total_num_l = prefix_num_l[:, :, -1:, :]
    total_den_l = prefix_den_l[:, :, -1:, :]
    suffix_num_l = total_num_l - prefix_num_l
    suffix_den_l = total_den_l - prefix_den_l

    values = (prefix_num_u + suffix_num_l) / (prefix_den_u + suffix_den_l)
    return values.amin(dim=2)


def softmax_box_expectation_bounds(
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    coeff_l: torch.Tensor,
    coeff_u: torch.Tensor,
    masks: torch.Tensor | None = None,
    vertex_chunk: int = 64,
    method: str = "threshold",
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "threshold":
        lower = softmax_box_expectation_min(score_l, score_u, coeff_l)
        upper = -softmax_box_expectation_min(score_l, score_u, -coeff_u)
        return lower, upper
    if method == "exhaustive":
        if masks is None:
            masks = vertex_masks(score_l.shape[-1], score_l.device)
        return softmax_box_expectation_bounds_exhaustive(
            score_l, score_u, coeff_l, coeff_u, masks, vertex_chunk
        )
    raise ValueError(f"unknown softmax box method {method}")


def _minimize_affine_on_box(
    coeff: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return bias + torch.where(coeff >= 0, coeff * lower.unsqueeze(-1), coeff * upper.unsqueeze(-1)).sum(dim=2)


def _combine_component_relaxations(
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    coeff: torch.Tensor,
    component_bounds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    batch, queries, keys = score_l.shape
    heads = coeff.shape[-1]
    total_a = torch.zeros(batch, queries, keys, heads, device=score_l.device, dtype=score_l.dtype)
    total_b = torch.zeros(batch, queries, heads, device=score_l.device, dtype=score_l.dtype)
    for key, (a_l, b_l, a_u, b_u) in enumerate(component_bounds):
        c = coeff[:, key, :]
        use_lower = c >= 0
        selected_a = torch.where(
            use_lower[:, None, None, :],
            a_l[:, :, :, None],
            a_u[:, :, :, None],
        )
        selected_b = torch.where(use_lower[:, None, :], b_l[:, :, None], b_u[:, :, None])
        total_a = total_a + selected_a * c[:, None, None, :]
        total_b = total_b + selected_b * c[:, None, :]
    return _minimize_affine_on_box(total_a, score_l, score_u, total_b)


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, min=-30.0, max=30.0))


def wei_lse_component_bounds(
    score_l: torch.Tensor, score_u: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Linearized Wei et al. LSE softmax bounds for every component.

    Each tuple contains affine lower and upper bounds on softmax(score)[j]:
    (a_lower, b_lower, a_upper, b_upper), where a_* has shape B x Q x K.
    """
    mid = (score_l + score_u) / 2
    exp_l = _safe_exp(score_l)
    exp_u = _safe_exp(score_u)
    width = score_u - score_l
    slope = torch.where(width.abs() > 1e-8, (exp_u - exp_l) / width, exp_l)
    intercept = exp_l - slope * score_l
    chord_sum = (slope * mid + intercept).sum(dim=-1)
    soft_mid = torch.softmax(mid, dim=-1)
    lse_mid = torch.logsumexp(mid, dim=-1)

    bounds = []
    keys = score_l.shape[-1]
    for key in range(keys):
        exp_key = _safe_exp(mid[:, :, key])
        lower_val = exp_key / chord_sum
        grad_l = -exp_key[:, :, None] * slope / (chord_sum[:, :, None] ** 2)
        grad_l[:, :, key] = grad_l[:, :, key] + exp_key / chord_sum
        b_l = lower_val - (grad_l * mid).sum(dim=-1)

        denominator_min = exp_l[:, :, key] + (exp_u.sum(dim=-1) - exp_u[:, :, key])
        denominator_max = exp_u[:, :, key] + (exp_l.sum(dim=-1) - exp_l[:, :, key])
        p_min = torch.clamp(exp_l[:, :, key] / denominator_min, min=1e-12)
        p_max = torch.clamp(exp_u[:, :, key] / denominator_max, min=1e-12)
        log_min = torch.log(p_min)
        log_max = torch.log(p_max)
        denom = log_max - log_min
        chord_slope = torch.where(
            denom.abs() > 1e-8,
            (p_max - p_min) / denom,
            p_max,
        )
        z_mid = mid[:, :, key] - lse_mid
        upper_val = chord_slope * (z_mid - log_min) + p_min
        grad_u = -chord_slope[:, :, None] * soft_mid
        grad_u[:, :, key] = grad_u[:, :, key] + chord_slope
        b_u = upper_val - (grad_u * mid).sum(dim=-1)
        bounds.append((grad_l, b_l, grad_u, b_u))
    return bounds


def galileo_component_bounds(
    score_l: torch.Tensor, score_u: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """GaLileo-style n-dimensional monotone linear softmax bounds."""
    batch, queries, keys = score_l.shape
    grid = torch.linspace(
        1.0 / 33.0,
        1.0,
        33,
        device=score_l.device,
        dtype=score_l.dtype,
    )
    bounds = []
    for target in range(keys):
        z_l = -score_u.clone()
        z_u = -score_l.clone()
        z_l[:, :, target] = score_l[:, :, target]
        z_u[:, :, target] = score_u[:, :, target]

        def z_to_score(z: torch.Tensor) -> torch.Tensor:
            score = -z
            score = score.clone()
            score[:, :, target] = z[:, :, target]
            return score

        p_at_l = torch.softmax(z_to_score(z_l), dim=-1)
        p_at_u = torch.softmax(z_to_score(z_u), dim=-1)
        f_l = p_at_l[:, :, target]
        f_u = p_at_u[:, :, target]

        a_l_z = torch.empty_like(z_l)
        a_u_z = torch.empty_like(z_l)
        widths = torch.clamp(z_u - z_l, min=1e-8)
        for dim in range(keys):
            lower_avgs = []
            upper_avgs = []
            for frac in grid:
                z_lower_edge = z_l.clone()
                z_lower_edge[:, :, dim] = z_l[:, :, dim] + frac * widths[:, :, dim]
                f_lower_edge = torch.softmax(z_to_score(z_lower_edge), dim=-1)[:, :, target]
                lower_avgs.append((f_lower_edge - f_l) / (frac * widths[:, :, dim]))

                z_upper_edge = z_u.clone()
                z_upper_edge[:, :, dim] = z_u[:, :, dim] - frac * widths[:, :, dim]
                f_upper_edge = torch.softmax(z_to_score(z_upper_edge), dim=-1)[:, :, target]
                upper_avgs.append((f_u - f_upper_edge) / (frac * widths[:, :, dim]))
            a_l_z[:, :, dim] = torch.stack(lower_avgs, dim=0).amin(dim=0)
            a_u_z[:, :, dim] = torch.stack(upper_avgs, dim=0).amin(dim=0)

        a_l_z = torch.clamp(a_l_z - 1e-7, min=0.0)
        a_u_z = torch.clamp(a_u_z - 1e-7, min=0.0)
        b_l = f_l - (a_l_z * z_l).sum(dim=-1)
        b_u = f_u - (a_u_z * z_u).sum(dim=-1)

        signs = -torch.ones(keys, device=score_l.device, dtype=score_l.dtype)
        signs[target] = 1
        a_l = a_l_z * signs
        a_u = a_u_z * signs
        bounds.append((a_l, b_l, a_u, b_u))
    return bounds


def linear_softmax_expectation_lower(
    score_l: torch.Tensor,
    score_u: torch.Tensor,
    coeff: torch.Tensor,
    relaxation: str,
) -> torch.Tensor:
    if relaxation == "wei_lse":
        component_bounds = wei_lse_component_bounds(score_l, score_u)
    elif relaxation == "galileo":
        component_bounds = galileo_component_bounds(score_l, score_u)
    else:
        raise ValueError(f"unknown linear relaxation {relaxation}")
    return _combine_component_relaxations(score_l, score_u, coeff, component_bounds)


def cert_linear_softmax(
    x0: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, torch.Tensor],
    eps: float,
    relaxation: str,
    split_dims: int,
) -> torch.Tensor:
    branches: list[tuple[torch.Tensor, torch.Tensor]] = [(x0 - eps, x0 + eps)]
    flat_dim = x0.shape[1] * x0.shape[2]
    for idx in range(split_dims):
        dim = idx % flat_dim
        token = dim // x0.shape[2]
        feat = dim % x0.shape[2]
        new_branches = []
        for low, high in branches:
            mid = (low[:, token, feat] + high[:, token, feat]) / 2
            low_a, high_a = low.clone(), high.clone()
            high_a[:, token, feat] = mid
            low_b, high_b = low.clone(), high.clone()
            low_b[:, token, feat] = mid
            new_branches.extend([(low_a, high_a), (low_b, high_b)])
        branches = new_branches

    branch_lbs = []
    for x_l, x_u in branches:
        *_unused, v_l, v_u, score_l, score_u = qkv_score_intervals(x_l, x_u, params)
        signed_wo = params["wo"] * y.unsqueeze(1)
        coeff = torch.where(
            signed_wo[:, None, :] >= 0,
            signed_wo[:, None, :] * v_l,
            signed_wo[:, None, :] * v_u,
        )
        per_query_head = linear_softmax_expectation_lower(score_l, score_u, coeff, relaxation)
        branch_lbs.append(per_query_head.sum(dim=(1, 2)) / x0.shape[1])
    return torch.stack(branch_lbs, dim=0).min(dim=0).values


def cert_vertex_softmax(
    x0: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, torch.Tensor],
    eps: float,
    masks: torch.Tensor | None,
    vertex_chunk: int,
    split_dims: int,
) -> torch.Tensor:
    branches: list[tuple[torch.Tensor, torch.Tensor]] = [(x0 - eps, x0 + eps)]
    flat_dim = x0.shape[1] * x0.shape[2]
    for idx in range(split_dims):
        dim = idx % flat_dim
        token = dim // x0.shape[2]
        feat = dim % x0.shape[2]
        new_branches = []
        for low, high in branches:
            mid = (low[:, token, feat] + high[:, token, feat]) / 2
            low_a, high_a = low.clone(), high.clone()
            high_a[:, token, feat] = mid
            low_b, high_b = low.clone(), high.clone()
            low_b[:, token, feat] = mid
            new_branches.extend([(low_a, high_a), (low_b, high_b)])
        branches = new_branches

    branch_lbs = []
    for x_l, x_u in branches:
        *_unused, v_l, v_u, score_l, score_u = qkv_score_intervals(x_l, x_u, params)
        h_l, h_u = softmax_box_expectation_bounds(score_l, score_u, v_l, v_u, masks, vertex_chunk)
        branch_lbs.append(output_margin_lower(h_l, h_u, y, params))
    return torch.stack(branch_lbs, dim=0).min(dim=0).values


def pgd_attack_margin(
    x0: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, torch.Tensor],
    eps: float,
    steps: int,
    restarts: int,
    step_size: float,
) -> torch.Tensor:
    best_margin = torch.full((x0.shape[0],), float("inf"), device=x0.device)
    for _ in range(restarts):
        delta = torch.empty_like(x0).uniform_(-eps, eps)
        for _step in range(steps):
            delta.requires_grad_(True)
            signed_margin = y * batched_attention_margin(x0 + delta, params)
            grad = torch.autograd.grad(signed_margin.sum(), delta)[0]
            delta = (delta - step_size * grad.sign()).detach()
            delta = torch.clamp(delta, min=-eps, max=eps)
        with torch.no_grad():
            signed_margin = y * batched_attention_margin(x0 + delta, params)
            best_margin = torch.minimum(best_margin, signed_margin)
    return best_margin


def run_setting(args: argparse.Namespace, device: torch.device, setting: str) -> list[Result]:
    methods = args.methods
    masks = None
    all_results: list[Result] = []
    for eps in args.epsilons:
        accum = {
            method: {
                "certified": 0,
                "cert_lowers": [],
                "attack_margins": [],
                "nominal_margins": [],
                "seen": 0,
                "elapsed": 0.0,
            }
            for method in methods
        }
        for _ in range(math.ceil(args.trials / args.batch)):
            batch = min(args.batch, args.trials - accum[methods[0]]["seen"])
            if batch <= 0:
                break
            x0 = torch.randn(batch, args.seq_len, args.d_in, device=device)
            params = random_params(batch, args.d_in, args.d_head, device, args.weight_scale, args.out_scale)
            with torch.no_grad():
                nominal = batched_attention_margin(x0, params)
                y = torch.where(nominal >= 0, 1.0, -1.0)
                nominal_margin = y * nominal
            attack_margin = pgd_attack_margin(
                x0, y, params, eps, args.pgd_steps, args.pgd_restarts, args.pgd_step_size
            ).detach()
            for method in methods:
                method_start = time.time()
                if method == "ibp":
                    with torch.no_grad():
                        cert = cert_ibp(x0, y, params, eps)
                elif method == "simplex":
                    with torch.no_grad():
                        cert = cert_simplex(x0, y, params, eps)
                elif method == "softmax_lp":
                    with torch.no_grad():
                        cert = cert_softmax_lp(x0, y, params, eps)
                elif method == "vertex":
                    with torch.no_grad():
                        cert = cert_vertex_softmax(
                            x0, y, params, eps, masks, args.vertex_chunk, split_dims=0
                        )
                elif method.startswith("vertex_split"):
                    split_dims = int(method.replace("vertex_split", ""))
                    with torch.no_grad():
                        cert = cert_vertex_softmax(
                            x0, y, params, eps, masks, args.vertex_chunk, split_dims=split_dims
                        )
                elif method in {"wei_lse", "galileo"}:
                    with torch.no_grad():
                        cert = cert_linear_softmax(
                            x0, y, params, eps, relaxation=method, split_dims=0
                        )
                elif method.startswith("wei_lse_split"):
                    split_dims = int(method.replace("wei_lse_split", ""))
                    with torch.no_grad():
                        cert = cert_linear_softmax(
                            x0, y, params, eps, relaxation="wei_lse", split_dims=split_dims
                        )
                elif method.startswith("galileo_split"):
                    split_dims = int(method.replace("galileo_split", ""))
                    with torch.no_grad():
                        cert = cert_linear_softmax(
                            x0, y, params, eps, relaxation="galileo", split_dims=split_dims
                        )
                else:
                    raise ValueError(f"unknown method {method}")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                accum[method]["elapsed"] += time.time() - method_start
                accum[method]["certified"] += (cert > 0).sum().item()
                accum[method]["cert_lowers"].append(cert.detach().cpu())
                accum[method]["attack_margins"].append(attack_margin.detach().cpu())
                accum[method]["nominal_margins"].append(nominal_margin.detach().cpu())
                accum[method]["seen"] += batch
        for method in methods:
            info = accum[method]
            certs = torch.cat(info["cert_lowers"])
            attacks = torch.cat(info["attack_margins"])
            nominals = torch.cat(info["nominal_margins"])
            all_results.append(
                Result(
                    setting=setting,
                    method=method,
                    epsilon=eps,
                    trials=info["seen"],
                    certified_rate=info["certified"] / info["seen"],
                    attack_robust_rate=(attacks > 0).float().mean().item(),
                    mean_nominal_margin=nominals.mean().item(),
                    mean_cert_lower=certs.mean().item(),
                    mean_attack_margin=attacks.mean().item(),
                    mean_gap_attack_minus_cert=(attacks - certs).mean().item(),
                    elapsed_sec=info["elapsed"],
                )
            )
        print(f"finished eps={eps}", flush=True)
        for row in all_results[-len(methods) :]:
            print(row, flush=True)
    return all_results


def write_results(path: Path, results: list[Result], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    config = vars(args).copy()
    config["out"] = str(config["out"])
    path.with_suffix(".json").write_text(
        json.dumps({"config": config, "results": [asdict(r) for r in results]}, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--d-in", type=int, required=True)
    parser.add_argument("--d-head", type=int, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.005, 0.01, 0.03, 0.05])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ibp", "simplex", "vertex", "vertex_split1"],
    )
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--out-scale", type=float, default=1.0)
    parser.add_argument("--vertex-chunk", type=int, default=64)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--pgd-restarts", type=int, default=2)
    parser.add_argument("--pgd-step-size", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setting = f"s{args.seq_len}_din{args.d_in}_dh{args.d_head}_ws{args.weight_scale}"
    print(
        f"device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} setting={setting}",
        flush=True,
    )
    results = run_setting(args, device, setting)
    write_results(args.out, results, args)


if __name__ == "__main__":
    main()
