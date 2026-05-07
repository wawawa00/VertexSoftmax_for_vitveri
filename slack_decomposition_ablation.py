#!/usr/bin/env python3
"""Controlled slack-decomposition diagnostics for Vertex-Softmax.

This script is intentionally small and synthetic. It does not create new
headline robustness numbers. Instead, it quantifies the separate relaxation
losses named in the paper's limitations section on tiny instances where a dense
two-dimensional grid can approximate the reachable-set optimum.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclass
class DetailRow:
    diagnostic: str
    seed: int
    epsilon: float
    softmax_gap: float | None
    score_box_gap: float | None
    value_gap: float | None
    rowwise_gap: float | None
    suffix_gap: float | None
    final_gap_to_grid: float
    certificate_lower: float
    grid_reference: float


@dataclass
class SummaryRow:
    diagnostic: str
    epsilon: float
    seeds: int
    softmax_gap_mean: float | None
    softmax_gap_sem: float | None
    score_box_gap_mean: float | None
    score_box_gap_sem: float | None
    value_gap_mean: float | None
    value_gap_sem: float | None
    rowwise_gap_mean: float | None
    rowwise_gap_sem: float | None
    suffix_gap_mean: float | None
    suffix_gap_sem: float | None
    final_gap_to_grid_mean: float
    final_gap_to_grid_sem: float


def randn(shape: tuple[int, ...], rng: np.random.Generator, scale: float = 1.0) -> Array:
    return rng.normal(0.0, scale, size=shape)


def dense_grid(eps: float, points: int) -> Array:
    axis = np.linspace(-eps, eps, points, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)


def softmax(scores: Array, axis: int = -1) -> Array:
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


def logsumexp(values: Array) -> float:
    shift = float(np.max(values))
    return shift + math.log(float(np.exp(values - shift).sum()))


def affine_values(x: Array, weight: Array, bias: Array) -> Array:
    return x @ weight.T + bias


def affine_interval(weight: Array, bias: Array, lower: Array, upper: Array) -> tuple[Array, Array]:
    w_pos = np.maximum(weight, 0.0)
    w_neg = np.minimum(weight, 0.0)
    out_l = bias + np.sum(w_pos * lower + w_neg * upper, axis=-1)
    out_u = bias + np.sum(w_pos * upper + w_neg * lower, axis=-1)
    return out_l, out_u


def vertex_box_min(score_l: Array, score_u: Array, coeff: Array) -> float:
    """Exact min of coeff^T softmax(score) over an independent score box."""
    shift = float(np.max(score_u))
    y_l = np.exp(score_l - shift)
    y_u = np.exp(score_u - shift)
    order = np.argsort(coeff)
    c = coeff[order]
    yl = y_l[order]
    yu = y_u[order]

    prefix_num_u = np.concatenate(([0.0], np.cumsum(c * yu)))
    prefix_den_u = np.concatenate(([0.0], np.cumsum(yu)))
    prefix_num_l = np.concatenate(([0.0], np.cumsum(c * yl)))
    prefix_den_l = np.concatenate(([0.0], np.cumsum(yl)))
    suffix_num_l = prefix_num_l[-1] - prefix_num_l
    suffix_den_l = prefix_den_l[-1] - prefix_den_l
    values = (prefix_num_u + suffix_num_l) / (prefix_den_u + suffix_den_l)
    return float(np.min(values))


def _safe_exp(values: Array) -> Array:
    return np.exp(np.clip(values, -30.0, 30.0))


def wei_lse_box_lower(score_l: Array, score_u: Array, coeff: Array) -> float:
    """Wei-LSE-style component relaxation, specialized to one score row."""
    mid = (score_l + score_u) / 2.0
    exp_l = _safe_exp(score_l)
    exp_u = _safe_exp(score_u)
    width = score_u - score_l
    slope = np.where(np.abs(width) > 1e-8, (exp_u - exp_l) / width, exp_l)
    intercept = exp_l - slope * score_l
    chord_sum = float(np.sum(slope * mid + intercept))
    soft_mid = softmax(mid)
    lse_mid = logsumexp(mid)

    total_a = np.zeros_like(score_l)
    total_b = 0.0
    for key in range(score_l.shape[0]):
        exp_key = float(_safe_exp(np.array([mid[key]]))[0])
        lower_val = exp_key / chord_sum
        grad_l = -exp_key * slope / (chord_sum**2)
        grad_l[key] += exp_key / chord_sum
        b_l = lower_val - float(np.dot(grad_l, mid))

        denominator_min = exp_l[key] + (float(exp_u.sum()) - exp_u[key])
        denominator_max = exp_u[key] + (float(exp_l.sum()) - exp_l[key])
        p_min = max(float(exp_l[key] / denominator_min), 1e-12)
        p_max = max(float(exp_u[key] / denominator_max), 1e-12)
        log_min = math.log(p_min)
        log_max = math.log(p_max)
        denom = log_max - log_min
        chord_slope = (p_max - p_min) / denom if abs(denom) > 1e-8 else p_max
        z_mid = mid[key] - lse_mid
        upper_val = chord_slope * (z_mid - log_min) + p_min
        grad_u = -chord_slope * soft_mid
        grad_u[key] += chord_slope
        b_u = upper_val - float(np.dot(grad_u, mid))

        if coeff[key] >= 0:
            total_a += coeff[key] * grad_l
            total_b += coeff[key] * b_l
        else:
            total_a += coeff[key] * grad_u
            total_b += coeff[key] * b_u

    endpoint = np.where(total_a >= 0, score_l, score_u)
    return float(total_b + np.dot(total_a, endpoint))


def single_row_attention(seed: int, eps: float, grid: Array) -> DetailRow:
    rng = np.random.default_rng(10_000 + seed)
    dim = 2
    keys = 4
    x_l = np.full((dim,), -eps)
    x_u = np.full((dim,), eps)

    score_w = randn((keys, dim), rng, 1.25)
    score_b = randn((keys,), rng, 0.25)
    value_w = randn((keys, dim), rng, 0.75)
    value_b = randn((keys,), rng, 0.35)

    score_l, score_u = affine_interval(score_w, score_b, x_l, x_u)
    value_l, _value_u = affine_interval(value_w, value_b, x_l, x_u)

    scores = affine_values(grid, score_w, score_b)
    values = affine_values(grid, value_w, value_b)
    probs = softmax(scores, axis=1)

    grid_joint = float(np.min(np.sum(probs * values, axis=1)))
    grid_fixed_value = float(np.min(np.sum(probs * value_l, axis=1)))
    vertex = vertex_box_min(score_l, score_u, value_l)
    wei = wei_lse_box_lower(score_l, score_u, value_l)

    return DetailRow(
        diagnostic="single-row attention",
        seed=seed,
        epsilon=eps,
        softmax_gap=vertex - wei,
        score_box_gap=grid_fixed_value - vertex,
        value_gap=grid_joint - grid_fixed_value,
        rowwise_gap=None,
        suffix_gap=None,
        final_gap_to_grid=grid_joint - vertex,
        certificate_lower=vertex,
        grid_reference=grid_joint,
    )


def two_row_residual(seed: int, eps: float, grid: Array) -> DetailRow:
    rng = np.random.default_rng(20_000 + seed)
    dim = 2
    rows = 3
    keys = 4
    x_l = np.full((dim,), -eps)
    x_u = np.full((dim,), eps)

    score_w = randn((rows, keys, dim), rng, 1.1)
    score_b = randn((rows, keys), rng, 0.25)
    coeff = randn((rows, keys), rng, 0.8)

    vertex_sum = 0.0
    wei_sum = 0.0
    rowwise_reachable = 0.0
    row_values = []
    for row in range(rows):
        score_l, score_u = affine_interval(score_w[row], score_b[row], x_l, x_u)
        row_scores = affine_values(grid, score_w[row], score_b[row])
        row_probs = softmax(row_scores, axis=1)
        row_value = np.sum(row_probs * coeff[row], axis=1)
        row_values.append(row_value)
        rowwise_reachable += float(np.min(row_value))
        vertex_sum += vertex_box_min(score_l, score_u, coeff[row])
        wei_sum += wei_lse_box_lower(score_l, score_u, coeff[row])

    joint_grid = float(np.min(np.sum(np.stack(row_values, axis=1), axis=1)))

    return DetailRow(
        diagnostic="two-row residual",
        seed=seed,
        epsilon=eps,
        softmax_gap=vertex_sum - wei_sum,
        score_box_gap=rowwise_reachable - vertex_sum,
        value_gap=None,
        rowwise_gap=joint_grid - rowwise_reachable,
        suffix_gap=None,
        final_gap_to_grid=joint_grid - vertex_sum,
        certificate_lower=vertex_sum,
        grid_reference=joint_grid,
    )


def relu_suffix_lower_bound(
    h_l: Array,
    h_u: Array,
    linear_w: Array,
    linear_b: float,
    hidden_w: Array,
    hidden_b: Array,
    hidden_out: Array,
) -> float:
    pre_l, pre_u = affine_interval(hidden_w, hidden_b, h_l, h_u)
    active = pre_l >= 0
    inactive = pre_u <= 0
    crossing = ~(active | inactive)
    positive_coeff = hidden_out >= 0

    slope = np.zeros_like(pre_l)
    intercept = np.zeros_like(pre_l)
    slope = np.where(active, 1.0, slope)

    pos_crossing = crossing & positive_coeff
    lower_slope = (pre_u > -pre_l).astype(np.float64)
    slope = np.where(pos_crossing, lower_slope, slope)

    neg_crossing = crossing & (~positive_coeff)
    denom = np.maximum(pre_u - pre_l, 1e-12)
    upper_slope = pre_u / denom
    upper_intercept = -pre_l * pre_u / denom
    slope = np.where(neg_crossing, upper_slope, slope)
    intercept = np.where(neg_crossing, upper_intercept, intercept)

    coeff_h = linear_w + np.sum(hidden_out[:, None] * slope[:, None] * hidden_w, axis=0)
    bias = linear_b + float(np.sum(hidden_out * (slope * hidden_b + intercept)))
    endpoint = np.where(coeff_h >= 0, h_l, h_u)
    return float(bias + np.dot(coeff_h, endpoint))


def relu_suffix(seed: int, eps: float, grid: Array) -> DetailRow:
    rng = np.random.default_rng(30_000 + seed)
    dim = 2
    h_dim = 4
    hidden = 6
    x_l = np.full((dim,), -eps)
    x_u = np.full((dim,), eps)

    h_w = randn((h_dim, dim), rng, 0.95)
    h_b = randn((h_dim,), rng, 0.25)
    linear_w = randn((h_dim,), rng, 0.5)
    linear_b = float(randn((), rng, 0.15))
    hidden_w = randn((hidden, h_dim), rng, 0.85)
    hidden_b = randn((hidden,), rng, 0.3)
    hidden_out = randn((hidden,), rng, 0.65)

    h_l, h_u = affine_interval(h_w, h_b, x_l, x_u)
    h_grid = affine_values(grid, h_w, h_b)
    grid_values = (
        h_grid @ linear_w
        + linear_b
        + np.maximum(h_grid @ hidden_w.T + hidden_b, 0.0) @ hidden_out
    )
    grid_min = float(np.min(grid_values))
    crown_lb = relu_suffix_lower_bound(h_l, h_u, linear_w, linear_b, hidden_w, hidden_b, hidden_out)

    return DetailRow(
        diagnostic="ReLU suffix",
        seed=seed,
        epsilon=eps,
        softmax_gap=None,
        score_box_gap=None,
        value_gap=None,
        rowwise_gap=None,
        suffix_gap=grid_min - crown_lb,
        final_gap_to_grid=grid_min - crown_lb,
        certificate_lower=crown_lb,
        grid_reference=grid_min,
    )


def _finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean_sem(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = _finite(values)
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    if len(finite) <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, math.sqrt(var / len(finite))


def summarize(rows: list[DetailRow]) -> list[SummaryRow]:
    grouped: dict[str, list[DetailRow]] = {}
    for row in rows:
        grouped.setdefault(row.diagnostic, []).append(row)

    summaries = []
    for diagnostic, group in grouped.items():
        softmax_mean, softmax_sem = mean_sem([row.softmax_gap for row in group])
        score_mean, score_sem = mean_sem([row.score_box_gap for row in group])
        value_mean, value_sem = mean_sem([row.value_gap for row in group])
        rowwise_mean, rowwise_sem = mean_sem([row.rowwise_gap for row in group])
        suffix_mean, suffix_sem = mean_sem([row.suffix_gap for row in group])
        final_mean, final_sem = mean_sem([row.final_gap_to_grid for row in group])
        if final_mean is None or final_sem is None:
            raise RuntimeError(f"missing final gap for {diagnostic}")
        summaries.append(
            SummaryRow(
                diagnostic=diagnostic,
                epsilon=group[0].epsilon,
                seeds=len(group),
                softmax_gap_mean=softmax_mean,
                softmax_gap_sem=softmax_sem,
                score_box_gap_mean=score_mean,
                score_box_gap_sem=score_sem,
                value_gap_mean=value_mean,
                value_gap_sem=value_sem,
                rowwise_gap_mean=rowwise_mean,
                rowwise_gap_sem=rowwise_sem,
                suffix_gap_mean=suffix_mean,
                suffix_gap_sem=suffix_sem,
                final_gap_to_grid_mean=final_mean,
                final_gap_to_grid_sem=final_sem,
            )
        )
    order = {"single-row attention": 0, "two-row residual": 1, "ReLU suffix": 2}
    return sorted(summaries, key=lambda row: order[row.diagnostic])


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.3f}"


def write_latex_table(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Slack-decomposition diagnostics on controlled tiny instances. Entries are mean lower-bound gaps over random seeds; larger values mean more looseness from that source. The softmax gap is Vertex-Softmax minus Wei-LSE on the same score boxes. Grid-based gaps use a dense two-dimensional input grid and are diagnostics rather than proof-producing certificates.}",
        "\\label{tab:slack_decomposition_app}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Diagnostic & $\\epsilon$ & Softmax & Score box & Value & Row-wise & Suffix & Final \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.diagnostic} & {row.epsilon:.2f} & "
            f"{fmt(row.softmax_gap_mean)} & {fmt(row.score_box_gap_mean)} & "
            f"{fmt(row.value_gap_mean)} & {fmt(row.rowwise_gap_mean)} & "
            f"{fmt(row.suffix_gap_mean)} & {fmt(row.final_gap_to_grid_mean)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--grid-points", type=int, default=181)
    parser.add_argument("--detail-out", type=Path, default=Path("results/slack_decomposition_detail.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/slack_decomposition_summary.csv"))
    parser.add_argument("--tex-out", type=Path, default=Path("paper_latex/slack_decomposition_table.tex"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = dense_grid(args.epsilon, args.grid_points)
    rows: list[DetailRow] = []
    for seed in range(args.seeds):
        rows.append(single_row_attention(seed, args.epsilon, grid))
        rows.append(two_row_residual(seed, args.epsilon, grid))
        rows.append(relu_suffix(seed, args.epsilon, grid))

    summaries = summarize(rows)
    write_csv(args.detail_out, rows)
    write_csv(args.summary_out, summaries)
    write_latex_table(args.tex_out, summaries)

    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
