#!/usr/bin/env python3
"""Paper-style scalable vertex-softmax sweep across sequence lengths and seeds."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from attention_cert_method_sweep import Result, run_setting, seed_all


@dataclass
class DetailRow:
    seq_len: int
    d_in: int
    d_head: int
    seed: int
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


@dataclass
class SummaryRow:
    seq_len: int
    d_in: int
    d_head: int
    method: str
    epsilon: float
    seeds: int
    trials_total: int
    certified_rate_mean: float
    certified_rate_std: float
    attack_robust_rate_mean: float
    mean_cert_lower: float
    mean_attack_margin: float
    mean_gap_attack_minus_cert: float
    elapsed_sec_total: float


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _methods_for_k(base_methods: list[str], seq_len: int, galileo_max_k: int) -> list[str]:
    methods = []
    for method in base_methods:
        if method == "galileo" and seq_len > galileo_max_k:
            continue
        methods.append(method)
    return methods


def _run_one(
    args: argparse.Namespace,
    device: torch.device,
    seq_len: int,
    seed: int,
) -> list[DetailRow]:
    seed_all(seed)
    methods = _methods_for_k(args.methods, seq_len, args.galileo_max_k)
    run_args = SimpleNamespace(
        seq_len=seq_len,
        d_in=args.d_in,
        d_head=args.d_head,
        trials=args.trials,
        batch=min(args.batch, args.trials),
        epsilons=args.epsilons,
        methods=methods,
        weight_scale=args.weight_scale,
        out_scale=args.out_scale,
        vertex_chunk=args.vertex_chunk,
        pgd_steps=args.pgd_steps,
        pgd_restarts=args.pgd_restarts,
        pgd_step_size=args.pgd_step_size,
    )
    setting = f"s{seq_len}_din{args.d_in}_dh{args.d_head}_ws{args.weight_scale}_seed{seed}"
    results: list[Result] = run_setting(run_args, device, setting)
    return [
        DetailRow(
            seq_len=seq_len,
            d_in=args.d_in,
            d_head=args.d_head,
            seed=seed,
            **asdict(result),
        )
        for result in results
    ]


def _summarize(rows: list[DetailRow]) -> list[SummaryRow]:
    grouped: dict[tuple[int, int, int, str, float], list[DetailRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.seq_len, row.d_in, row.d_head, row.method, row.epsilon)].append(row)

    summaries = []
    for (seq_len, d_in, d_head, method, epsilon), group in sorted(grouped.items()):
        summaries.append(
            SummaryRow(
                seq_len=seq_len,
                d_in=d_in,
                d_head=d_head,
                method=method,
                epsilon=epsilon,
                seeds=len(group),
                trials_total=sum(row.trials for row in group),
                certified_rate_mean=_mean([row.certified_rate for row in group]),
                certified_rate_std=_std([row.certified_rate for row in group]),
                attack_robust_rate_mean=_mean([row.attack_robust_rate for row in group]),
                mean_cert_lower=_mean([row.mean_cert_lower for row in group]),
                mean_attack_margin=_mean([row.mean_attack_margin for row in group]),
                mean_gap_attack_minus_cert=_mean([row.mean_gap_attack_minus_cert for row in group]),
                elapsed_sec_total=sum(row.elapsed_sec for row in group),
            )
        )
    return summaries


def _write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d-in", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--methods", nargs="+", default=["ibp", "wei_lse", "galileo", "vertex"])
    parser.add_argument("--galileo-max-k", type=int, default=16)
    parser.add_argument("--weight-scale", type=float, default=0.75)
    parser.add_argument("--out-scale", type=float, default=1.0)
    parser.add_argument("--vertex-chunk", type=int, default=4096)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-restarts", type=int, default=1)
    parser.add_argument("--pgd-step-size", type=float, default=0.003)
    parser.add_argument("--detail-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seq_lens={args.seq_lens} seeds={args.seeds}", flush=True)

    detail_rows: list[DetailRow] = []
    for seq_len in args.seq_lens:
        for seed in args.seeds:
            print(f"running seq_len={seq_len} seed={seed}", flush=True)
            detail_rows.extend(_run_one(args, device, seq_len, seed))
            _write_csv(args.detail_out, detail_rows)
            _write_csv(args.summary_out, _summarize(detail_rows))

    summaries = _summarize(detail_rows)
    _write_csv(args.detail_out, detail_rows)
    _write_csv(args.summary_out, summaries)

    print("summary", flush=True)
    for row in summaries:
        print(row, flush=True)


if __name__ == "__main__":
    main()

