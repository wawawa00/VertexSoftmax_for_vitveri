#!/usr/bin/env python3
"""Benchmark threshold vertex-softmax against exhaustive vertex enumeration."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from attention_cert_method_sweep import (
    softmax_box_expectation_bounds,
    softmax_box_expectation_bounds_exhaustive,
    vertex_masks,
)


@dataclass
class BenchmarkRow:
    keys: int
    batch: int
    queries: int
    heads: int
    repeats: int
    threshold_sec: float
    exhaustive_sec: float | None
    max_lower_diff: float | None
    max_upper_diff: float | None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_case(args: argparse.Namespace, device: torch.device, keys: int) -> BenchmarkRow:
    torch.manual_seed(args.seed + keys)
    batch = args.batch
    queries = args.queries or keys
    heads = args.heads
    center = torch.randn(batch, queries, keys, device=device)
    radius = torch.rand(batch, queries, keys, device=device) * args.radius
    score_l = center - radius
    score_u = center + radius
    coeff_l = torch.randn(batch, keys, heads, device=device)
    coeff_u = torch.randn(batch, keys, heads, device=device)

    _sync(device)
    start = time.time()
    threshold_l = threshold_u = None
    for _ in range(args.repeats):
        threshold_l, threshold_u = softmax_box_expectation_bounds(score_l, score_u, coeff_l, coeff_u)
    _sync(device)
    threshold_sec = (time.time() - start) / args.repeats

    exhaustive_sec = None
    max_lower_diff = None
    max_upper_diff = None
    if keys <= args.max_exhaustive_keys:
        masks = vertex_masks(keys, device)
        _sync(device)
        start = time.time()
        exhaustive_l = exhaustive_u = None
        for _ in range(args.repeats):
            exhaustive_l, exhaustive_u = softmax_box_expectation_bounds_exhaustive(
                score_l,
                score_u,
                coeff_l,
                coeff_u,
                masks,
                args.vertex_chunk,
            )
        _sync(device)
        exhaustive_sec = (time.time() - start) / args.repeats
        max_lower_diff = float((threshold_l - exhaustive_l).abs().max().detach().cpu())
        max_upper_diff = float((threshold_u - exhaustive_u).abs().max().detach().cpu())

    return BenchmarkRow(
        keys=keys,
        batch=batch,
        queries=queries,
        heads=heads,
        repeats=args.repeats,
        threshold_sec=threshold_sec,
        exhaustive_sec=exhaustive_sec,
        max_lower_diff=max_lower_diff,
        max_upper_diff=max_upper_diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=int, nargs="+", default=[8, 12, 16, 32, 64, 128])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--queries", type=int, default=0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-exhaustive-keys", type=int, default=16)
    parser.add_argument("--vertex-chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for keys in args.keys:
        row = run_case(args, device, keys)
        rows.append(row)
        print(row, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


if __name__ == "__main__":
    main()

