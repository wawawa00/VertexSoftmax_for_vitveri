"""Run a validation grid for full-MLP Vertex-CROWN to check whether the
composed bound is competitive with direct CROWN on small ViT-block settings."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridConfig:
    name: str
    seed: int
    dim: int
    heads: int
    mlp_dim: int


QUICK_GRID = [
    GridConfig("d16_h2_mlp16_s0", seed=0, dim=16, heads=2, mlp_dim=16),
    GridConfig("d16_h2_mlp32_s0", seed=0, dim=16, heads=2, mlp_dim=32),
    GridConfig("d16_h2_mlp64_s0", seed=0, dim=16, heads=2, mlp_dim=64),
    GridConfig("d32_h4_mlp64_s0", seed=0, dim=32, heads=4, mlp_dim=64),
]

SEED_GRID = [
    GridConfig("d16_h2_mlp32_s0", seed=0, dim=16, heads=2, mlp_dim=32),
    GridConfig("d16_h2_mlp32_s1", seed=1, dim=16, heads=2, mlp_dim=32),
    GridConfig("d32_h4_mlp64_s0", seed=0, dim=32, heads=4, mlp_dim=64),
    GridConfig("d32_h4_mlp64_s1", seed=1, dim=32, heads=4, mlp_dim=64),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-prefix", type=str, default="full_mlp_validation_grid")
    parser.add_argument("--grid", choices=["quick", "seed"], default="quick")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--cert-limit", type=int, default=32)
    parser.add_argument("--cert-denominator", choices=["clean", "eval"], default="eval")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.03])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["CROWN", "objective_vertex_crown", "crown_objective_vertex_hybrid"],
    )
    parser.add_argument("--full-mlp-preactivation-bounds", choices=["hbox", "vertex", "vertex-batched"], default="hbox")
    parser.add_argument("--full-mlp-suffix-slope-mode", choices=["candidates", "optimized"], default="candidates")
    parser.add_argument("--full-mlp-suffix-opt-iters", type=int, default=20)
    parser.add_argument("--full-mlp-suffix-opt-lr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_config(args: argparse.Namespace, config: GridConfig, gpu: str) -> dict[str, object]:
    root = args.root
    results_dir = root / "results"
    logs_dir = root / "logs"
    results_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    prefix = results_dir / f"{args.out_prefix}_{config.name}"
    log_path = logs_dir / f"{args.out_prefix}_{config.name}.log"
    cmd = [
        sys.executable,
        str(root / "tiny_vit_block_benchmark.py"),
        "--out-prefix",
        str(prefix),
        "--classes",
        "0",
        "1",
        "--seed",
        str(config.seed),
        "--train-limit",
        str(args.train_limit),
        "--eval-limit",
        str(args.eval_limit),
        "--cert-limit",
        str(args.cert_limit),
        "--cert-denominator",
        args.cert_denominator,
        "--patch-size",
        "7",
        "--dim",
        str(config.dim),
        "--heads",
        str(config.heads),
        "--block-mode",
        "full_mlp",
        "--mlp-dim",
        str(config.mlp_dim),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "256",
        "--lr",
        "0.002",
        "--epsilons",
        *[str(eps) for eps in args.epsilons],
        "--methods",
        *args.methods,
        "--full-mlp-preactivation-bounds",
        args.full_mlp_preactivation_bounds,
        "--full-mlp-suffix-slope-mode",
        args.full_mlp_suffix_slope_mode,
        "--full-mlp-suffix-opt-iters",
        str(args.full_mlp_suffix_opt_iters),
        "--full-mlp-suffix-opt-lr",
        str(args.full_mlp_suffix_opt_lr),
        "--pgd-steps",
        "20",
        "--pgd-restarts",
        "1",
        "--pgd-step-size",
        "0.005",
        "--device",
        args.device,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"START config={config.name} gpu={gpu}", flush=True)
    if args.dry_run:
        print(" ".join(cmd), flush=True)
        return {"config": config.name, "gpu": gpu, "returncode": 0, "elapsed_sec": 0.0, "prefix": str(prefix)}

    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - start
    print(f"DONE config={config.name} gpu={gpu} returncode={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return {
        "config": config.name,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "prefix": str(prefix),
        "log_path": str(log_path),
    }


def read_summary(prefix: Path, config: str) -> list[dict[str, str]]:
    path = prefix.with_name(prefix.name + "_summary.csv")
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["config"] = config
        row["source_summary"] = str(path)
    return rows


def write_combined_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = ["config", "source_summary", *[key for key in rows[0].keys() if key not in {"config", "source_summary"}]]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def format_num(value: float, digits: int = 4) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def write_report(rows: list[dict[str, str]], output_path: Path, run_status: list[dict[str, object]]) -> None:
    by_pair: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        by_pair.setdefault((row["config"], row["epsilon"]), {})[row["method"]] = row

    lines = [
        "# Full-MLP Vertex-CROWN Validation Grid",
        "",
        "This is a diagnostic grid for the composed full-MLP certificate:",
        "",
        "`CROWN suffix over h` -> `linear objective in attention-residual state` -> `Vertex-Softmax through attention`.",
        "",
        "It is not intended as a final paper table.",
        "",
        "## Run Status",
        "",
        "| config | gpu | returncode | elapsed sec |",
        "|---|---:|---:|---:|",
    ]
    for status in run_status:
        lines.append(
            f"| {status['config']} | {status['gpu']} | {status['returncode']} | {float(status['elapsed_sec']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Method Comparison Against Direct CROWN",
            "",
            "| config | eps | method | CROWN lower | method lower | delta | CROWN cert | method cert | method/CROWN time | verdict |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    wins = ties = losses = missing = 0
    for (config, epsilon), methods in sorted(by_pair.items()):
        crown = methods.get("CROWN")
        if crown is None:
            missing += 1
            continue
        comparison_methods = sorted(method for method in methods if method != "CROWN")
        if not comparison_methods:
            missing += 1
            continue
        crown_lower = safe_float(crown, "mean_lower")
        crown_cert = safe_float(crown, "cert_acc")
        crown_time = safe_float(crown, "sec_per_image")
        for method in comparison_methods:
            row = methods[method]
            method_lower = safe_float(row, "mean_lower")
            delta = method_lower - crown_lower
            method_cert = safe_float(row, "cert_acc")
            method_time = safe_float(row, "sec_per_image")
            time_ratio = method_time / crown_time if crown_time > 0 else float("nan")
            if delta > 1e-4:
                verdict = "method wins"
                wins += 1
            elif delta < -1e-4:
                verdict = "crown wins"
                losses += 1
            else:
                verdict = "tie"
                ties += 1
            lines.append(
                "| "
                + " | ".join(
                    [
                        config,
                        epsilon,
                        method,
                        format_num(crown_lower),
                        format_num(method_lower),
                        format_num(delta),
                        format_num(crown_cert, 3),
                        format_num(method_cert, 3),
                        format_num(time_ratio, 3),
                        verdict,
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Method wins over CROWN: {wins}",
            f"- Ties: {ties}",
            f"- CROWN wins: {losses}",
            f"- Missing pairs: {missing}",
            "",
        ]
    )
    if losses > wins:
        lines.extend(
            [
                "The current full-MLP composition is mostly a soundness extension, not yet a stronger certificate.",
                "The likely bottleneck is the suffix interface: the MLP is relaxed over an independent box for the attention-residual state `h`, which can erase the tightness gained by exactly solving the score-box softmax subproblem.",
                "",
                "Recommended next refinement: optimize the ReLU lower-slope choices in the MLP suffix against the final Vertex-Softmax lower bound, or replace the independent `h` box with a tighter affine interface.",
            ]
        )
    elif wins > 0:
        lines.extend(
            [
                "The composed method has at least one full-MLP win region.",
                "The next step is to expand the winning configurations across seeds and certification budgets before using them in the paper.",
            ]
        )
    else:
        lines.append("The grid is inconclusive; increase cert-limit and add seeds before drawing a paper-level conclusion.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configs = QUICK_GRID if args.grid == "quick" else SEED_GRID
    run_status: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(args.gpus))) as executor:
        futures = []
        for idx, config in enumerate(configs):
            gpu = args.gpus[idx % len(args.gpus)]
            futures.append(executor.submit(run_config, args, config, gpu))
        for future in as_completed(futures):
            run_status.append(future.result())

    rows: list[dict[str, str]] = []
    for status in run_status:
        rows.extend(read_summary(Path(str(status["prefix"])), str(status["config"])))

    results_dir = args.root / "results"
    combined_path = results_dir / f"{args.out_prefix}_combined_summary.csv"
    report_path = results_dir / f"{args.out_prefix}_report.md"
    write_combined_summary(rows, combined_path)
    write_report(rows, report_path, run_status)
    print(f"wrote {combined_path}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
