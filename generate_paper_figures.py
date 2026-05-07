"""Generate paper figures and tables from CSV results."""

from __future__ import annotations

import glob
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

METHOD_LABELS = {
    "ibp": "IBP",
    "wei_lse": "Wei-LSE",
    "galileo": "GaLileo-style",
    "vertex": "Vertex",
    "crown": "CROWN",
    "alpha_crown": "alpha-CROWN",
    "score_vertex_split1": "Vertex",
    "abcrown_complete": "ABCrown-BaB, 600s",
}

METHOD_COLORS = {
    "IBP": "#8f8f8f",
    "Wei-LSE": "#d28b26",
    "GaLileo-style": "#6f63b6",
    "Vertex": "#1b8a5a",
    "CROWN": "#4f7db8",
    "alpha-CROWN": "#c44e52",
    "ABCrown-BaB, 600s": "#7a7a7a",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_runtime() -> None:
    df = pd.read_csv(RESULTS / "scalable_threshold_benchmark_gpu.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(
        df["keys"],
        df["threshold_sec"] * 1000,
        marker="o",
        linewidth=2.0,
        color=METHOD_COLORS["Vertex"],
        label="Threshold solver",
    )
    ex = df.dropna(subset=["exhaustive_sec"])
    ax.plot(
        ex["keys"],
        ex["exhaustive_sec"] * 1000,
        marker="s",
        linewidth=2.0,
        color="#9b3a3a",
        label="Exhaustive vertices",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Attention row length K")
    ax.set_ylabel("Runtime per benchmark batch (ms)")
    ax.set_title("Exact Vertex-Softmax Runtime Scaling")
    ax.set_xticks(df["keys"])
    ax.set_xticklabels([str(int(k)) for k in df["keys"]])
    ax.legend()
    save(fig, "fig_threshold_runtime")


def plot_scalable_sweep() -> None:
    df = pd.read_csv(RESULTS / "paper_scalable_combined_summary.csv")
    df = df[df["epsilon"].round(6) == 0.02].copy()
    methods = ["wei_lse", "galileo", "vertex"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    for method in methods:
        sub = df[df["method"] == method].sort_values("seq_len")
        if sub.empty:
            continue
        label = METHOD_LABELS[method]
        axes[0].plot(
            sub["seq_len"],
            sub["mean_cert_lower"],
            marker="o",
            linewidth=2.0,
            label=label,
            color=METHOD_COLORS[label],
        )
        axes[1].plot(
            sub["seq_len"],
            sub["mean_gap_attack_minus_cert"],
            marker="o",
            linewidth=2.0,
            label=label,
            color=METHOD_COLORS[label],
        )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["seq_len"].unique()))
        ax.set_xticklabels([str(int(k)) for k in sorted(df["seq_len"].unique())])
        ax.set_xlabel("Sequence length K")
    axes[0].set_ylabel("Mean certified lower bound")
    axes[0].set_title("Bound Tightness at eps=0.02")
    axes[1].set_ylabel("Attack gap (lower is better)")
    axes[1].set_title("Gap To Best Attack at eps=0.02")
    axes[1].legend(loc="best")
    save(fig, "fig_scalable_sweep_eps002")


def plot_win_region() -> None:
    df = pd.read_csv(RESULTS / "win_region_multiseed_summary.csv")
    df = df.sort_values(["seq_len", "d_in", "epsilon"]).reset_index(drop=True)
    labels = [f"s{r.seq_len},d{r.d_in},eps={r.epsilon:g}" for r in df.itertuples()]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.bar(
        x - width / 2,
        df["crown_cert_mean"],
        width,
        yerr=df["crown_cert_sem"],
        label="CROWN",
        color=METHOD_COLORS["CROWN"],
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        df["hybrid_cert_mean"],
        width,
        yerr=df["hybrid_cert_sem"],
        label="Vertex-CROWN",
        color=METHOD_COLORS["Vertex"],
        capsize=3,
    )
    ax.set_ylabel("Certified rate")
    ax.set_title("Confirmed Toy-Attention Win Regions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, max(0.8, float(df["hybrid_cert_mean"].max()) + 0.12))
    ax.legend()
    save(fig, "fig_win_region_certified_rates")


def aggregate(files: list[str]) -> pd.DataFrame:
    rows = [pd.read_csv(path) for path in files]
    df = pd.concat(rows, ignore_index=True)
    group_cols = ["seq_len", "d_in", "d_head", "epsilon", "weight_scale", "method"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            certified_rate_mean=("certified_rate", "mean"),
            certified_rate_std=("certified_rate", "std"),
            mean_lower=("mean_lower", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            mean_sec_per_trial=("mean_sec_per_trial", "mean"),
        )
        .fillna({"certified_rate_std": 0.0})
    )
    return agg


def plot_strong_baselines() -> None:
    alpha = aggregate(sorted(glob.glob(str(RESULTS / "alpha_vertex_broad_*.csv"))))
    complete = aggregate(sorted(glob.glob(str(RESULTS / "complete_vertex_broad_*_summary.csv"))))

    alpha_settings = [
        (4, 16, 0.03),
        (4, 16, 0.05),
        (4, 32, 0.03),
        (6, 16, 0.03),
    ]
    complete_settings = [
        (4, 16, 0.05),
        (4, 32, 0.03),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True)

    def draw_panel(ax, data, settings, methods, title):
        labels = [f"s{s},d{d},eps={eps:g}" for s, d, eps in settings]
        x = np.arange(len(settings))
        width = 0.8 / len(methods)
        for idx, method in enumerate(methods):
            vals = []
            errs = []
            for s, d, eps in settings:
                row = data[
                    (data["seq_len"] == s)
                    & (data["d_in"] == d)
                    & (data["epsilon"].round(6) == eps)
                    & (data["method"] == method)
                ]
                vals.append(float(row["certified_rate_mean"].iloc[0]) if not row.empty else 0.0)
                errs.append(float(row["certified_rate_std"].iloc[0]) if not row.empty else 0.0)
            label = METHOD_LABELS[method]
            offset = (idx - (len(methods) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs,
                label=label,
                color=METHOD_COLORS[label],
                capsize=2,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylim(0, 0.9)

    draw_panel(
        axes[0],
        alpha,
        alpha_settings,
        ["crown", "alpha_crown", "score_vertex_split1"],
        "alpha-CROWN comparison",
    )
    draw_panel(
        axes[1],
        complete,
        complete_settings,
        ["crown", "alpha_crown", "score_vertex_split1", "abcrown_complete"],
        "Fixed-budget ABCrown-BaB comparison",
    )
    axes[0].set_ylabel("Certified rate")
    axes[1].legend(loc="upper right")
    save(fig, "fig_strong_baseline_certified_rates")


def write_tables() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    scalable = pd.read_csv(RESULTS / "paper_scalable_combined_summary.csv")
    scalable = scalable[
        (scalable["epsilon"].round(6) == 0.02)
        & (scalable["method"].isin(["wei_lse", "galileo", "vertex"]))
    ].copy()
    scalable["method"] = scalable["method"].map(METHOD_LABELS)
    scalable = scalable[
        [
            "seq_len",
            "method",
            "certified_rate_mean",
            "mean_cert_lower",
            "mean_gap_attack_minus_cert",
            "elapsed_sec_total",
        ]
    ]

    alpha = aggregate(sorted(glob.glob(str(RESULTS / "alpha_vertex_broad_*.csv"))))
    alpha["method"] = alpha["method"].map(METHOD_LABELS)
    alpha = alpha[
        [
            "seq_len",
            "d_in",
            "epsilon",
            "method",
            "certified_rate_mean",
            "mean_lower",
            "mean_sec_per_trial",
        ]
    ]

    text = [
        "# Generated Paper Tables",
        "",
        "## Scalable Sweep at eps=0.02",
        "",
        scalable.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Broad alpha-CROWN Comparison",
        "",
        alpha.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (FIGURES / "generated_tables.md").write_text("\n".join(text))


def main() -> None:
    setup_style()
    plot_threshold_runtime()
    plot_scalable_sweep()
    plot_win_region()
    plot_strong_baselines()
    write_tables()
    print(f"Wrote figures and generated tables to {FIGURES}")


if __name__ == "__main__":
    main()
