import csv
import glob
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent


def read_method_times() -> dict[str, float]:
    files = sorted(ROOT.glob("results/full_mlp_reinf_main_d32h4m64_eps002_s*_c200_summary.csv"))
    if not files:
        raise FileNotFoundError("missing full_mlp_reinf_main summary files")
    times: dict[str, list[float]] = {}
    for path in files:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                times.setdefault(row["method"], []).append(float(row["sec_per_image"]))
    return {method: mean(values) for method, values in times.items()}


def scaled_threshold_time() -> float:
    path = ROOT / "results/scalable_threshold_benchmark_gpu.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if int(row["keys"]) == 16)
    threshold_sec = float(row["threshold_sec"])
    benchmark_objectives = int(row["batch"]) * int(row["queries"]) * int(row["heads"])
    full_block_objectives = 9 * 4 * 16  # targets x heads x query rows for the d32/h4/m64 10-class block.
    return threshold_sec * full_block_objectives / benchmark_objectives


def write_table(path: Path) -> None:
    times = read_method_times()
    crown = times["CROWN"]
    vertex = times["objective_vertex_crown"]
    hybrid = times["crown_objective_vertex_hybrid"]
    threshold = scaled_threshold_time()
    dispatch = max(hybrid - vertex, 0.0)
    bound_construction = max(hybrid - crown - threshold - dispatch, 0.0)

    rows = [
        ("Direct CROWN", crown),
        ("Score/value/suffix bounds", bound_construction),
        ("Vertex sort/sweep", threshold),
        ("Hybrid overhead", dispatch),
        ("Total hybrid", hybrid),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Coarse runtime accounting for the RTX 4090 full-block MNIST setting $d=32,h=4,m=64,\\epsilon=0.02$. Direct CROWN, Vertex path, and Hybrid totals are averaged over three seed runs with 200 certified images per seed. The Vertex-Softmax sort/sweep entry is estimated by scaling the measured $K=16$ threshold microbenchmark to $9$ targets, $4$ heads, and $16$ query rows; the remaining Vertex-path time is therefore attributed to score, value, and suffix-bound construction.}",
        "\\label{tab:runtime_breakdown_app}",
        "\\begin{tabular}{p{0.48\\columnwidth}rr}",
        "\\toprule",
        "Component & Sec./img & Fraction \\\\",
        "\\midrule",
    ]
    for name, seconds in rows:
        fraction = seconds / hybrid if hybrid > 0 else 0.0
        lines.append(f"{name} & {seconds:.4f} & {fraction:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    out = ROOT / "paper_latex/runtime_breakdown_table.tex"
    write_table(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
