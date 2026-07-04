"""Benchmark pandas vs GPU-accelerated cuDF pipeline across dataset sizes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

Engine = Literal["pandas", "cudf"]


def run_timed_pipeline(data_path: Path, engine: Engine = "pandas") -> dict:
    """Run one (engine, dataset) benchmark in its own isolated subprocess.

    Isolation matters because `cudf.pandas.install()` only takes effect if it
    runs before pandas is imported anywhere in that process. Running each
    engine in a fresh interpreter guarantees a clean import order for cudf
    and leaves plain pandas runs completely unaffected.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline._benchmark_worker",
            "--engine",
            engine,
            "--data",
            str(data_path),
        ],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "engine": engine,
            "seconds": None,
            "error": proc.stderr.strip()[-500:] or f"worker exited with code {proc.returncode}",
        }

    # The worker prints exactly one JSON line; take the last non-empty line
    # in case any library prints extra output before it.
    last_line = [line for line in proc.stdout.splitlines() if line.strip()][-1]
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {
            "engine": engine,
            "seconds": None,
            "error": f"could not parse worker output: {last_line[:300]}",
        }


def benchmark_all_sizes(
    data_dir: Path,
    sizes: list[str] | None = None,
    engines: list[Engine] | None = None,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    sizes = sizes or ["50k", "500k", "2m"]
    engines = engines or ["pandas", "cudf"]

    records: list[dict] = []
    for size_label in sizes:
        csv_path = data_dir / f"complaints_{size_label}.csv"
        if not csv_path.exists():
            continue
        for engine in engines:
            result = run_timed_pipeline(csv_path, engine=engine)
            result["engine"] = engine
            result["dataset"] = size_label
            records.append(result)

    return pd.DataFrame(records)


def write_takeaway(results: pd.DataFrame, output_path: Path) -> str:
    """Generate a one-line decision-relevant benchmark takeaway."""
    usable = results.dropna(subset=["seconds"])
    if usable.empty:
        takeaway = (
            "Run the Colab benchmark on a T4 GPU to compare pandas and cuDF timings."
        )
        output_path.write_text(takeaway, encoding="utf-8")
        return takeaway

    pivot = usable.pivot_table(index="dataset", columns="engine", values="seconds", aggfunc="mean")
    if "pandas" not in pivot.columns or "cudf" not in pivot.columns:
        takeaway = "Partial benchmark results saved; run both engines on all dataset sizes."
        output_path.write_text(takeaway, encoding="utf-8")
        return takeaway

    speedups = (pivot["pandas"] / pivot["cudf"]).dropna()
    avg_speedup = speedups.mean()
    largest = pivot.index[-1]
    largest_speedup = speedups.iloc[-1] if len(speedups) else avg_speedup

    takeaway = (
        f"The GPU pipeline is {avg_speedup:.1f}x faster on average "
        f"({largest_speedup:.1f}x at {largest} scale), letting the priority list "
        "refresh hourly instead of daily."
    )
    output_path.write_text(takeaway, encoding="utf-8")
    return takeaway


def save_benchmark_chart(results: pd.DataFrame, chart_path: Path) -> None:
    import matplotlib.pyplot as plt

    usable = results.dropna(subset=["seconds"]).copy()
    if usable.empty:
        return

    order = ["50k", "500k", "2m"]
    usable["dataset"] = pd.Categorical(usable["dataset"], categories=order, ordered=True)
    usable = usable.sort_values("dataset")

    fig, ax = plt.subplots(figsize=(8, 5))
    for engine, group in usable.groupby("engine"):
        ax.plot(group["dataset"].astype(str), group["seconds"], marker="o", label=engine)

    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Seconds")
    ax.set_title("CityPulse Pipeline: pandas vs cuDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def run_benchmark(
    data_dir: Path | str = "data",
    output_dir: Path | str = "outputs",
    engines: list[Engine] | None = None,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = benchmark_all_sizes(data_dir, engines=engines)
    results.to_csv(output_dir / "benchmark_results.csv", index=False)
    takeaway = write_takeaway(results, output_dir / "benchmark_takeaway.txt")
    save_benchmark_chart(results, output_dir / "benchmark_chart.png")
    print(takeaway)
    return results


if __name__ == "__main__":
    run_benchmark()