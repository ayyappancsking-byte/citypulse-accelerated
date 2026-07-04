"""Isolated worker process for a single (engine, dataset) benchmark run.

Runs entirely inside its own Python interpreter so that enabling
`cudf.pandas.install()` for the "cudf" engine never shares process state
with a plain-pandas run. `cudf.pandas.install()` only has an effect if it
runs before pandas is imported anywhere in this process, so for the cudf
engine we do that first, before any other import.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["pandas", "cudf"], required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    result: dict = {
        "engine": args.engine,
        "seconds": None,
        "rows_loaded": None,
        "rows_cleaned": None,
        "open_ranked": None,
        "areas_scored": None,
        "error": None,
    }

    try:
        if args.engine == "cudf":
            # Must happen before the first `import pandas` in this process.
            # Import via importlib so static analyzers that don't have
            # cudf available won't flag a missing import at edit-time.
            import importlib

            cudf_pd = importlib.import_module("cudf.pandas")
            cudf_pd.install()

        import pandas as pd  # picked up fresh in THIS process only

        # Local import so pipeline modules resolve pandas/cudf.pandas
        # consistently within this same isolated process.
        from pipeline.clean import clean_complaints
        from pipeline.score import score_and_rank

        data_path = Path(args.data)

        start = time.perf_counter()
        raw = pd.read_csv(data_path)
        cleaned = clean_complaints(raw)
        ranked_open, area_summary, _ = score_and_rank(cleaned)
        elapsed = time.perf_counter() - start

        result.update(
            {
                "seconds": elapsed,
                "rows_loaded": len(raw),
                "rows_cleaned": len(cleaned),
                "open_ranked": len(ranked_open),
                "areas_scored": len(area_summary),
            }
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the parent
        result["error"] = f"{type(exc).__name__}: {exc}"

    # Single JSON line on stdout; the parent process parses only this line.
    print(json.dumps(result))


if __name__ == "__main__":
    main()