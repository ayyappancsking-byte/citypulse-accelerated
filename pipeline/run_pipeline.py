"""Run the full CityPulse ingest → clean → score pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.clean import clean_complaints
from pipeline.score import score_and_rank

LOOKER_AREA_COLUMNS = {
    "area": "Area",
    "open_complaints": "Open Complaints",
    "avg_urgency": "Avg Urgency Score",
    "max_urgency": "Max Urgency Score",
    "complaints_per_capita_7d": "Complaints Per Capita",
}


def save_looker_area_hotspots(area_summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a human-readable CSV copy for manual Looker Studio import."""
    looker_df = area_summary.rename(columns=LOOKER_AREA_COLUMNS)
    looker_df.to_csv(output_dir / "looker_area_hotspots.csv", index=False)


def run(data_path: Path, output_dir: Path, top_n: int = 25) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    cleaned = clean_complaints(raw)
    ranked_open, area_summary, daily_density = score_and_rank(cleaned)

    cleaned.to_parquet(output_dir / "cleaned_complaints.parquet", index=False)
    ranked_open.head(top_n).to_csv(output_dir / "top_ranked_actions.csv", index=False)
    area_summary.to_csv(output_dir / "area_hotspots.csv", index=False)
    save_looker_area_hotspots(area_summary, output_dir)
    daily_density.to_csv(output_dir / "daily_density.csv", index=False)

    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Open/in-progress ranked: {len(ranked_open):,}")
    print(f"Top area hotspot: {area_summary.iloc[0]['area'] if len(area_summary) else 'n/a'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CityPulse scoring pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/complaints_50k.csv"),
        help="Path to complaint CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for pipeline outputs.",
    )
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()
    run(args.data, args.output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
