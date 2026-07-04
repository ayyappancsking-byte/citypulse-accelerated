"""
Generate fully synthetic civic complaint records for CityPulse Accelerated.

All data is programmatically invented — no real addresses, ward names, or complaint text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Plausible bounding box for a fictional metro region (not tied to any real city).
LAT_MIN, LAT_MAX = 12.80, 13.20
LON_MIN, LON_MAX = 77.40, 77.90

AREAS = [
    "Ward 1",
    "Ward 2",
    "Ward 3",
    "Sector 4",
    "Sector 7",
    "Zone North",
    "Zone South",
    "Zone East",
    "Zone West",
    "District Central",
    "District Riverside",
    "District Hillside",
]

CATEGORIES = ["pothole", "garbage", "water_leak", "streetlight", "other"]

STATUSES = ["open", "in_progress", "resolved", "closed"]

DESCRIPTION_TEMPLATES = {
    "pothole": [
        "Pothole reported on {street}, causing traffic slowdown.",
        "Large pothole near {street} intersection needs repair.",
        "Repeated pothole damage on {street} after recent rains.",
    ],
    "garbage": [
        "Uncollected garbage pile accumulating near {street}.",
        "Overflowing bins reported on {street} for three days.",
        "Illegal dumping sighted behind {street} market area.",
    ],
    "water_leak": [
        "Water leak observed on {street} sidewalk.",
        "Burst pipe suspected under {street} causing pooling.",
        "Low pressure and leak reported along {street}.",
    ],
    "streetlight": [
        "Streetlight out on {street} creating safety concern.",
        "Flickering streetlight near {street} crosswalk.",
        "Multiple dark spots reported along {street}.",
    ],
    "other": [
        "General maintenance issue reported on {street}.",
        "Noise and obstruction complaint filed for {street}.",
        "Miscellaneous civic concern logged for {street}.",
    ],
}


def _random_coordinates(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    lats = rng.uniform(LAT_MIN, LAT_MAX, size=n)
    lons = rng.uniform(LON_MIN, LON_MAX, size=n)
    return lats, lons


def _build_descriptions(
    rng: np.random.Generator, categories: np.ndarray, fake: Faker
) -> list[str]:
    descriptions: list[str] = []
    for category in categories:
        templates = DESCRIPTION_TEMPLATES[category]
        template = templates[rng.integers(0, len(templates))]
        street = fake.street_name()
        descriptions.append(template.format(street=street))
    return descriptions


def generate_complaints(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    complaint_ids = [f"CMP-{i:08d}" for i in range(1, n_rows + 1)]

    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-06-30")
    seconds_span = int((end - start).total_seconds())
    random_seconds = rng.integers(0, seconds_span, size=n_rows)
    timestamps = start + pd.to_timedelta(random_seconds, unit="s")

    area_choices = rng.choice(AREAS, size=n_rows)
    category_choices = rng.choice(
        CATEGORIES,
        size=n_rows,
        p=[0.28, 0.24, 0.18, 0.18, 0.12],
    )

    status_choices = rng.choice(
        STATUSES,
        size=n_rows,
        p=[0.45, 0.15, 0.25, 0.15],
    )

    priority_flags = rng.random(n_rows) < 0.12

    lats, lons = _random_coordinates(rng, n_rows)

    # Inject a small fraction of missing coordinates for cleaning tests.
    missing_mask = rng.random(n_rows) < 0.008
    lats = lats.astype(float)
    lons = lons.astype(float)
    lats[missing_mask] = np.nan
    lons[missing_mask] = np.nan

    descriptions = _build_descriptions(rng, category_choices, fake)

    return pd.DataFrame(
        {
            "complaint_id": complaint_ids,
            "timestamp": timestamps,
            "area": area_choices,
            "category": category_choices,
            "description_text": descriptions,
            "latitude": lats,
            "longitude": lons,
            "reporter_priority_flag": priority_flags,
            "status": status_choices,
        }
    )


def save_dataset(df: pd.DataFrame, output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"complaints_{label}.csv"
    parquet_path = output_dir / f"complaints_{label}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df):,} rows -> {csv_path.name}, {parquet_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic civic complaint data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated CSV/Parquet files.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50_000, 500_000, 2_000_000],
        help="Row counts to generate.",
    )
    args = parser.parse_args()

    for size in args.sizes:
        label = f"{size // 1000}k" if size < 1_000_000 else f"{size // 1_000_000}m"
        df = generate_complaints(size, seed=1000 + size)
        save_dataset(df, args.output_dir, label)


if __name__ == "__main__":
    main()
