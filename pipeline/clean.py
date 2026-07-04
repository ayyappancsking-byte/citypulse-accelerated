"""Stage A: ingest and clean civic complaint records."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

VALID_CATEGORIES = {"pothole", "garbage", "water_leak", "streetlight", "other"}


def clean_complaints(df: "pd.DataFrame") -> "pd.DataFrame":
    """Parse timestamps, deduplicate, fix coordinates, and normalize categories."""
    import pandas as pd

    cleaned = df.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp", "complaint_id"])

    cleaned = cleaned.drop_duplicates(subset=["complaint_id"], keep="first")

    cleaned["category"] = (
        cleaned["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    cleaned.loc[~cleaned["category"].isin(VALID_CATEGORIES), "category"] = "other"

    cleaned["area"] = cleaned["area"].astype(str).str.strip()
    cleaned["status"] = cleaned["status"].astype(str).str.strip().str.lower()
    cleaned["reporter_priority_flag"] = cleaned["reporter_priority_flag"].fillna(False).astype(bool)

    cleaned = cleaned.dropna(subset=["latitude", "longitude"])
    cleaned = cleaned[
        cleaned["latitude"].between(-90, 90) & cleaned["longitude"].between(-180, 180)
    ]

    return cleaned.reset_index(drop=True)
