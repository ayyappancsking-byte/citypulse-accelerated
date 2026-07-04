"""Stage B: aggregate, score urgency, and rank open complaints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Invented population counts for generic civic zones (not real census data).
AREA_POPULATION: dict[str, int] = {
    "Ward 1": 42_500,
    "Ward 2": 38_200,
    "Ward 3": 51_000,
    "Sector 4": 29_800,
    "Sector 7": 33_400,
    "Zone North": 47_600,
    "Zone South": 44_100,
    "Zone East": 36_900,
    "Zone West": 40_300,
    "District Central": 55_700,
    "District Riverside": 31_500,
    "District Hillside": 27_900,
}

CATEGORY_SEVERITY = {
    "water_leak": 1.0,
    "pothole": 0.85,
    "streetlight": 0.7,
    "garbage": 0.65,
    "other": 0.5,
}


def _population_table(pd_module) -> "pd.DataFrame":
    return pd_module.DataFrame(
        [{"area": area, "population": pop} for area, pop in AREA_POPULATION.items()]
    )


def score_and_rank(df: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Compute density signals, rolling windows, per-capita rates, and urgency scores.

    Returns:
        ranked_open: open complaints ranked by urgency_score (descending)
        area_summary: per-area hotspot metrics
        daily_density: area/category/day density table
    """
    import pandas as pd

    work = df.copy()
    work["date"] = work["timestamp"].dt.floor("D")

    daily_density = (
        work.groupby(["area", "category", "date"], as_index=False)
        .size()
        .rename(columns={"size": "daily_count"})
    )

    area_daily = (
        work.groupby(["area", "date"], as_index=False)
        .size()
        .rename(columns={"size": "area_daily_count"})
        .sort_values(["area", "date"])
    )

    area_daily["rolling_7d_count"] = (
        area_daily.groupby("area")["area_daily_count"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).sum())
    )

    pop = _population_table(pd)
    area_daily = area_daily.merge(pop, on="area", how="left")
    area_daily["population"] = area_daily["population"].fillna(area_daily["population"].median())
    area_daily["complaints_per_capita_7d"] = (
        area_daily["rolling_7d_count"] / area_daily["population"]
    ) * 10_000

    latest_day = work["date"].max()
    recency_days = (latest_day - work["date"]).dt.days.clip(lower=0)
    work["recency_factor"] = 1.0 / (1.0 + recency_days * 0.08)

    work = work.merge(
        area_daily[["area", "date", "rolling_7d_count", "complaints_per_capita_7d"]],
        on=["area", "date"],
        how="left",
    )

    work["category_severity"] = work["category"].map(CATEGORY_SEVERITY).fillna(0.5)
    work["density_signal"] = work["complaints_per_capita_7d"].fillna(0)
    work["priority_boost"] = work["reporter_priority_flag"].astype(float) * 0.25

    work["urgency_score"] = (
        0.35 * work["category_severity"]
        + 0.25 * work["recency_factor"]
        + 0.30 * work["density_signal"].clip(upper=5.0) / 5.0
        + 0.10 * work["priority_boost"]
    )

    open_mask = work["status"].isin(["open", "in_progress"])
    ranked_open = (
        work.loc[open_mask]
        .sort_values("urgency_score", ascending=False)
        .reset_index(drop=True)
    )

    area_summary = (
        work.loc[open_mask]
        .groupby("area", as_index=False)
        .agg(
            open_complaints=("complaint_id", "count"),
            avg_urgency=("urgency_score", "mean"),
            max_urgency=("urgency_score", "max"),
            complaints_per_capita_7d=("complaints_per_capita_7d", "max"),
        )
        .sort_values("avg_urgency", ascending=False)
    )

    return ranked_open, area_summary, daily_density
