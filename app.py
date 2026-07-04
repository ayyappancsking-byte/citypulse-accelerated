"""
CityPulse Accelerated — Streamlit dashboard for civic issue prioritization.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline.clean import clean_complaints
from pipeline.score import score_and_rank

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def reload_env() -> None:
    """Reload .env so GEMINI_API_KEY / GEMINI_MODEL edits apply without full restart."""
    load_dotenv(ROOT / ".env", override=True)


reload_env()


def ensure_demo_dataset() -> None:
    """Create the 50k synthetic demo file if missing (e.g. on Streamlit Cloud)."""
    demo_path = DATA_DIR / "complaints_50k.csv"
    if demo_path.exists():
        return
    from data.generate_synthetic_data import generate_complaints, save_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_dataset(generate_complaints(50_000, seed=50_000), DATA_DIR, "50k")

DATASET_OPTIONS = {
    "50k (demo)": DATA_DIR / "complaints_50k.csv",
    "500k": DATA_DIR / "complaints_500k.csv",
    "2M": DATA_DIR / "complaints_2m.csv",
}


@st.cache_data(show_spinner="Loading and scoring complaints...")
def load_ranked_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(csv_path)
    cleaned = clean_complaints(raw)
    ranked_open, area_summary, daily_density = score_and_rank(cleaned)
    return ranked_open, area_summary, daily_density


def render_hotspot_chart(area_summary: pd.DataFrame) -> None:
    top_areas = area_summary.head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_areas["area"], top_areas["open_complaints"], color="#2563eb")
    ax.set_xlabel("Open complaints")
    ax.set_title("Top hotspot areas (open issues)")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def get_gemini_api_key() -> str | None:
    reload_env()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def get_gemini_model() -> str:
    reload_env()
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _create_gemini_client(api_key: str):
    """Create a Gemini client using google-genai (supports AIza and AQ. auth keys)."""
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "The google-genai package is required for Ask CityPulse. "
            "Run: pip install google-genai"
        ) from exc

    return genai.Client(api_key=api_key)


def ask_gemini(question: str) -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        return (
            "Set GEMINI_API_KEY in your environment or a `.env` file in the project root "
            "to enable natural-language queries. The rest of CityPulse works without it."
        )

    try:
        client = _create_gemini_client(api_key)
    except ImportError as exc:
        return str(exc)

    model = get_gemini_model()

    try:
        response = client.models.generate_content(model=model, contents=question.strip())
    except Exception as exc:  # noqa: BLE001 - surface API errors in the UI
        return f"Gemini request failed: {exc}"

    text = getattr(response, "text", None)
    if not text:
        return "Gemini returned an empty response. Try again or check your API key permissions."
    return text.strip()


def render_dashboard() -> None:
    st.subheader("Weekly dispatch priorities")
    st.caption(
        "Ranked open issues by urgency — category severity, recency, density, and reporter priority."
    )

    available = {label: path for label, path in DATASET_OPTIONS.items() if path.exists()}
    if not available:
        st.warning(
            "No dataset found. Run `python data/generate_synthetic_data.py --sizes 50000` "
            "from the project root, then refresh."
        )
        return

    choice = st.selectbox("Dataset", list(available.keys()))
    top_n = st.slider("Top N actions to show", 5, 50, 20)

    ranked_open, area_summary, _ = load_ranked_data(str(available[choice]))

    col1, col2, col3 = st.columns(3)
    col1.metric("Open / in-progress", f"{len(ranked_open):,}")
    col2.metric("Areas tracked", f"{area_summary['area'].nunique():,}")
    col3.metric(
        "Highest urgency",
        f"{ranked_open.iloc[0]['urgency_score']:.2f}" if len(ranked_open) else "—",
    )

    display_cols = [
        "complaint_id",
        "area",
        "category",
        "timestamp",
        "urgency_score",
        "reporter_priority_flag",
        "description_text",
    ]
    st.dataframe(ranked_open[display_cols].head(top_n), use_container_width=True, hide_index=True)

    st.markdown("#### Hotspot areas")
    render_hotspot_chart(area_summary)


def render_ask_citypulse() -> None:
    st.subheader("Ask CityPulse")

    question = st.text_area(
        "Your question",
        placeholder="Which water leaks in Zone North should we dispatch first?",
        height=80,
    )
    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Consulting CityPulse..."):
                answer = ask_gemini(question.strip())
            st.markdown(answer)


def render_acceleration_proof() -> None:
    st.subheader("Acceleration proof")
    st.caption("pandas vs NVIDIA RAPIDS cuDF — run the Colab notebook on a T4 GPU for full results.")

    results_path = OUTPUT_DIR / "benchmark_results.csv"
    takeaway_path = OUTPUT_DIR / "benchmark_takeaway.txt"
    chart_path = OUTPUT_DIR / "benchmark_chart.png"

    if takeaway_path.exists():
        st.success(takeaway_path.read_text(encoding="utf-8"))
    else:
        st.info(
            "Benchmark outputs not found yet. Open `notebooks/CityPulse_Benchmark.ipynb` "
            "in Google Colab (Runtime → Change runtime type → T4 GPU), then run all cells."
        )

    if results_path.exists():
        results = pd.read_csv(results_path)
        st.dataframe(results, use_container_width=True, hide_index=True)

        usable = results.dropna(subset=["seconds"])
        if not usable.empty:
            order = ["50k", "500k", "2m"]
            usable["dataset"] = pd.Categorical(usable["dataset"], categories=order, ordered=True)
            usable = usable.sort_values("dataset")
            fig, ax = plt.subplots(figsize=(8, 4))
            for engine, group in usable.groupby("engine"):
                ax.plot(
                    group["dataset"].astype(str),
                    group["seconds"],
                    marker="o",
                    label=engine,
                )
            ax.set_xlabel("Dataset size")
            ax.set_ylabel("Seconds")
            ax.set_title("Pipeline runtime: pandas vs cuDF")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    elif chart_path.exists():
        st.image(str(chart_path), caption="Benchmark chart from Colab run")

    st.markdown(
        """
**How to reproduce on Colab (free T4 GPU):**
1. Push this repo to GitHub and open the notebook in Colab.
2. Install RAPIDS cuDF for your CUDA version.
3. Run `pipeline/benchmark.py` for all three dataset sizes.
4. Download `outputs/benchmark_chart.png` for slide 9 of your deck.
        """
    )


def main() -> None:
    ensure_demo_dataset()
    st.set_page_config(
        page_title="CityPulse Accelerated",
        page_icon="🏙️",
        layout="wide",
    )
    st.title("CityPulse Accelerated")
    st.markdown(
        "Helps a **city civic-operations manager** decide **which neighborhood issues "
        "to dispatch crews to first this week** — with GPU-accelerated ranking via NVIDIA RAPIDS cuDF."
    )

    tab_dashboard, tab_ask, tab_proof = st.tabs(
        ["Dashboard", "Ask CityPulse", "Acceleration Proof"]
    )

    with tab_dashboard:
        render_dashboard()

    with tab_ask:
        render_ask_citypulse()

    with tab_proof:
        render_acceleration_proof()


if __name__ == "__main__":
    main()
