# CityPulse Accelerated

**CityPulse Accelerated** helps a city civic-operations manager decide which neighborhood issues to dispatch crews to first this week. It ingests synthetic citizen complaint data, scores issues by urgency and density, and demonstrates that GPU acceleration with NVIDIA RAPIDS cuDF makes priority ranking refresh dramatically faster than plain pandas.

Built for the NVIDIA **Accelerated Data Intelligence** track (Gen AI Academy APAC, Cohort 2).

## One-line pitch

Ingest → clean → score → rank civic complaints so dispatch teams act on the highest-urgency hotspots first — with proof that cuDF cuts refresh time enough for hourly instead of daily updates.

## Who, what, bottleneck

| Question | Answer |
|----------|--------|
| **User** | City civic-ops manager / municipal helpdesk lead |
| **Decision** | Which potholes, garbage, water leaks, and streetlight issues to fix first, and where |
| **Bottleneck** | Thousands of raw records across areas and time cannot be manually cross-referenced for urgency and hotspot patterns |
| **Pipeline** | Ingest → clean → join/aggregate/score (accelerated) → rank/visualize |
| **Proof** | Side-by-side pandas vs cuDF timing at 50K, 500K, and 2M rows |

## Project structure

```
citypulse-accelerated/
├── data/generate_synthetic_data.py
├── pipeline/
│   ├── clean.py
│   ├── score.py
│   ├── benchmark.py
│   └── run_pipeline.py
├── notebooks/CityPulse_Benchmark.ipynb
├── app.py
├── requirements.txt
└── outputs/          # generated at runtime
```

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Generate synthetic data (start with 50K for the demo app)
python data/generate_synthetic_data.py --sizes 50000

# Run scoring pipeline
python -m pipeline.run_pipeline --data data/complaints_50k.csv

# Launch Streamlit app
streamlit run app.py
```

For full benchmark datasets:

```bash
python data/generate_synthetic_data.py --sizes 50000 500000 2000000
```

## GPU benchmark (Google Colab)

**Repo:** https://github.com/ayyappancsking-byte/citypulse-accelerated

**Open notebook in Colab (one click):**  
https://colab.research.google.com/github/ayyappancsking-byte/citypulse-accelerated/blob/main/notebooks/CityPulse_Benchmark.ipynb

### Steps
1. Open the Colab link above (or upload `notebooks/CityPulse_Benchmark.ipynb` manually).
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all** — the notebook clones the repo, installs RAPIDS cuDF, generates 50K/500K/2M synthetic data, and benchmarks pandas vs cuDF.
4. Download `outputs/benchmark_chart.png` for **Slide 9** of your deck.

### Outputs
| File | Use |
|------|-----|
| `outputs/benchmark_chart.png` | Slide 9 chart |
| `outputs/benchmark_results.csv` | Raw timing data |
| `outputs/benchmark_takeaway.txt` | One-line demo sentence |

## Streamlit Community Cloud deployment

1. Push this repo to a **public GitHub** repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **Main file path** to `app.py`.
4. (Optional) Add `GEMINI_API_KEY` under **Secrets** for the Ask CityPulse tab.

No credit card required — hosting runs on Streamlit Community Cloud.

## Ask CityPulse (optional Gemini integration)

The **Ask CityPulse** tab sends your question to the **Google Gemini API** (Google Cloud AI). Set your API key in a `.env` file or as an environment variable (works with both legacy `AIza…` and newer `AQ.` auth keys):

```bash
# .env (project root — never commit this file)
GEMINI_API_KEY=your_key_here
```

Or set it in the shell:

```bash
set GEMINI_API_KEY=your_key_here   # Windows
export GEMINI_API_KEY=your_key_here  # Linux/macOS
```

Never commit API keys to the repository.

## Visualization (Google Looker Studio)

CityPulse exports a dashboard-ready CSV alongside the pipeline outputs — no API keys or extra dependencies.

1. Run the pipeline: `python -m pipeline.run_pipeline --data data/complaints_50k.csv`
2. Export `outputs/looker_area_hotspots.csv` (human-readable column headers for non-technical viewers).
3. Upload the file to **Google Sheets** (File → Import → Upload).
4. Open [lookerstudio.google.com](https://lookerstudio.google.com) → **Create** → **Report**.
5. Add a data source → **Google Sheets** → select the uploaded sheet.
6. Build a **bar chart**: dimension = **Area**, metric = **Open Complaints**.
7. Share the report with **Anyone with the link** and paste the link here:

   `[Looker Studio dashboard link — add after publishing]`

The original `outputs/area_hotspots.csv` (machine-friendly column names) is unchanged for code and the Streamlit app.

## Data originality

All complaint records are **100% synthetically generated** by `data/generate_synthetic_data.py` using Faker and NumPy. Area names are generic placeholders (Ward 1, Zone North, etc.). Coordinates fall inside a fictional bounding box. Description text uses original template sentences — no real government datasets or scraped complaint text.

## Tech stack

- **Python**, **pandas**, **Streamlit**
- **NVIDIA RAPIDS cuDF** (GPU benchmark via `cudf.pandas`)
- **Google Gemini API** (optional natural-language queries)
- **Matplotlib** for charts

## License

MIT — see [LICENSE](LICENSE).
