# Urban Air Quality Intelligence System

Predicting air-pollution hotspots and health impact assessment across cities.  
This project ingests pollutant + weather data, discovers patterns, trains models, detects anomalies,
estimates health risk, and serves an interactive Streamlit dashboard.

---

## ✨ What’s inside

- **Dashboard (Streamlit)**: Explore city data, run pipelines, view daily seasonality, clustering,
  association rules, forecasts, anomalies, and health risk.
- **Data pipeline**: City-wise fetching → feature engineering → pattern discovery → forecasts →
  anomalies → health scoring.
- **Models**
  - Regression (PM2.5)
  - Classification (AQI categories)
  - Time series (7-day PM2.5 via Prophet)
- **Pattern mining**: Daily seasonality, KMeans clustering, association rules (mlxtend).
- **Health risk**: Simple risk index from AQI/PM2.5 with category labels.
- **Live data utility (optional)**: Fetch current OpenWeatherMap air-pollution & weather snapshots.

---

## 🧭 Project structure

app/
main.py            # Streamlit dashboard
live.py            # Live-data helpers (optional)
bin/
run_city_pipeline.sh   # Full pipeline per city/date range
fetch_live_now.py      # Quick live fetch utility (optional)
data/
external/               # e.g., demographics CSV
processed/              # features_plus_demo.csv, *anomalies.csv, *health.csv
raw/                    # API pulls
models/
.joblib                # trained models
forecast_pm25.csv      # Prophet forecasts
reports/
seasonal.csv
assoc_rules.csv
patterns_.md
src/
data/
features/
modeling/
utils/
tests/

---

## 🔑 Configuration

We load configuration from a **`.env` in the current working directory (CWD)** if present, which **overrides**
environment variables (intentional for tests and local dev).

Create `.env` (or export these in your shell):

required for historical + live pulls

OPENWEATHERMAP_API_KEY=YOUR_KEY

optional, if you also wire WAQI

WAQI_TOKEN=YOUR_TOKEN

optional, if you add more sources later

VISUALCROSSING_API_KEY=YOUR_KEY

> Tests rely on the CWD `.env` override behavior.

---

## 🚀 Quick start

```bash
# 1) create venv & install
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) set your API key
export OPENWEATHERMAP_API_KEY=YOUR_KEY

# 3) run the dashboard
python -m streamlit run app/main.py

Open the printed Local URL (usually http://localhost:8501).

### Runbook (macOS + VS Code)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Put your key in .env or export before running
# echo "OPENWEATHERMAP_API_KEY=YOUR_KEY" > .env
python -m streamlit run app/main.py
```

### Live Data tips
- Live Now tab and `bin/fetch_live_now.py` require `OPENWEATHERMAP_API_KEY`.
- On errors, the app shows a friendly banner (no crashes) and how to fix.


⸻

🛠️ Pipelines & outputs

Run the full pipeline for a city

bash bin/run_city_pipeline.sh "Kolkata" 22.5726 88.3639 2024-08-17 2024-08-24

Outputs include:
	•	data/processed/<city>__features_plus_demo.csv
	•	data/processed/<city>__anomalies.csv
	•	data/processed/<city>__health.csv
	•	models/forecast_pm25.csv (and/or models/forecast_pm25_<city>.csv)
	•	reports/seasonal_<city>.csv
	•	reports/assoc_rules_<city>.csv
	•	reports/patterns_<city>.md

⸻

📊 Dashboard features
	•	City selector + date inputs
	•	Processed data preview (features + demographics)
	•	Daily seasonality CSV view (per city)
	•	Clustering summary (KMeans on pollutants + weather)
	•	Association rules (mlxtend; discretized into terciles)
	•	Forecasts (Prophet 7-day PM2.5)
	•	Anomaly detection view (*_anomalies.csv)
	•	Health risk view (*_health.csv)
	•	Run pipeline button (no st.experimental_rerun; clean UX)

Stability/UX improvements
	•	Resampling switched to lowercase 'h' (pandas deprecation safe).
	•	Added safe_concat() to silence future pandas concat dtype warnings.
	•	.env in CWD overrides environment variables (override=True) to keep tests/dev deterministic.

⸻

🤖 Modeling

src/modeling/train_supervised.py, src/modeling/train_time_series.py:
	•	Regression: RandomForest, GradientBoosting (PM2.5)
	•	Classification: AQI category
	•	Time series: Prophet 7-day PM2.5 (models/forecast_pm25*.csv)
	•	Metrics saved to models/supervised_metrics_<city>.json.

⸻

🚨 Anomalies & 🫁 Health risk
	•	src/modeling/detect_anomalies.py: flags outliers from the processed time series
	•	src/modeling/health_risk.py: computes a simple risk index & category
	•	Both write per-city CSVs in data/processed/.

⸻

🌐 Optional: Live data
	•	CLI: bin/fetch_live_now.py pulls current OpenWeatherMap air-pollution & weather.
	•	App: app/live.py provides small helpers for the dashboard.

Requires OPENWEATHERMAP_API_KEY. New keys may need time to activate.

⸻

🧪 Tests & Linting

# run unit tests (same as CI)
PYTHONPATH=. pytest -q

# lint tracked source & tests
flake8 src tests

CI pipeline
	•	PYTHONPATH=. ensures src imports work
	•	flake8 src tests only (not build/dist)
	•	Uses .env override logic for config tests

⸻

📁 Data sources (used / ready to plug in)
	•	OpenWeatherMap Air Pollution + Weather (free tier)
	•	(Optional) World Air Quality Index (WAQI)
	•	Visual Crossing Weather (optional historical)
	•	City demographics (CSV in data/external/)
	•	Extensible to PurpleAir, OpenAQ, CPCB, etc.

⸻

🔍 Troubleshooting
	•	ModuleNotFoundError: No module named 'src' → run tests with PYTHONPATH=.
	•	.env not respected → ensure .env is in the current directory you run from
	•	OpenWeatherMap 401 / empty live data → verify OPENWEATHERMAP_API_KEY is active
	•	Pandas FutureWarnings → already handled via 'h' resampling + safe_concat()



Thanks to OpenWeatherMap, WAQI, Visual Crossing, and the open data community.

Need badges, screenshots, or a quick demo GIF section added? Tell me the image URLs/titles and I’ll give you a drop-in snippet.
