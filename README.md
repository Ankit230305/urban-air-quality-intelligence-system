🌆 Urban Air Quality Intelligence System

Predicting air-pollution hotspots and health impact assessment across cities.

This project ingests pollutant + weather data, discovers patterns, trains models, detects anomalies, estimates health risk, and serves an interactive Streamlit dashboard.

⸻

✨ What’s inside
	•	Dashboard (Streamlit)
Explore city data, run pipelines, view EDA, daily seasonality, clustering, association rules, forecasts, anomalies, health risk, and live AQI snapshots.
	•	Data Pipeline
City-wise fetching → feature engineering → pattern discovery → forecasting → anomaly detection → health scoring.
	•	Models
	•	Regression: PM2.5 prediction (RandomForest, GradientBoosting)
	•	Classification: AQI categories
	•	Time series: Bayesian forecasting (CmdStanPy) and Prophet (7-day PM2.5)
	•	Pattern mining: Daily seasonality, KMeans clustering, association rules (mlxtend)
	•	Health Risk
	•	Computes a simple index from AQI/PM2.5
	•	Categorized into Safe / Moderate / Harmful / Highly Harmful
	•	Graphs are zoomed and readable
	•	Live Data Utility
	•	Optional real-time snapshot of pollutants and weather from OpenWeatherMap
	•	“Live Now” tab shows latest AQI, pollutants, and verdict

⸻

🧭 Project Structure

urban-air-quality-intelligence-system/
├── app/
│   ├── main.py              # Streamlit dashboard
│   ├── live.py              # Live-data helpers (optional)
├── bin/
│   ├── run_city_pipeline.sh # Full pipeline runner
│   ├── fetch_live_now.py    # Quick live fetch utility
├── data/
│   ├── raw/                 # API pulls
│   ├── processed/           # *_features.csv, *_anomalies.csv, *_health.csv
│   ├── external/            # Demographics CSV (optional)
├── models/
│   ├── forecast_pm25.csv    # Forecast artifacts
│   ├── supervised_metrics_city.json
├── reports/
│   ├── seasonal_city.csv
│   ├── assoc_rules_city.csv
│   ├── patterns_city.md
├── src/
│   ├── data/                # Data loaders
│   ├── modeling/            # Training + forecasting
│   ├── utils/               # Cleaning + sanitization helpers
│   ├── visualization/       # EDA + plotting
└── README.md


⸻

🔑 Configuration

All configuration comes from .env in project root.

# Required
OPENWEATHERMAP_API_KEY=YOUR_KEY

# Optional
WAQI_TOKEN=YOUR_TOKEN
VISUALCROSSING_API_KEY=YOUR_KEY
PURPLEAIR_API_KEY=YOUR_KEY

Tests respect .env overrides for reproducibility.

⸻

🚀 Quick Start
# 1) create venv & install
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) add your API key
echo "OPENWEATHERMAP_API_KEY=YOUR_KEY" > .env

# 3) run pipeline (example: Hyderabad)
bash bin/run_city_pipeline.sh "Hyderabad" 17.3850 78.4867 2024-08-17 2024-08-24

# 4) launch dashboard
python -m streamlit run app/main.py

App runs at 👉 http://localhost:8501

⸻

📊 Dashboard Tabs
	•	Overview:
Clean pollutant values (no blanks), AQI, risk verdict, anomalies, and map.
	•	EDA:
Time-series plots, pollutant distributions, correlation heatmaps, seasonality, and geospatial map.
	•	Patterns:
Seasonal cycles and association rules between pollutants.
	•	Forecasts:
7-day PM2.5 predictions (Bayesian + Prophet).
	•	Anomalies:
Flags outliers with AQI categories. Removed irrelevant cols (temp, humidity, precip).
	•	Health:
Health risk scores (0–1), categories, and zoomed graph.
	•	Models:
Pretty tables for regression & classification scores.
	•	Live Now:
Current AQI + pollutants with verdict: Safe / Moderate / Harmful / Highly Harmful.

⸻

✅ Latest Updates
	•	Fixed empty pollutant columns in Overview tab (backfilled from processed features).
	•	Added sanitization utilities (drop_empty_columns, drop_mostly_empty_columns, coerce_none_like).
	•	Health risk tab graphs now zoomed and accurate.
	•	Removed unused anomaly cols (temp, humidity, precip, wind_speed).
	•	Added real-time AQI risk band verdict in Live Now tab.
	•	Cleaned model tables for better readability.
	•	Stable UX: safe concatenation, .env overrides respected.

⸻

🧪 Development

Run tests:

PYTHONPATH=. pytest -q
flake8 src tests


⸻

📁 Data Sources
	•	OpenWeatherMap Air Pollution + Weather
	•	(Optional) World Air Quality Index (WAQI)
	•	(Optional) Visual Crossing Weather
	•	(Optional) PurpleAir, OpenAQ, CPCB
	•	City demographics (CSV in data/external/)



source ~/venvs/uaqi/bin/activate
cd ~/Projects/urban-air-quality-intelligence-system

set -a; source .env; set +a
PYTHONPATH=. streamlit run app/main.py

git add -A && git commit -am "update" && git push