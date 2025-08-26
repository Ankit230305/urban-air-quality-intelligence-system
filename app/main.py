#!/usr/bin/env python3
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Urban Air Quality Intelligence", layout="wide")

# -----------------------
# Helpers
# -----------------------
CITIES = {
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Vizag": (17.6868, 83.2185),
    "Vellore": (12.9165, 79.1325),
}


def slug_of(city: str) -> str:
    return city.lower().replace(" ", "_")


def find_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def find_processed_for_city(slug: str):
    return find_first_existing([
        f"data/processed/{slug}__features_plus_demo.csv",
        f"data/processed/{slug}_features_plus_demo.csv",
    ])


def find_patterns_for_city(slug: str):
    daily = find_first_existing([f"reports/seasonal_{slug}.csv"])
    rules = find_first_existing([f"reports/assoc_rules_{slug}.csv"])
    md = find_first_existing([f"reports/patterns_{slug}.md"])
    return daily, rules, md


def find_metrics_for_city(slug: str):
    return find_first_existing([f"models/supervised_metrics_{slug}.json"])


def find_forecast_for_city(slug: str):
    return find_first_existing([
        f"models/forecast_pm25_{slug}.csv",
        "models/forecast_pm25.csv",
    ])


def find_anomalies_for_city(slug: str):
    return find_first_existing([
        f"data/processed/{slug}__anomalies.csv",
        f"data/processed/{slug}_anomalies.csv",
    ])


def find_health_for_city(slug: str):
    return find_first_existing([
        f"data/processed/{slug}__health.csv",
        f"data/processed/{slug}_health.csv",
    ])


def load_csv_safe(path, parse_dates=None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return pd.DataFrame()


def as_numeric(df: pd.DataFrame, cols):
    if df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def kpi_card(col, label, value, suffix=""):
    with col:
        st.metric(label, f"{value}{suffix}")


# -----------------------
# Sidebar Controls
# -----------------------
with st.sidebar:
    st.header("Controls")

    city = st.selectbox("City", list(CITIES.keys()), index=list(CITIES.keys()).index("Kolkata") if "Kolkata" in CITIES else 0)
    lat, lon = CITIES[city]
    ll = st.text_input("Lat, Lon", value=f"{lat}, {lon}")

    d_start, d_end = st.date_input(
        "Date range (UTC)",
        (date(2024, 8, 17), date(2024, 8, 24))
    )

    run_btn = st.button("Build / Refresh city data", type="primary", use_container_width=True)

if run_btn:
    try:
        lat_str, lon_str = [x.strip() for x in ll.split(",")]
        cmd = [
            "bash", "bin/run_city_pipeline.sh",
            city, lat_str, lon_str,
            str(d_start), str(d_end)
        ]
        with st.spinner(f"Running pipeline for {city}..."):
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        st.success("Pipeline finished")
        st.expander("Build logs", expanded=False).code(proc.stdout + "\n" + proc.stderr)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")

slug = slug_of(city)

files = {
    "processed": find_processed_for_city(slug),
    "patterns_daily": find_patterns_for_city(slug)[0],
    "rules": find_patterns_for_city(slug)[1],
    "patterns_md": find_patterns_for_city(slug)[2],
    "metrics": find_metrics_for_city(slug),
    "forecast": find_forecast_for_city(slug),
    "anomalies": find_anomalies_for_city(slug),
    "health": find_health_for_city(slug),
}

st.title("🌆 Urban Air Quality Intelligence System")
st.caption(f"City: **{city}**  ·  Data file: **{files['processed'] if files['processed'] else 'not found'}**")

tabs = st.tabs(["Overview", "EDA", "Pattern Discovery", "Models", "Forecast", "Anomalies", "Health"])

# -----------------------
# Overview Tab
# -----------------------
with tabs[0]:
    df = load_csv_safe(files["processed"], parse_dates=["datetime"])
    df = as_numeric(df, ["pm2_5", "pm10", "no2", "o3", "so2", "co", "aqi", "temp", "humidity", "wind_speed", "precip"])

    if df.empty:
        st.info("No processed dataset found for this city yet. Click **Build / Refresh city data**.")
    else:
        latest = df.dropna(subset=["pm2_5", "pm10", "aqi"], how="all").tail(1)
        c1, c2, c3, c4 = st.columns(4)
        if not latest.empty:
            kpi_card(c1, "PM2.5 (latest)", round(float(latest["pm2_5"].iloc[0]), 2) if pd.notna(latest["pm2_5"].iloc[0]) else "NA", " µg/m³")
            kpi_card(c2, "PM10 (latest)", round(float(latest["pm10"].iloc[0]), 2) if pd.notna(latest["pm10"].iloc[0]) else "NA", " µg/m³")
            kpi_card(c3, "AQI (latest)", round(float(latest["aqi"].iloc[0]), 2) if pd.notna(latest["aqi"].iloc[0]) else "NA")
            kpi_card(c4, "Rows", f"{len(df):,}")

        st.markdown("#### Hourly Pollution Trend")
        ycols = [c for c in ["pm2_5", "pm10", "no2", "o3", "so2", "co"] if c in df.columns]
        if ycols:
            fig = px.line(df, x="datetime", y=ycols)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pollutant columns present to plot.")

# -----------------------
# EDA Tab
# -----------------------
with tabs[1]:
    st.markdown("### Exploratory Data Analysis")
    df = load_csv_safe(files["processed"], parse_dates=["datetime"])
    df = as_numeric(df, ["pm2_5", "pm10", "no2", "o3", "so2", "co", "aqi", "temp", "humidity", "wind_speed", "precip"])

    if df.empty:
        st.info("No processed data to analyze.")
    else:
        st.markdown("#### Summary (last 50 rows)")
        st.dataframe(df.tail(50), use_container_width=True)

        st.markdown("#### Correlation Heatmap")
        eda_cols = [c for c in ["pm2_5", "pm10", "no2", "o3", "so2", "co", "temp", "humidity", "wind_speed", "precip", "aqi"] if c in df.columns]
        if len(eda_cols) >= 2:
            corr = df[eda_cols].corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

        st.markdown("#### AQI Category Distribution")
        if "aqi_category" in df.columns:
            st.plotly_chart(px.histogram(df, x="aqi_category"), use_container_width=True)
        else:
            st.info("No 'aqi_category' column available.")

# -----------------------
# Pattern Discovery Tab
# -----------------------
with tabs[2]:
    st.markdown("### Pattern Discovery")
    daily = load_csv_safe(files["patterns_daily"], parse_dates=["datetime"])
    rules = load_csv_safe(files["rules"])

    if daily.empty:
        st.info("No seasonal/daily file found. (Run the pipeline first.)")
    else:
        st.markdown("#### Daily Means (tail)")
        st.dataframe(daily.tail(30), use_container_width=True)

        if "cluster" in daily.columns and daily["cluster"].notna().any():
            st.markdown("#### Cluster sizes")
            st.plotly_chart(px.histogram(daily.dropna(subset=["cluster"]), x="cluster"), use_container_width=True)

    st.markdown("#### Association Rules (top 50)")
    if rules.empty:
        st.info("No association rules file found or rules table is empty.")
    else:
        st.dataframe(rules.head(50), use_container_width=True)

# -----------------------
# Models Tab
# -----------------------
with tabs[3]:
    st.markdown("### Supervised Models — Metrics & Comparison")
    metrics = {}
    if files["metrics"]:
        try:
            metrics = json.loads(Path(files["metrics"]).read_text())
        except Exception as e:
            st.warning(f"Could not read metrics JSON: {e}")

    if not metrics:
        st.info("Metrics JSON not found. Run the pipeline first.")
    else:
        reg = metrics.get("regression", {})
        clf = metrics.get("classification", {})

        if reg:
            st.subheader("Regression (PM2.5)")
            reg_df = pd.DataFrame(reg).T.reset_index().rename(columns={"index": "Model"})
            st.dataframe(reg_df, use_container_width=True)

        if clf:
            st.subheader("Classification (AQI category)")
            clf_df = pd.DataFrame(clf).T.reset_index().rename(columns={"index": "Model"})
            st.dataframe(clf_df, use_container_width=True)

# -----------------------
# Forecast Tab
# -----------------------
with tabs[4]:
    st.markdown("### PM2.5 Forecast")
    fdf = load_csv_safe(files["forecast"], parse_dates=["ds"])
    if fdf.empty:
        fdf = load_csv_safe(files["forecast"], parse_dates=["datetime"])

    if fdf.empty:
        st.info("No forecast file found yet.")
    else:
        # Normalize columns
        if "ds" in fdf.columns:
            fdf = fdf.rename(columns={"ds": "datetime"})
        ycols = []
        if "yhat" in fdf.columns:
            ycols = ["yhat"]
            if "yhat_lower" in fdf.columns and "yhat_upper" in fdf.columns:
                st.area_chart(fdf.set_index("datetime")[["yhat_lower", "yhat_upper"]])
        elif "pm2_5" in fdf.columns:
            ycols = ["pm2_5"]
        else:
            # fallback to first numeric column
            ycols = fdf.select_dtypes(include=np.number).columns.tolist()[:1]

        if "datetime" in fdf.columns and ycols:
            st.plotly_chart(px.line(fdf, x="datetime", y=ycols), use_container_width=True)
        st.dataframe(fdf.tail(50), use_container_width=True)

# -----------------------
# Anomalies Tab
# -----------------------
with tabs[5]:
    st.markdown("### Anomaly Detection")
    adf = load_csv_safe(files["anomalies"], parse_dates=["datetime"])
    if adf.empty:
        st.info("No anomaly file found (data/processed/*_anomalies.csv).")
    else:
        st.dataframe(adf.tail(100), use_container_width=True)
        # Try a simple visualization if pm2_5 present and maybe is_anomaly
        if "pm2_5" in adf.columns:
            fig = px.scatter(adf, x="datetime", y="pm2_5",
                             color=adf["is_anomaly"].astype(str) if "is_anomaly" in adf.columns else None)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Health Tab
# -----------------------
with tabs[6]:
    st.markdown("### Health Risk Estimates")
    hdf = load_csv_safe(files["health"], parse_dates=["datetime"])
    if hdf.empty:
        st.info("No health file found (data/processed/*_health.csv).")
    else:
        st.dataframe(hdf.tail(50), use_container_width=True)
