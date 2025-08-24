"""Streamlit dashboard for the Urban Air Quality Intelligence System (tz-safe)."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import streamlit as st

# Optional mapping
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

st.set_page_config(page_title="Urban Air Quality Intelligence System", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")
city = st.sidebar.text_input("City name", value="Vellore")
lat  = st.sidebar.number_input("Latitude", value=12.9165, format="%.6f")
lon  = st.sidebar.number_input("Longitude", value=79.1325, format="%.6f")
start_date = st.sidebar.text_input("Start date", value="2024/08/17")
end_date   = st.sidebar.text_input("End date",   value="2024/08/24")
source_choice = st.sidebar.selectbox("Data source", ["Load from processed CSV", "Fetch live now"], index=0)

# Default processed path; auto-pick latest if missing
default_processed = Path("data/processed/merged_dataset.csv")
if not default_processed.exists():
    proc_dir = Path("data/processed")
    cands = []
    if proc_dir.exists():
        for pat in ("*features*.csv","*processed*.csv","*merged*.csv","*.csv"):
            cands += list(proc_dir.glob(pat))
    cands = [c for c in cands if c.is_file()]
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    if cands:
        default_processed = cands[0]

processed_path = st.sidebar.text_input("Processed CSV path", value=str(default_processed))
load_btn = st.sidebar.button("Load data")

st.title("Urban Air Quality Intelligence System")

# ---------------- Helpers ----------------
def _strip_tz(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    # Remove timezone if present
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt

def read_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    try:
        df = pd.read_csv(p, parse_dates=["datetime"])
    except Exception:
        df = pd.read_csv(p)
    if "datetime" in df.columns:
        df["datetime"] = _strip_tz(df["datetime"])
    return df

def filter_by_dates(df: pd.DataFrame, start_s: str, end_s: str) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    df = df.copy()
    df["datetime"] = _strip_tz(df["datetime"])
    df["_date"] = df["datetime"].dt.date  # compare as plain dates (tz-proof)

    s = pd.to_datetime(start_s, errors="coerce")
    e = pd.to_datetime(end_s,   errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return df
    s_d, e_d = s.date(), e.date()
    out = df[(df["_date"] >= s_d) & (df["_date"] <= e_d)].copy()
    return out.drop(columns=["_date"])

def show_map(df: pd.DataFrame, lat: float, lon: float):
    if {"latitude","longitude"}.issubset(df.columns):
        pts = df[["latitude","longitude"]].dropna().drop_duplicates()
    else:
        pts = pd.DataFrame({"latitude":[lat], "longitude":[lon]})
    if HAS_FOLIUM and not pts.empty:
        m = folium.Map(location=[pts["latitude"].iloc[0], pts["longitude"].iloc[0]], zoom_start=12)
        for _, r in pts.iterrows():
            folium.CircleMarker([r["latitude"], r["longitude"]], radius=4, color="#3186cc").add_to(m)
        st_folium(m, height=350, width=None)
    else:
        st.map(pts.rename(columns={"latitude":"lat","longitude":"lon"}))

def line_section(df: pd.DataFrame, cols: list, title: str):
    avail = [c for c in cols if c in df.columns]
    if not avail:
        st.info(f"No columns available for {title}: expected {cols}")
        return
    st.subheader(title)
    if "datetime" in df.columns:
        st.line_chart(df.set_index("datetime")[avail])
    else:
        st.line_chart(df[avail])

def scatter_pm25(df: pd.DataFrame, cols: list):
    import plotly.express as px
    if "pm2_5" not in df.columns:
        st.info("PM2.5 not present for scatter plots.")
        return
    for c in cols:
        if c in df.columns:
            st.markdown(f"**PM2.5 vs {c}**")
            fig = px.scatter(df, x=c, y="pm2_5", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

def corr_heatmap(df: pd.DataFrame):
    import plotly.express as px
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        st.info("No numeric columns for correlation heatmap.")
        return
    corr = num.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Main ----------------
error_placeholder = st.empty()
df = pd.DataFrame()

if source_choice == "Load from processed CSV":
    if load_btn:
        try:
            df = read_csv_safe(processed_path)
        except FileNotFoundError as e:
            error_placeholder.error(str(e))
        except Exception as e:
            error_placeholder.error(f"Failed to load processed file: {e}")
else:
    st.info("Live fetch disabled. Use 'Load from processed CSV'.")

# Load default even if button not pressed
if df.empty and Path(processed_path).exists():
    df = read_csv_safe(processed_path)

# Filter window (tz-proof)
if not df.empty:
    df = filter_by_dates(df, start_date, end_date)

# Current AQI preview (if present)
aqi_cols = [c for c in ["aqi","dominientpol","pm25","pm2_5","pm10","no2","o3","co","so2"] if c in df.columns]
if aqi_cols and not df.empty:
    st.subheader("Current AQI")
    st.dataframe(df[aqi_cols].head(1))
else:
    st.info("AQI fields not found; continuing with EDA…")

# Map
st.subheader("Location")
show_map(df, lat, lon)

tab_eda, tab_forecast, tab_anom, tab_health = st.tabs(["EDA", "Forecast", "Anomalies", "Health"])

with tab_eda:
    st.markdown("### Time Series (Pollutants)")
    line_section(df, ["pm2_5","pm10","no2","o3","so2","co"], "Pollutants")

    st.markdown("### Time Series (Weather)")
    line_section(df, ["temp","humidity","wind_speed","precip"], "Weather")

    st.markdown("### Weather ↔ PM2.5")
    scatter_pm25(df, ["temp","humidity","wind_speed","precip"])

    st.markdown("### Correlations")
    corr_heatmap(df)

with tab_forecast:
    fpath = Path("models/forecast_pm25.csv")
    if fpath.exists():
        f = pd.read_csv(fpath, parse_dates=["ds"])
        st.markdown("### 7-day PM2.5 Forecast (Prophet)")
        st.line_chart(f.set_index("ds")[["yhat","yhat_lower","yhat_upper"]])
        st.dataframe(f.tail(10))
    else:
        st.info("No forecast file found at models/forecast_pm25.csv. Train it first.")

with tab_anom:
    apath = Path("data/processed/vellore_anomalies.csv")
    if apath.exists():
        a = pd.read_csv(apath, parse_dates=["datetime"])
        st.markdown("### Detected Anomalies (spikes)")
        st.dataframe(a.tail(30))
    else:
        st.info("No anomaly file found. Run detect_anomalies script.")

with tab_health:
    hpath = Path("data/processed/vellore_health.csv")
    if hpath.exists():
        h = pd.read_csv(hpath, parse_dates=["datetime"])
        st.markdown("### Health Risk Estimates")
        st.dataframe(h.tail(30))
    else:
        st.info("No health risk file found. Run health_risk script.")
