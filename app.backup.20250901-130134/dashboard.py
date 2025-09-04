from __future__ import annotations
import os, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Urban Air Quality Intelligence System", layout="wide")

CITIES = {
    "Delhi":      {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":     {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru":  {"lat": 12.9716, "lon": 77.5946},
    "Chennai":    {"lat": 13.0827, "lon": 80.2707},
    "Hyderabad":  {"lat": 17.3850, "lon": 78.4867},
    "Kolkata":    {"lat": 22.5726, "lon": 88.3639},
    "Vizag":      {"lat": 17.6868, "lon": 83.2185},
    "Vellore":    {"lat": 12.9165, "lon": 79.1325},
}

def slug_of(city: str) -> str:
    return city.strip().lower().replace(" ", "_")

def _first_present_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_num(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return float(default)
        if isinstance(val, (list, tuple)) and val:
            val = val[0]
        x = pd.to_numeric(val, errors="coerce")
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_pick(df: pd.DataFrame, candidates: Iterable[str], row: int = 0, default: float = 0.0) -> float:
    col = _first_present_column(df, candidates)
    if not col or df.empty:
        return float(default)
    try:
        return safe_num(df[col].iloc[row], default=default)
    except Exception:
        return float(default)

def aqi_from_pm25(pm25: float) -> float:
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for Cl, Ch, Il, Ih in bps:
        if pm25 <= Ch:
            return (Ih - Il) / (Ch - Cl) * (pm25 - Cl) + Il
    return 500.0

def aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def latest_live_path(city: str) -> Optional[Path]:
    live_dir = Path("data/live")
    if not live_dir.exists(): return None
    files = list(live_dir.glob(f"{slug_of(city)}_live_*.csv"))
    if not files: return None
    return max(files, key=lambda p: p.stat().st_mtime)

def load_live(city: str) -> pd.DataFrame:
    p = latest_live_path(city)
    if not p: return pd.DataFrame()
    df = pd.read_csv(p)
    if "pm2.5" in df.columns:  # purpleair normalization
        df = df.rename(columns={"pm2.5": "pm2_5"})
    for cand in ("datetime", "timestamp", "time"):
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            break
    return df

def run_fetch_live_now(city: str, lat: float, lon: float) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    cmd = [sys.executable, "bin/fetch_live_now.py", "--city", city, "--lat", str(lat), "--lon", str(lon)]
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        return (out.returncode == 0), ((out.stdout or "") + (out.stderr or "")).strip()
    except FileNotFoundError:
        return False, "bin/fetch_live_now.py not found. Make sure it exists."
    except Exception as e:
        return False, f"Error: {e}"

st.sidebar.header("Urban Air Quality Intelligence")
city = st.sidebar.selectbox("City", list(CITIES.keys()), index=1)
lat = CITIES[city]["lat"]; lon = CITIES[city]["lon"]

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("🔄 Fetch live now", use_container_width=True):
        ok, msg = run_fetch_live_now(city, lat, lon)
        st.success("Live data fetched", icon="✅") if ok else st.error(msg, icon="⚠️")
with c2:
    st.caption(f"Lat/Lon: {lat:.4f}, {lon:.4f}")

st.title("Urban Air Quality Intelligence System")
st.caption("Live monitoring • EDA & patterns • Forecast & health insights")

st.subheader("Overview (Live)")
df_live = load_live(city)
if df_live.empty:
    st.warning("No live file found yet. Click 'Fetch live now' to create one.", icon="ℹ️")
else:
    pm25 = safe_pick(df_live, ["pm2_5","pm25","pm2p5","pm2_5_10minute","pm2.5"], default=0.0)
    pm10 = safe_pick(df_live, ["pm10"], default=0.0)
    aqi_raw = safe_pick(df_live, ["aqi"], default=np.nan)
    aqi = safe_num(aqi_raw, default=aqi_from_pm25(pm25))
    temp = safe_pick(df_live, ["temp","temperature"], default=0.0)
    humidity = safe_pick(df_live, ["humidity","rh"], default=0.0)
    wind = safe_pick(df_live, ["wind_speed","windspeed","w"], default=0.0)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("AQI", f"{aqi:.0f}", aqi_category(aqi))
    m2.metric("PM2.5 (µg/m³)", f"{pm25:.1f}")
    m3.metric("PM10 (µg/m³)", f"{pm10:.1f}")
    m4.metric("Temp (°C)", f"{temp:.1f}")
    m5.metric("Humidity (%)", f"{humidity:.0f}")
    m6.metric("Wind (km/h)", f"{wind:.1f}")

    st.map(pd.DataFrame({"lat":[lat], "lon":[lon]}), zoom=10)

    with st.expander("Show raw live row", expanded=False):
        st.dataframe(df_live.head(10), use_container_width=True)

st.subheader("Exploratory Data (Processed)")
proc_path = Path("data/processed") / f"{slug_of(city)}__features.csv"
if proc_path.exists():
    dfp = pd.read_csv(proc_path)
    tcol = _first_present_column(dfp, ["datetime","ds","timestamp","time"])
    if tcol: dfp[tcol] = pd.to_datetime(dfp[tcol], errors="coerce")

    if tcol and "pm2_5" in dfp.columns:
        fig = px.line(dfp.sort_values(tcol), x=tcol, y="pm2_5", title=f"{city} • PM2.5 over time")
        st.plotly_chart(fig, use_container_width=True)

    catcol = _first_present_column(dfp, ["aqi_category","aqi_cat"])
    if catcol:
        counts = dfp[catcol].value_counts(dropna=False).reset_index()
        counts.columns = ["category","count"]
        fig2 = px.bar(counts, x="category", y="count", title=f"{city} • AQI categories")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Show processed sample", expanded=False):
        st.dataframe(dfp.head(200), use_container_width=True)
else:
    st.info("No processed file found for this city yet. Run your pipeline to generate EDA inputs.", icon="ℹ️")

st.markdown(f"**Last refreshed:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
