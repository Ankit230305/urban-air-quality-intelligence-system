"""Streamlit dashboard for the Urban Air Quality Intelligence System."""
import os, sys, subprocess, textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Make project importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional mapping
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# ---------- helpers ----------
CITY_PRESETS = {
    "Vellore":   (12.9165, 79.1325),
    "Chennai":   (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.6139, 77.2090),
    "Kolkata":   (22.5726, 88.3639),
    "Vizag":     (17.6868, 83.2185),
}

POLLUTANTS = ["pm2_5","pm10","no2","o3","so2","co"]

def slug_of(city: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in city.lower())

def find_processed_for_city(slug: str) -> Path|None:
    proc = ROOT / "data" / "processed"
    if not proc.exists():
        return None
    pats = [
        f"*{slug}*features_plus_demo*.csv",
        f"*{slug}*features*.csv",
        f"*{slug}*.csv",
    ]
    cands = []
    for pat in pats:
        cands += list(proc.glob(pat))
    cands = [c for c in cands if c.is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def read_csv_safe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    # normalize tz
    try:
        df["datetime"] = df["datetime"].dt.tz_convert(None)
    except Exception:
        try:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        except Exception:
            pass
    # coerce numerics
    for c in POLLUTANTS + ["temp","humidity","wind_speed","precip","aqi","latitude","longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop rows where ALL pollutants are missing
    if set(POLLUTANTS).intersection(df.columns):
        mask = pd.DataFrame({c: df[c].notna() for c in POLLUTANTS if c in df.columns}).any(axis=1)
        df = df.loc[mask].copy()
    return df.sort_values("datetime")

def filter_by_dates(df: pd.DataFrame, start_s: str, end_s: str) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    start = pd.to_datetime(start_s, errors="coerce")
    end   = pd.to_datetime(end_s,   errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return df
    # make naive
    try:
        s = df["datetime"]
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)
        df = df.assign(datetime=s)
    except Exception:
        pass
    m = (df["datetime"] >= start) & (df["datetime"] <= end)
    return df.loc[m].copy()

def show_map(df: pd.DataFrame, lat: float, lon: float):
    if {"latitude","longitude"}.issubset(df.columns):
        pts = df[["latitude","longitude"]].dropna().drop_duplicates()
        if pts.empty:
            pts = pd.DataFrame({"latitude":[lat], "longitude":[lon]})
    else:
        pts = pd.DataFrame({"latitude":[lat], "longitude":[lon]})
    if HAS_FOLIUM:
        m = folium.Map(location=[pts["latitude"].iloc[0], pts["longitude"].iloc[0]], zoom_start=11)
        for _, r in pts.iterrows():
            folium.CircleMarker([r["latitude"], r["longitude"]], radius=4, color="#3186cc").add_to(m)
        st_folium(m, height=300, width=None)
    else:
        st.map(pts.rename(columns={"latitude":"lat","longitude":"lon"}))

def line_section(df: pd.DataFrame, cols: list, title: str):
    avail = [c for c in cols if c in df.columns]
    if not avail or df.empty:
        st.info(f"No columns available for {title}.")
        return
    st.subheader(title)
    st.line_chart(df.set_index("datetime")[avail])

def scatter_pm25(df: pd.DataFrame, cols: list):
    import plotly.express as px
    if "pm2_5" not in df.columns or df["pm2_5"].dropna().empty:
        st.info("PM2.5 column is empty.")
        return
    for c in cols:
        if c in df.columns and not df[c].dropna().empty:
            st.markdown(f"**PM2.5 vs {c}**")
            fig = px.scatter(df, x=c, y="pm2_5", opacity=0.6, trendline="ols")
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

# ---------- UI ----------
st.set_page_config(page_title="Urban Air Quality Intelligence System", layout="wide")
st.title("Urban Air Quality Intelligence System")

with st.sidebar:
    st.header("Configuration")

    city = st.selectbox("City name", list(CITY_PRESETS.keys()), index=list(CITY_PRESETS.keys()).index("Vellore"))
    lat_default, lon_default = CITY_PRESETS[city]
    lat = st.number_input("Latitude", value=float(lat_default), format="%.6f")
    lon = st.number_input("Longitude", value=float(lon_default), format="%.6f")

    start_date = st.text_input("Start date (YYYY/MM/DD)", value="2024/08/17")
    end_date   = st.text_input("End date (YYYY/MM/DD)",   value="2024/08/24")

    data_source = st.selectbox("Data source", ["Auto (build if needed)", "Load from processed CSV only"], index=0)

    # pick latest file BEFORE rendering the text_input to avoid session_state overwrite errors
    default_processed = find_processed_for_city(slug_of(city)) or (ROOT/"data/processed/merged_dataset.csv")
    processed_path = st.text_input("Processed CSV path", value=str(default_processed))

    colA, colB = st.columns(2)
    do_build = colA.button("Build/Refresh data")
    do_load  = colB.button("Load data")

# optional: run the pipeline when requested
if do_build and data_source.startswith("Auto"):
    cmd = [
        "bash", str(ROOT/"bin/run_city_pipeline.sh"),
        city, f"{lat}", f"{lon}",
        start_date.replace("/","-"), end_date.replace("/","-")
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    with st.spinner(f"Building data for {city}…"):
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, check=False)
            st.code(proc.stdout or "", language="bash")
            if proc.stderr:
                st.code(proc.stderr, language="bash")
        except Exception as e:
            st.error(f"Build failed: {e}")

# ---------- Load data ----------
error_placeholder = st.empty()
loaded_path = None
df = pd.DataFrame()

if do_load:
    try:
        df = read_csv_safe(Path(processed_path))
        loaded_path = Path(processed_path)
    except Exception as e:
        error_placeholder.error(f"Failed to load processed file: {e}")

if df.empty:  # fallback to latest for the city
    try:
        latest = find_processed_for_city(slug_of(city))
        if latest:
            df = read_csv_safe(latest)
            loaded_path = latest
    except Exception as e:
        error_placeholder.error(str(e))

if df.empty:
    st.warning("No processed data was loaded. Click **Build/Refresh data**, then **Load data**.")
    st.stop()

# Filter to date window visible to user
df = filter_by_dates(df, start_date, end_date)

# Show what file we loaded
st.caption(f"Loaded: `{loaded_path}`  • rows={len(df)}")

# Quick top metrics if available
top_cols = [c for c in ["aqi","pm2_5","pm10","no2","o3","co","so2"] if c in df.columns]
if top_cols and not df[top_cols].dropna(how="all").empty:
    st.dataframe(df[top_cols].head(1))
else:
    st.info("No AQI/pollutant metrics found in this slice.")

# Map
st.subheader("Location"); show_map(df, lat, lon)

# Tabs
tab_eda, tab_forecast, tab_anom, tab_health = st.tabs(["EDA", "Forecast", "Anomalies", "Health"])

with tab_eda:
    st.markdown("### Time Series (Pollutants)"); line_section(df, POLLUTANTS, "Pollutants")
    st.markdown("### Time Series (Weather)");    line_section(df, ["temp","humidity","wind_speed","precip"], "Weather")
    st.markdown("### Weather ↔ PM2.5");          scatter_pm25(df, ["temp","humidity","wind_speed","precip"])
    st.markdown("### Correlations");             corr_heatmap(df)

with tab_forecast:
    fpath = ROOT / "models" / f"forecast_pm25_{slug_of(city)}.csv"
    if fpath.exists():
        f = pd.read_csv(fpath, parse_dates=["ds"])
        st.markdown("### 7-day PM2.5 Forecast (Prophet)")
        st.line_chart(f.set_index("ds")[["yhat","yhat_lower","yhat_upper"]])
        st.dataframe(f.tail(10))
    else:
        st.info(f"No forecast found at {fpath}. Click **Build/Refresh data**.")

with tab_anom:
    apath = ROOT / "data" / "processed" / f"{slug_of(city)}_anomalies.csv"
    if apath.exists():
        a = pd.read_csv(apath, parse_dates=["datetime"])
        st.markdown("### Detected Pollution Spikes"); st.dataframe(a.tail(50))
    else:
        st.info("No anomaly file found. Click **Build/Refresh data**.")

with tab_health:
    hpath = ROOT / "data" / "processed" / f"{slug_of(city)}_health.csv"
    if hpath.exists():
        h = pd.read_csv(hpath, parse_dates=["datetime"])
        st.markdown("### Health Risk Estimates & Advisories"); st.dataframe(h.tail(50))
    else:
        st.info("No health file found. Click **Build/Refresh data**.")
