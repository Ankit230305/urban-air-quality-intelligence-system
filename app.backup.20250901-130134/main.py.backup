"""City-aware Streamlit dashboard for the Urban Air Quality Intelligence System."""
import os, sys, pathlib, subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional mapping
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# ---- City presets ----
CITY_PRESETS = {
    "Delhi":            {"lat": 28.6139, "lon": 77.2090},
    "Hyderabad":        {"lat": 17.3850, "lon": 78.4867},
    "Chennai":          {"lat": 13.0827, "lon": 80.2707},
    "Mumbai":           {"lat": 19.0760, "lon": 72.8777},
    "Kolkata":          {"lat": 22.5726, "lon": 88.3639},
    "Bengaluru":        {"lat": 12.9716, "lon": 77.5946},
    "Vizag":            {"lat": 17.6868, "lon": 83.2185},
    "Vellore":          {"lat": 12.9165, "lon": 79.1325},  # keep your original
}

# ---- UI config ----
st.set_page_config(page_title="Urban Air Quality Intelligence System", layout="wide")
st.title("Urban Air Quality Intelligence System")

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")

preset_city = st.sidebar.selectbox("Preset city", list(CITY_PRESETS.keys()), index=list(CITY_PRESETS.keys()).index("Vellore") if "Vellore" in CITY_PRESETS else 0)
city = st.sidebar.text_input("City name", value=preset_city)
slug = city.lower().replace(" ", "_")

# Show preset lat/lon for the chosen city; allow overriding
default_lat = CITY_PRESETS.get(preset_city, {}).get("lat", 0.0)
default_lon = CITY_PRESETS.get(preset_city, {}).get("lon", 0.0)
lat = st.sidebar.number_input("Latitude", value=float(default_lat), format="%.6f")
lon = st.sidebar.number_input("Longitude", value=float(default_lon), format="%.6f")

# Dates (string inputs; we make them robust in code)
start_date = st.sidebar.text_input("Start date (YYYY/MM/DD)", value="2024/08/17")
end_date   = st.sidebar.text_input("End date (YYYY/MM/DD)",   value="2024/08/24")

source_choice = st.sidebar.selectbox("Data source", ["Auto (build if needed)", "Load from processed CSV"], index=0)

# Find the best default processed CSV for this city
def find_processed_for_city(slug: str) -> Path:
    proc_dir = Path("data/processed")
    if not proc_dir.exists():
        return Path("data/processed/merged_dataset.csv")
    pats = (
        f"*{slug}*features_plus_demo*.csv",
        f"*{slug}*features*.csv",
        f"*{slug}*processed*.csv",
        f"*{slug}*.csv",
        "*features_plus_demo*.csv",
        "*features*.csv",
        "*processed*.csv",
        "*.csv",
    )
    candidates = []
    for pat in pats:
        candidates += list(proc_dir.glob(pat))
    candidates = [c for c in candidates if c.is_file()]
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else Path("data/processed/merged_dataset.csv")

default_processed = find_processed_for_city(slug)
processed_path = st.sidebar.text_input("Processed CSV path", value=str(default_processed))

colA, colB = st.sidebar.columns(2)
build_btn = colA.button("Build/Refresh data")
load_btn  = colB.button("Load data")

# ---------------- Helpers ----------------
def read_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    try:
        df = pd.read_csv(p, parse_dates=["datetime"])
    except Exception:
        df = pd.read_csv(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df

def filter_by_dates(df: pd.DataFrame, start_s: str, end_s: str) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    df = df.copy()
    df["datetime"] = dt
    start = pd.to_datetime(start_s, errors="coerce")
    end   = pd.to_datetime(end_s,   errors="coerce")
    try:
        start = start.tz_localize(None)
    except Exception:
        pass
    try:
        end = end.tz_localize(None)
    except Exception:
        pass
    if pd.isna(start) or pd.isna(end):
        return df
    if end < start:
        start, end = end, start
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    out = df.loc[mask]
    if out.empty:
        st.warning("No rows in this date range; showing full range.")
        return df
    return out

def show_map(df: pd.DataFrame, lat: float, lon: float):
    if {"latitude", "longitude"}.issubset(df.columns):
        pts = df[["latitude","longitude"]].dropna().drop_duplicates()
        if pts.empty:
            pts = pd.DataFrame({"latitude":[lat], "longitude":[lon]})
    else:
        pts = pd.DataFrame({"latitude":[lat], "longitude":[lon]})
    if HAS_FOLIUM:
        m = folium.Map(location=[pts["latitude"].iloc[0], pts["longitude"].iloc[0]], zoom_start=11)
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
    st.line_chart(df.set_index("datetime")[avail])

def scatter_pm25(df: pd.DataFrame, cols: list):
    import plotly.express as px
    if "pm2_5" not in df.columns:
        st.info("PM2.5 column not present for scatter plots.")
        return
    for c in cols:
        if c in df.columns:
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

def make_aqi_cat(aqi):
    try:
        aqi = float(aqi)
    except Exception:
        return np.nan
    if aqi < 51: return "Good"
    if aqi < 101: return "Moderate"
    if aqi < 201: return "Unhealthy"
    if aqi < 301: return "Very Unhealthy"
    return "Hazardous"

# ---------------- Main logic ----------------
status = st.empty()
df = pd.DataFrame()

# Build/refresh pipeline for this city if requested
if build_btn:
    with st.status(f"Building data for {city}…", expanded=True) as ststat:
        try:
            cmd = [
                "bash", "bin/run_city_pipeline.sh",
                city, str(lat), str(lon),
                start_date.replace("/", "-"),
                end_date.replace("/", "-"),
            ]
            st.write("Running:", " ".join(cmd))
            subprocess.check_call(cmd)
            ststat.update(label=f"Build complete for {city}.", state="complete")
            # update processed_path to the latest city file
            processed_path = str(find_processed_for_city(slug))
        except subprocess.CalledProcessError as e:
            ststat.update(label="Build failed.", state="error")
            st.error(f"Pipeline failed: {e}")

# Auto or manual load
if source_choice.startswith("Auto"):
    processed_path = str(find_processed_for_city(slug))

if load_btn or source_choice.startswith("Auto"):
    try:
        df = read_csv_safe(processed_path)
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Failed to load processed file: {e}")

st.caption(f"Using file: {processed_path}")
if not df.empty:
    df = df.sort_values("datetime")
    df = filter_by_dates(df, start_date, end_date)
    mn = pd.to_datetime(df["datetime"], errors="coerce").min()
    mx = pd.to_datetime(df["datetime"], errors="coerce").max()
    st.caption(f"Loaded file: {processed_path} | rows={len(df)} | datetime range: {mn} → {mx}")

# Current AQI snapshot (if present)
aqi_cols = [c for c in ["aqi","dominientpol","pm25","pm2_5","pm10","no2","o3","co","so2"] if c in df.columns]
if aqi_cols:
    st.dataframe(df[aqi_cols].head(1))
else:
    st.info("AQI fields not found; continuing with EDA…")

# Map
st.subheader("Location")
show_map(df, lat, lon)

# Tabs
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
    fpath = Path(f"models/forecast_pm25_{slug}.csv")
    if not fpath.exists():
        # fallback
        fpath = Path("models/forecast_pm25.csv")
    if fpath.exists():
        f = pd.read_csv(fpath, parse_dates=["ds"])
        st.markdown("### 7-day PM2.5 Forecast (Prophet)")
        st.line_chart(f.set_index("ds")[["yhat","yhat_lower","yhat_upper"]])
        st.dataframe(f.tail(10))
    else:
        st.info("No forecast file found. Click 'Build/Refresh data' to generate one.")

with tab_anom:
    apath = Path(f"data/processed/{slug}_anomalies.csv")
    if apath.exists():
        a = pd.read_csv(apath, parse_dates=["datetime"])
        st.markdown("### Detected Anomalies (spikes)")
        st.dataframe(a.tail(50))
    else:
        st.info("No anomaly file found. Click 'Build/Refresh data'.")

with tab_health:
    hpath = Path(f"data/processed/{slug}_health.csv")
    if hpath.exists():
        h = pd.read_csv(hpath, parse_dates=["datetime"])
        st.markdown("### Health Risk Estimates")
        st.dataframe(h.tail(50))
    else:
        st.info("No health risk file found. Click 'Build/Refresh data'.")
