# app/main.py
# Urban Air Quality Intelligence System - Streamlit Dashboard (refined)
# Uses 4 keys from env:
#   OPENWEATHERMAP_API_KEY, VISUAL_CROSSING_API_KEY, WAQI_TOKEN, PURPLEAIR_API_KEY

from __future__ import annotations

import os
import math
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pydeck as pdk

# =============== Paths / Constants ===============
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
LIVE_DIR = DATA_DIR / "live"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

for _d in (DATA_DIR, RAW_DIR, PROC_DIR, LIVE_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Vizag": (17.6868, 83.2185),
    "Vellore": (12.9165, 79.1325),
}

warnings.filterwarnings("ignore", category=FutureWarning)

# =============== Small utils ===============
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def slug_of(city: str) -> str:
    return city.lower().replace(" ", "_")

def coalesce(*vals, default=None):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        return v
    return default

def safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return float(x)
        v = float(str(x).strip())
        return v if not math.isnan(v) else default
    except Exception:
        return default

def read_csv_safe(p: Path, **kwargs) -> pd.DataFrame:
    try:
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p, **kwargs)
    except Exception:
        return pd.DataFrame()

def safe_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def display_df(df: pd.DataFrame, note: str = "Values filled for display"):
    """Show df with all NaN replaced so no blanks appear in the UI."""
    if df.empty:
        st.info("No rows to display.", icon="ℹ️")
        return
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].fillna(0)
        else:
            df2[c] = df2[c].fillna("NA")
    st.dataframe(df2)
    st.caption(f"※ {note}")

def plotly_safe(fig):
    """Handle Streamlit versions (new width='stretch' vs legacy use_container_width)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

# =============== AQI helpers (PM2.5 fallback) ===============
_PM25_BREAKS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def aqi_from_pm25(pm25: Optional[float]) -> Optional[float]:
    if pm25 is None:
        return None
    for c_lo, c_hi, a_lo, a_hi in _PM25_BREAKS:
        if c_lo <= pm25 <= c_hi:
            return (a_hi - a_lo) / (c_hi - c_lo) * (pm25 - c_lo) + a_lo
    return None

def aqi_category(aqi: Optional[float]) -> str:
    if aqi is None:
        return "Unknown"
    v = float(aqi)
    if v <= 50: return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy (SG)"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"

# =============== API keys ===============
OWM_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
VC_KEY = os.environ.get("VISUAL_CROSSING_API_KEY")
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")
PA_KEY = os.environ.get("PURPLEAIR_API_KEY")

# =============== Live API clients (cached) ===============
@st.cache_data(ttl=300)
def live_owm(lat: float, lon: float) -> Dict[str, Any]:
    if not OWM_KEY:
        return {}
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    r = requests.get(url, params={"lat": lat, "lon": lon, "appid": OWM_KEY}, timeout=15)
    return r.json() if r.status_code == 200 else {}

@st.cache_data(ttl=300)
def live_vc(lat: float, lon: float) -> Dict[str, Any]:
    if not VC_KEY:
        return {}
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}"
    r = requests.get(url, params={"unitGroup": "metric", "include": "current", "key": VC_KEY, "contentType": "json"}, timeout=15)
    return r.json() if r.status_code == 200 else {}

@st.cache_data(ttl=300)
def live_waqi(lat: float, lon: float) -> Dict[str, Any]:
    if not WAQI_TOKEN:
        return {}
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
    r = requests.get(url, params={"token": WAQI_TOKEN}, timeout=15)
    return r.json() if r.status_code == 200 else {}

@st.cache_data(ttl=300)
def live_purpleair(lat: float, lon: float) -> Dict[str, Any]:
    if not PA_KEY:
        return {}
    span = 0.35
    params = {
        "fields": "sensor_index,name,latitude,longitude,pm2.5,pm2.5_10minute,pm2.5_30minute,pm2.5_60minute,last_seen",
        "nwlng": lon - span, "nwlat": lat + span,
        "selng": lon + span, "selat": lat - span,
    }
    url = "https://api.purpleair.com/v1/sensors"
    r = requests.get(url, headers={"X-API-Key": PA_KEY}, params=params, timeout=15)
    return r.json() if r.status_code == 200 else {}

def waqi_get(waqi: Dict[str, Any], key: str) -> Optional[float]:
    try:
        iaqi = (waqi.get("data") or {}).get("iaqi") or {}
        vobj = iaqi.get(key) or {}
        return safe_float(vobj.get("v"))
    except Exception:
        return None

def unify_live(lat: float, lon: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch from 4 providers and coalesce into one row + raw dicts."""
    raw = {
        "owm": live_owm(lat, lon),
        "vc": live_vc(lat, lon),
        "waqi": live_waqi(lat, lon),
        "pa": live_purpleair(lat, lon),
    }

    pm25 = pm10 = no2 = o3 = so2 = co = None
    temp = humidity = wind = precip = None
    aqi = None

    # OWM components
    try:
        entry = (raw["owm"].get("list") or [])[0]
        comps = entry.get("components") or {}
        pm25 = coalesce(pm25, safe_float(comps.get("pm2_5")))
        pm10 = coalesce(pm10, safe_float(comps.get("pm10")))
        no2  = coalesce(no2,  safe_float(comps.get("no2")))
        o3   = coalesce(o3,   safe_float(comps.get("o3")))
        so2  = coalesce(so2,  safe_float(comps.get("so2")))
        co   = coalesce(co,   safe_float(comps.get("co")))
    except Exception:
        pass

    # WAQI AQI + pollutants if available
    try:
        if raw["waqi"].get("status") == "ok":
            aqi = coalesce(aqi, safe_float((raw["waqi"]["data"] or {}).get("aqi")))
            pm25 = coalesce(pm25, waqi_get(raw["waqi"], "pm25"))
            pm10 = coalesce(pm10, waqi_get(raw["waqi"], "pm10"))
            no2  = coalesce(no2,  waqi_get(raw["waqi"], "no2"))
            o3   = coalesce(o3,   waqi_get(raw["waqi"], "o3"))
            so2  = coalesce(so2,  waqi_get(raw["waqi"], "so2"))
            co   = coalesce(co,   waqi_get(raw["waqi"], "co"))
    except Exception:
        pass

    # Visual Crossing weather
    try:
        cc = raw["vc"].get("currentConditions") or {}
        temp = coalesce(temp, safe_float(cc.get("temp")))
        humidity = coalesce(humidity, safe_float(cc.get("humidity")))
        wind = coalesce(wind, safe_float(cc.get("windspeed")))
        precip = coalesce(precip, safe_float(cc.get("precip")))
    except Exception:
        pass

    # PurpleAir median
    try:
        data = raw["pa"].get("data") or []
        fields = raw["pa"].get("fields") or []
        def fidx(name): return fields.index(name) if name in fields else None
        i_lat = fidx("latitude"); i_lon = fidx("longitude")
        i_pm60 = fidx("pm2.5_60minute"); i_pm = fidx("pm2.5")
        vals = []
        for row in data:
            v = None
            if i_pm60 is not None:
                v = safe_float(row[i_pm60])
            if v is None and i_pm is not None:
                v = safe_float(row[i_pm])
            if v is not None:
                vals.append(v)
        if vals:
            pm25 = coalesce(pm25, float(np.median(vals)))
    except Exception:
        pass

    if aqi is None:
        aqi = aqi_from_pm25(pm25)

    df = pd.DataFrame([{
        "datetime": now_utc_iso(),
        "lat": lat, "lon": lon,
        "pm2_5": pm25, "pm10": pm10, "no2": no2, "o3": o3, "so2": so2, "co": co,
        "temp": temp, "humidity": humidity, "wind_speed": wind, "precip": precip,
        "aqi": aqi, "aqi_category": aqi_category(aqi),
    }])
    return df, raw

# =============== Loaders ===============
def load_city_processed(city: str) -> pd.DataFrame:
    p = PROC_DIR / f"{slug_of(city)}__features.csv"
    return read_csv_safe(p, parse_dates=["datetime"])

def load_city_anomalies(city: str) -> pd.DataFrame:
    p = PROC_DIR / f"{slug_of(city)}__anomalies.csv"
    return read_csv_safe(p, parse_dates=["datetime"])

def load_city_health(city: str) -> pd.DataFrame:
    p = PROC_DIR / f"{slug_of(city)}__health.csv"
    return read_csv_safe(p, parse_dates=["datetime"])

def load_city_reports(city: str) -> Dict[str, Any]:
    slug = slug_of(city)
    seasonal = read_csv_safe(REPORTS_DIR / f"seasonal_{slug}.csv")
    assoc = read_csv_safe(REPORTS_DIR / f"assoc_rules_{slug}.csv")
    md_path = REPORTS_DIR / f"patterns_{slug}.md"
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    return {"seasonal": seasonal, "assoc": assoc, "patterns_md": md}

def load_forecast() -> pd.DataFrame:
    return read_csv_safe(MODELS_DIR / "forecast_pm25.csv", parse_dates=["ds"])

# =============== UI helpers ===============
def city_selector(suffix: str) -> Tuple[str, float, float]:
    city = st.selectbox("City", list(CITIES.keys()), index=1, key=f"city_{suffix}")
    lat, lon = CITIES[city]
    return city, lat, lon

def kpi(label: str, value: Any, help_text: Optional[str] = None):
    st.metric(label, value if value is not None else "NA", help=help_text)

def draw_city_map(lat: float, lon: float, pa_raw: Dict[str, Any]):
    layers = []
    # City marker
    city_df = pd.DataFrame([{"lat": lat, "lon": lon}])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=city_df,
            get_position="[lon, lat]",
            get_color="[200, 30, 0, 160]",
            get_radius=800,
            pickable=False,
        )
    )
    # PurpleAir sensors
    try:
        data = pa_raw.get("data") or []
        fields = pa_raw.get("fields") or []
        if data and fields:
            i_lat = fields.index("latitude") if "latitude" in fields else None
            i_lon = fields.index("longitude") if "longitude" in fields else None
            i_pm  = fields.index("pm2.5_60minute") if "pm2.5_60minute" in fields else (fields.index("pm2.5") if "pm2.5" in fields else None)
            rows = []
            for row in data:
                la = safe_float(row[i_lat]) if i_lat is not None else None
                lo = safe_float(row[i_lon]) if i_lon is not None else None
                pm = safe_float(row[i_pm]) if (i_pm is not None) else None
                if la is None or lo is None:
                    continue
                rows.append({"lat": la, "lon": lo, "pm": pm})
            if rows:
                pa_df = pd.DataFrame(rows)
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=pa_df,
                        get_position="[lon, lat]",
                        get_fill_color="[255, 165, 0, 130]",
                        get_radius=250,
                        pickable=True,
                    )
                )
    except Exception:
        pass

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=9, pitch=0)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "PM2.5: {pm}"})
    st.pydeck_chart(deck)  # do NOT pass width as str (pydeck expects int)

def snapshot_live(city: str, lat: float, lon: float, df: pd.DataFrame, raw: Dict[str, Any]):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = slug_of(city)
    csv_path = LIVE_DIR / f"{slug}_live_{ts}.csv"
    json_path = LIVE_DIR / f"{slug}_live_{ts}.json"
    try:
        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        st.toast(f"Saved snapshot: {csv_path.name}", icon="✅")
    except Exception as e:
        st.toast(f"Snapshot save failed: {e}", icon="⚠️")

# =============== App ===============
st.set_page_config(page_title="Urban Air Quality Intelligence", layout="wide")

# Sidebar
st.sidebar.title("UAQI Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Monitor", "All Cities", "EDA", "Forecast", "Models & Scores", "Anomalies & Health", "Files & Reports"],
    index=0,
    key="nav",
)

with st.sidebar:
    st.caption("API keys (env):")
    st.code("OPENWEATHERMAP_API_KEY\nVISUAL_CROSSING_API_KEY\nWAQI_TOKEN\nPURPLEAIR_API_KEY", language="bash")
    st.caption(f"Now: {now_utc_iso()} UTC")
    cols = st.columns(4)
    cols[0].markdown(f"**OWM** {'✅' if OWM_KEY else '❌'}")
    cols[1].markdown(f"**VC** {'✅' if VC_KEY else '❌'}")
    cols[2].markdown(f"**WAQI** {'✅' if WAQI_TOKEN else '❌'}")
    cols[3].markdown(f"**PA** {'✅' if PA_KEY else '❌'}")
    if st.button("Clear live cache", use_container_width=True):
        live_owm.clear()
        live_vc.clear()
        live_waqi.clear()
        live_purpleair.clear()
        st.toast("Live cache cleared.", icon="🧹")

st.title("🌆 Urban Air Quality Intelligence System")

# =============== Pages ===============
if page == "Overview":
    st.subheader("Project Summary")
    st.write("Live monitoring + EDA + Forecast + Models + Health impact — with robust fallbacks (no blanks).")

    city, lat, lon = city_selector("overview")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**City Map & Sensors**")
        _, raw = unify_live(lat, lon)
        draw_city_map(lat, lon, raw.get("pa") or {})
    with c2:
        st.markdown("**Quick KPIs**")
        df_live, raw = unify_live(lat, lon)
        if df_live.empty:
            st.info("No live data yet (rate limit/keys?).", icon="ℹ️")
        else:
            row = df_live.iloc[0]
            kpi("AQI", int(round(safe_float(row["aqi"], 0) or 0)))
            kpi("Category", row.get("aqi_category", "Unknown"))
            kpi("PM2.5 (µg/m³)", safe_float(row.get("pm2_5"), 0))
            kpi("Temp (°C)", safe_float(row.get("temp"), 0))
            if st.button("Save live snapshot", use_container_width=True, key="snap_overview"):
                snapshot_live(city, lat, lon, df_live, raw)

    st.markdown("---")
    st.write("**On-disk availability**")
    slug = slug_of(city)
    c3, c4 = st.columns(2)
    with c3:
        st.write("Processed / Live")
        items = [
            (PROC_DIR / f"{slug}__features.csv", (PROC_DIR / f"{slug}__features.csv").exists()),
            (PROC_DIR / f"{slug}__anomalies.csv", (PROC_DIR / f"{slug}__anomalies.csv").exists()),
            (PROC_DIR / f"{slug}__health.csv", (PROC_DIR / f"{slug}__health.csv").exists()),
            (LIVE_DIR / f"{slug}_live_*.csv", any(LIVE_DIR.glob(f"{slug}_live_*.csv"))),
        ]
        for pth, ok in items:
            st.write(f"- {pth} {'✅' if ok else '❌'}")
    with c4:
        st.write("Reports / Models")
        for pth in [
            REPORTS_DIR / f"seasonal_{slug}.csv",
            REPORTS_DIR / f"assoc_rules_{slug}.csv",
            REPORTS_DIR / f"patterns_{slug}.md",
            MODELS_DIR / "pm25_regressor.joblib",
            MODELS_DIR / "aqi_classifier.joblib",
            MODELS_DIR / "forecast_pm25.csv",
        ]:
            st.write(f"- {pth} {'✅' if pth.exists() else '❌'}")

    st.caption(f"Last refreshed: {now_utc_iso()} UTC")

elif page == "Live Monitor":
    st.subheader("Live Air Quality & Weather")
    city, lat, lon = city_selector("live")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Map (city + PurpleAir)**")
        _, raw = unify_live(lat, lon)
        draw_city_map(lat, lon, raw.get("pa") or {})
    with right:
        df_live, raw = unify_live(lat, lon)
        if df_live.empty:
            st.info("No live response (try Clear cache, or check API limits).", icon="ℹ️")
        else:
            row = df_live.iloc[0].to_dict()
            st.markdown("**Current KPIs**")
            kpi("AQI", int(round(safe_float(row.get("aqi"), 0) or 0)))
            kpi("Category", row.get("aqi_category") or "Unknown")
            kpi("PM2.5", safe_float(row.get("pm2_5"), 0))
            kpi("PM10", safe_float(row.get("pm10"), 0))
            kpi("NO₂", safe_float(row.get("no2"), 0))
            kpi("O₃", safe_float(row.get("o3"), 0))
            kpi("SO₂", safe_float(row.get("so2"), 0))
            kpi("CO", safe_float(row.get("co"), 0))
            kpi("Temp °C", safe_float(row.get("temp"), 0))
            kpi("Humidity %", safe_float(row.get("humidity"), 0))
            kpi("Wind m/s", safe_float(row.get("wind_speed"), 0))
            kpi("Precip mm", safe_float(row.get("precip"), 0))
            if st.button("Save live snapshot", use_container_width=True, key="snap_live"):
                snapshot_live(city, lat, lon, pd.DataFrame([row]), raw)

    st.markdown("---")
    st.markdown("**Pollutants bar**")
    df_live, raw = unify_live(lat, lon)
    if not df_live.empty:
        r = df_live.iloc[0].to_dict()
        polls = {k: r.get(k) for k in ["pm2_5", "pm10", "no2", "o3", "so2", "co"]}
        poll_df = pd.DataFrame([{"pollutant": k, "value": safe_float(v, 0)} for k, v in polls.items()])
        fig = px.bar(poll_df, x="pollutant", y="value", title=f"{city} – Live pollutants")
        plotly_safe(fig)
    with st.expander("Raw API responses (debug)"):
        st.write("OWM Air:", raw.get("owm"))
        st.write("Visual Crossing:", raw.get("vc"))
        st.write("WAQI:", raw.get("waqi"))
        st.write("PurpleAir:", raw.get("pa"))
    st.caption(f"Last updated: {now_utc_iso()} UTC")

elif page == "All Cities":
    st.subheader("Live Snapshot – All Cities")
    rows = []
    raws = {}
    for name, (lat, lon) in CITIES.items():
        df, raw = unify_live(lat, lon)
        raws[name] = raw
        if df.empty:
            rows.append({"city": name, "aqi": 0, "aqi_category": "Unknown", "pm2_5": 0, "temp": 0})
        else:
            r = df.iloc[0].to_dict()
            rows.append({
                "city": name,
                "aqi": int(round(safe_float(r.get("aqi"), 0) or 0)),
                "aqi_category": r.get("aqi_category", "Unknown"),
                "pm2_5": safe_float(r.get("pm2_5"), 0),
                "temp": safe_float(r.get("temp"), 0),
            })
    table = pd.DataFrame(rows).sort_values("aqi")
    display_df(table, note="Missing numeric values shown as 0; categorical as 'NA'")

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    city, lat, lon = city_selector("eda")
    df = load_city_processed(city)
    if df.empty:
        st.info("No processed file for this city. Generate it via your pipeline.", icon="ℹ️")
    else:
        st.write(f"Rows: {len(df):,} | Cols: {len(df.columns)}")
        date_col = "datetime" if "datetime" in df.columns else ("ds" if "ds" in df.columns else None)
        if date_col:
            df = df.sort_values(date_col)
        num_cols = [c for c in ["pm2_5","pm10","no2","o3","so2","co","aqi","temp","humidity","wind_speed","precip"] if c in df.columns]

        # Distributions
        if num_cols:
            st.markdown("**Distributions**")
            for c in num_cols:
                fig = px.histogram(df.fillna({c:0}), x=c, nbins=50, title=f"{c} distribution")
                plotly_safe(fig)

        # Time-series
        if date_col and num_cols:
            st.markdown("**Time-series (last 2000 rows)**")
            ts = df[[date_col] + num_cols].copy()
            for c in num_cols:
                ts[c] = ts[c].fillna(0)
            if len(ts) > 2000:
                ts = ts.tail(2000)
            for c in num_cols:
                fig = px.line(ts, x=date_col, y=c, title=f"{c} over time")
                plotly_safe(fig)

        # Correlation
        if len(num_cols) >= 2:
            st.markdown("**Correlation heatmap**")
            corr = df[num_cols].fillna(0).corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, title="Correlation (numeric)")
            plotly_safe(fig)

        # AQI category counts
        catcol = next((c for c in ["aqi_category","aqi_cat","category"] if c in df.columns), None)
        if catcol:
            st.markdown("**AQI category counts**")
            counts = df[catcol].fillna("NA").value_counts().reset_index()
            counts.columns = ["category","count"]
            fig2 = px.bar(counts, x="category", y="count", title=f"{city} • AQI categories")
            plotly_safe(fig2)

        with st.expander("Sample (first 200 rows)"):
            display_df(df.head(200))

elif page == "Forecast":
    st.subheader("7-day Forecast")
    city, lat, lon = city_selector("forecast")
    fc = load_forecast()
    if fc.empty:
        # Fallback: synth forecast from current pm2.5 if available
        df_live, _ = unify_live(lat, lon)
        y = safe_float(df_live.iloc[0].get("pm2_5"), 0) if not df_live.empty else 0
        dates = pd.date_range(datetime.now(timezone.utc), periods=7, freq="D")
        fc = pd.DataFrame({"ds": dates, "yhat": [y]*7})
        st.info("Forecast file missing; showing naive 7-day constant forecast from current PM2.5.", icon="ℹ️")
    fig = px.line(fc, x="ds", y=[c for c in fc.columns if c != "ds"], title="Forecast")
    plotly_safe(fig)
    with st.expander("Forecast table"):
        display_df(fc.tail(300))

elif page == "Models & Scores":
    st.subheader("Models & Scores")
    paths = [
        MODELS_DIR / "pm25_regressor.joblib",
        MODELS_DIR / "aqi_classifier.joblib",
        MODELS_DIR / "forecast_pm25.csv",
    ]
    for p in paths:
        st.write(f"- {p} {'✅' if p.exists() else '❌'}")
    if (MODELS_DIR / "forecast_pm25.csv").exists():
        fc = load_forecast()
        if not fc.empty:
            fig = px.line(fc, x="ds", y=[c for c in fc.columns if c != "ds" and pd.api.types.is_numeric_dtype(fc[c])], title="Forecast overview")
            plotly_safe(fig)

elif page == "Anomalies & Health":
    st.subheader("Anomalies & Health")
    city, lat, lon = city_selector("ah")
    an = load_city_anomalies(city)
    hl = load_city_health(city)

    if an.empty and hl.empty:
        st.info("No anomalies/health files found for this city.", icon="ℹ️")
    else:
        if not an.empty:
            st.markdown("**Anomalies (sample)**")
            display_df(an.head(500))
        if not hl.empty:
            st.markdown("**Health impact (sample & lines)**")
            display_df(hl.head(500))
            xcol = "datetime" if "datetime" in hl.columns else ("ds" if "ds" in hl.columns else None)
            if xcol:
                for c in [c for c in hl.columns if c != xcol and pd.api.types.is_numeric_dtype(hl[c])][:4]:
                    fig = px.line(hl.fillna({c:0}).sort_values(xcol), x=xcol, y=c, title=f"{c} over time")
                    plotly_safe(fig)

elif page == "Files & Reports":
    st.subheader("Files & Reports")
    city, lat, lon = city_selector("files")
    slug = slug_of(city)
    rep = load_city_reports(city)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**seasonal_***")
        if not rep["seasonal"].empty:
            display_df(rep["seasonal"].head(500))
        else:
            st.info("Not found.", icon="ℹ️")
        st.markdown("**assoc_rules_***")
        if not rep["assoc"].empty:
            display_df(rep["assoc"].head(500))
        else:
            st.info("Not found.", icon="ℹ️")
    with c2:
        st.markdown("**patterns_*.md**")
        if rep["patterns_md"]:
            st.markdown(rep["patterns_md"])
        else:
            st.info("Not found.", icon="ℹ️")

    st.markdown("---")
    st.write("**Live CSV snapshots (latest):**")
    lives = sorted(LIVE_DIR.glob(f"{slug}_live_*.csv"))
    if lives:
        st.code(str(lives[-1].name))
        df_live_file = read_csv_safe(lives[-1])
        display_df(df_live_file.tail(50))
    else:
        st.info("No live snapshots saved for this city.", icon="ℹ️")