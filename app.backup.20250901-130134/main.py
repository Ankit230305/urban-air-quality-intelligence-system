from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pydeck as pdk


# --------------------------- Config ---------------------------

st.set_page_config(
    page_title="Urban AQ Intelligence – Live",
    page_icon="🌫️",
    layout="wide",
)

DATA_DIR = Path("data/live")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "Delhi":      (28.6139, 77.2090),
    "Mumbai":     (19.0760, 72.8777),
    "Bengaluru":  (12.9716, 77.5946),
    "Chennai":    (13.0827, 80.2707),
    "Hyderabad":  (17.3850, 78.4867),
    "Kolkata":    (22.5726, 88.3639),
    "Vizag":      (17.6868, 83.2185),
    "Vellore":    (12.9165, 79.1325),
}

OWM_KEY  = os.getenv("OPENWEATHERMAP_API_KEY", "")
VC_KEY   = os.getenv("VISUAL_CROSSING_API_KEY", "")
WAQI_KEY = os.getenv("WAQI_TOKEN", "")
PA_KEY   = os.getenv("PURPLEAIR_API_KEY", "")


# ----------------------- Utilities / Safety -------------------

def slug_of(city: str) -> str:
    return city.lower().replace(" ", "_")

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x, default: float | None = None) -> Optional[float]:
    """Best-effort number conversion; returns default (or None) if NA/invalid."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            if pd.isna(x):
                return default
            return float(x)
        x = str(x).strip()
        if x == "" or x.lower() in {"nan", "none", "na"}:
            return default
        return float(x)
    except Exception:
        return default

def aqi_category(aqi: Optional[float]) -> str:
    if aqi is None:
        return "Unknown"
    v = float(aqi)
    if v <= 50:
        return "Good"
    if v <= 100:
        return "Moderate"
    if v <= 150:
        return "Unhealthy for SG"
    if v <= 200:
        return "Unhealthy"
    if v <= 300:
        return "Very Unhealthy"
    return "Hazardous"

def aqi_from_pm25(pm: Optional[float]) -> Optional[float]:
    """US EPA breakpoints (simplified)."""
    if pm is None:
        return None
    pm = float(pm)
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for Cl, Ch, Il, Ih in bps:
        if Cl <= pm <= Ch:
            return round((Ih - Il) / (Ch - Cl) * (pm - Cl) + Il, 0)
    return None

def bbox(lat: float, lon: float, pad: float = 0.5) -> Dict[str, float]:
    return {
        "nwlat": lat + pad, "nwlng": lon - pad,
        "selat": lat - pad, "selng": lon + pad,
    }


# --------------------------- Fetchers -------------------------

@st.cache_data(ttl=120)
def fetch_owm_air(lat: float, lon: float) -> Dict[str, Any]:
    if not OWM_KEY:
        return {"error": "OPENWEATHERMAP_API_KEY missing"}
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    r = requests.get(url, params={"lat": lat, "lon": lon, "appid": OWM_KEY}, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def fetch_visualcrossing(lat: float, lon: float) -> Dict[str, Any]:
    if not VC_KEY:
        return {"error": "VISUAL_CROSSING_API_KEY missing"}
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}"
    r = requests.get(
        url,
        params={"unitGroup": "metric", "include": "current", "key": VC_KEY, "contentType": "json"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=180)
def fetch_waqi(lat: float, lon: float) -> Dict[str, Any]:
    if not WAQI_KEY:
        return {"error": "WAQI_TOKEN missing"}
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
    r = requests.get(url, params={"token": WAQI_KEY}, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=180)
def fetch_purpleair(lat: float, lon: float) -> Dict[str, Any]:
    if not PA_KEY:
        return {"error": "PURPLEAIR_API_KEY missing"}
    bb = bbox(lat, lon, pad=0.5)
    url = "https://api.purpleair.com/v1/sensors"
    params = {
        "fields": "name,latitude,longitude,pm2.5,pm2.5_10minute,pm2.5_30minute,pm2.5_60minute,last_seen",
        "nwlng": bb["nwlng"], "nwlat": bb["nwlat"], "selng": bb["selng"], "selat": bb["selat"],
    }
    r = requests.get(url, headers={"X-API-Key": PA_KEY}, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# -------------------------- Composition ----------------------

@dataclass
class LivePoint:
    city: str
    lat: float
    lon: float
    aqi: Optional[float]
    aqi_cat: str
    pm2_5: Optional[float]
    pm10: Optional[float]
    no2: Optional[float]
    o3: Optional[float]
    so2: Optional[float]
    co: Optional[float]
    temp_c: Optional[float]
    humidity: Optional[float]
    wind_kph: Optional[float]
    fetched_at: datetime

def compose_live(city: str, lat: float, lon: float) -> Tuple[LivePoint, pd.DataFrame, Dict[str, Any]]:
    raw: Dict[str, Any] = {}
    # OWM
    try:
        raw["owm"] = fetch_owm_air(lat, lon)
    except Exception as e:
        raw["owm"] = {"error": str(e)}
    # VC
    try:
        raw["vc"] = fetch_visualcrossing(lat, lon)
    except Exception as e:
        raw["vc"] = {"error": str(e)}
    # WAQI
    try:
        raw["waqi"] = fetch_waqi(lat, lon)
    except Exception as e:
        raw["waqi"] = {"error": str(e)}
    # PurpleAir
    try:
        raw["pa"] = fetch_purpleair(lat, lon)
    except Exception as e:
        raw["pa"] = {"error": str(e)}

    # Extract numbers safely
    pm25 = pm10 = no2 = o3 = so2 = co = aqi = None

    # from OWM
    try:
        comp = (raw.get("owm") or {}).get("list", [{}])[0].get("components", {})
        pm25 = pm25 or safe_float(comp.get("pm2_5"))
        pm10 = pm10 or safe_float(comp.get("pm10"))
        no2  = no2  or safe_float(comp.get("no2"))
        o3   = o3   or safe_float(comp.get("o3"))
        so2  = so2  or safe_float(comp.get("so2"))
        co   = co   or safe_float(comp.get("co"))
        aqi_owm  = safe_float((raw.get("owm") or {}).get("list", [{}])[0].get("main", {}).get("aqi"))
        # OWM AQI is 1–5 scale; convert roughly to US AQI midpoints
        if aqi_owm is not None and 1 <= aqi_owm <= 5:
            aqi = [25, 75, 125, 175, 275][int(aqi_owm) - 1]
    except Exception:
        pass

    # WAQI direct AQI
    try:
        aqi = aqi or safe_float((raw.get("waqi") or {}).get("data", {}).get("aqi"))
        if pm25 is None:
            pm25 = safe_float(((raw.get("waqi") or {}).get("data", {}).get("iaqi", {}) or {}).get("pm25", {}).get("v"))
    except Exception:
        pass

    # Weather from VC
    temp_c = humidity = wind_kph = None
    try:
        cur = (raw.get("vc") or {}).get("currentConditions", {})
        temp_c   = safe_float(cur.get("temp"))
        humidity = safe_float(cur.get("humidity"))
        wind_kph = safe_float(cur.get("windspeed"))
    except Exception:
        pass

    # AQI fallback from PM2.5
    if aqi is None:
        aqi = aqi_from_pm25(pm25)

    lp = LivePoint(
        city=city, lat=lat, lon=lon,
        aqi=aqi, aqi_cat=aqi_category(aqi),
        pm2_5=pm25, pm10=pm10, no2=no2, o3=o3, so2=so2, co=co,
        temp_c=temp_c, humidity=humidity, wind_kph=wind_kph,
        fetched_at=datetime.now(timezone.utc),
    )

    # Build a single-row DataFrame for display/CSV
    row = {
        "city": city, "lat": lat, "lon": lon,
        "aqi": safe_float(aqi), "aqi_category": lp.aqi_cat,
        "pm2_5": safe_float(pm25), "pm10": safe_float(pm10),
        "no2": safe_float(no2), "o3": safe_float(o3),
        "so2": safe_float(so2), "co": safe_float(co),
        "temp_c": safe_float(temp_c), "humidity": safe_float(humidity),
        "wind_kph": safe_float(wind_kph),
        "fetched_at": lp.fetched_at.isoformat(),
    }
    df = pd.DataFrame([row])

    # Save artifacts (non-blocking failures are okay)
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"{slug_of(city)}_live_{ts}"
        df.to_csv(DATA_DIR / f"{stem}.csv", index=False)
        with open(DATA_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return lp, df, raw


# --------------------------- UI -------------------------------

st.title("🌫️ Urban Air Quality Intelligence – Live Dashboard")
st.caption("Live readings from OpenWeatherMap, Visual Crossing, WAQI, and PurpleAir. "
           f"Last refresh: {now_utc_iso()} UTC")

with st.sidebar:
    st.header("Controls")
    city = st.selectbox("City", list(CITIES.keys()), index=1)
    lat, lon = CITIES[city]

    st.subheader("API Keys")
    st.write(f"OWM: {'✅' if OWM_KEY else '❌'}   VC: {'✅' if VC_KEY else '❌'}")
    st.write(f"WAQI: {'✅' if WAQI_KEY else '❌'}   PurpleAir: {'✅' if PA_KEY else '❌'}")
    st.caption("Set env vars before running:\n"
               "export OPENWEATHERMAP_API_KEY=...\n"
               "export VISUAL_CROSSING_API_KEY=...\n"
               "export WAQI_TOKEN=...\n"
               "export PURPLEAIR_API_KEY=...")

    fetch_btn = st.button("🔄 Fetch LIVE now", use_container_width=True)

# Always fetch once on load; button lets user refresh again
if "last_city" not in st.session_state or st.session_state.get("last_city") != city or fetch_btn:
    st.session_state["lp"], st.session_state["df"], st.session_state["raw"] = compose_live(city, lat, lon)
    st.session_state["last_city"] = city

lp: LivePoint = st.session_state["lp"]
df: pd.DataFrame = st.session_state["df"]
raw: Dict[str, Any] = st.session_state["raw"]

# Overview cards
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="AQI", value=int(lp.aqi) if lp.aqi is not None else "—", delta=lp.aqi_cat)
c2.metric(label="PM2.5 (µg/m³)", value=lp.pm2_5 if lp.pm2_5 is not None else "—")
c3.metric(label="Temp (°C)", value=lp.temp_c if lp.temp_c is not None else "—")
c4.metric(label="Humidity (%)", value=lp.humidity if lp.humidity is not None else "—")

# Map (city + PurpleAir sensors if available)
st.subheader(f"📍 Map – {city}")
layers: List[pdk.Layer] = []

# City marker
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"lat": lat, "lon": lon, "size": 10}]),
        get_position="[lon, lat]",
        get_radius=200,
        pickable=True,
    )
)

# PurpleAir sensors (if present)
pa = raw.get("pa") or {}
pa_data = pa.get("data") or []
pa_fields = pa.get("fields") or []
if pa_data and pa_fields:
    try:
        tbl = pd.DataFrame(pa_data, columns=pa_fields)
        tbl = tbl.rename(columns={"pm2.5": "pm25"})
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=tbl.rename(columns={"latitude": "lat", "longitude": "lon"}),
                get_position="[lon, lat]",
                get_radius=120,
                pickable=True,
            )
        )
        st.caption(f"PurpleAir sensors near {city}: {len(tbl)}")
    except Exception:
        st.caption("PurpleAir sensors: (unable to parse)")

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=10),
        layers=layers,
        tooltip={"text": "{name}\nPM2.5: {pm25}"},
    ),
    use_container_width=True,
)

# Tables / EDA-lite
st.subheader("🧾 Current live snapshot")
st.dataframe(df, use_container_width=True)

# Pollutant bar (only the ones we have)
polls = {
    "PM2.5": lp.pm2_5, "PM10": lp.pm10, "NO₂": lp.no2, "O₃": lp.o3, "SO₂": lp.so2, "CO": lp.co,
}
poll_df = pd.DataFrame(
    [{"pollutant": k, "value": v} for k, v in polls.items() if v is not None]
)
if not poll_df.empty:
    fig = px.bar(poll_df, x="pollutant", y="value", title=f"{city} – Live pollutants")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No pollutant values returned yet for this location.", icon="ℹ️")

# Raw responses (optional expander)
with st.expander("Raw API responses (debug)"):
    st.write("OpenWeatherMap Air:", raw.get("owm"))
    st.write("Visual Crossing:", raw.get("vc"))
    st.write("WAQI:", raw.get("waqi"))
    st.write("PurpleAir:", raw.get("pa"))

st.caption(f"Last updated: {now_utc_iso()} UTC")
