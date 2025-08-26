from __future__ import annotations
import os
import math
from typing import Optional, Dict, Any, List
import requests
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load .env from project root
load_dotenv(find_dotenv(usecwd=True), override=True)


def _req_json(url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None, timeout: int = 60) -> Dict[str, Any]:
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------------- OpenWeatherMap ----------------


def fetch_openweathermap_air_pollution(lat: float, lon: float, start: str, end: str, api_key: Optional[str] = None, **kwargs) -> pd.DataFrame:
    key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
    if not key:
        raise RuntimeError("OPENWEATHERMAP_API_KEY is not set. Please add it to .env")
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat,
        "lon": lon,
        "start": int(pd.Timestamp(start).timestamp()),
        "end": int(pd.Timestamp(end).timestamp()),
        "appid": key,
    }
    js = _req_json(url, params=params)
    rows: List[Dict[str, Any]] = []
    for x in js.get("list", []):
        dt = pd.to_datetime(x.get("dt"), unit="s", utc=True, errors="coerce")
        c = x.get("components", {}) or {}
        rows.append(
            {
                "datetime": dt,
                "pm2_5": c.get("pm2_5"),
                "pm10": c.get("pm10"),
                "o3": c.get("o3"),
                "no2": c.get("no2"),
                "so2": c.get("so2"),
                "co": c.get("co"),
                "latitude": lat,
                "longitude": lon,
            }
        )
    return pd.DataFrame(rows)

# ---------------- OpenAQ v3 ----------------


def fetch_openaq_v3(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    radius_m: int = 10000,
    token: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    OpenAQ v3 measurements with pagination.
    Tolerant to extra kwargs (e.g., city, hourly) and supports env fallbacks:
      OPENAQ_LAT, OPENAQ_LON, OPENAQ_START, OPENAQ_END, OPENAQ_RADIUS_M
    """
    # strip unknown kwargs that some callers pass
    kwargs.pop("city", None)
    kwargs.pop("hourly", None)

    # env fallbacks
    r_env = os.getenv("OPENAQ_RADIUS_M")
    if r_env:
        try:
            radius_m = int(r_env)
        except Exception:
            pass
    if lat is None:
        v = os.getenv("OPENAQ_LAT")
        if v:
            try:
                lat = float(v)
            except Exception:
                pass
    if lon is None:
        v = os.getenv("OPENAQ_LON")
        if v:
            try:
                lon = float(v)
            except Exception:
                pass
    if start is None:
        start = os.getenv("OPENAQ_START")
    if end is None:
        end = os.getenv("OPENAQ_END")

    # If still missing required params, return empty DataFrame rather than erroring
    if None in (lat, lon, start, end):
        return pd.DataFrame()

    base = "https://api.openaq.org/v3/measurements"
    headers: Dict[str, str] = {}
    tok = token or os.getenv("OPENAQ_API_KEY")
    if tok:
        headers["X-API-Key"] = tok

    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_m,
        "date_from": pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to": pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 1000,
        "page": 1,
        "sort": "asc",
        "order_by": "datetime",
    }

    rows: List[Dict[str, Any]] = []
    while True:
        js = _req_json(base, params=params, headers=headers)
        results = js.get("results", []) or []
        for r in results:
            dt = pd.to_datetime(((r.get("date") or {}).get("utc")), errors="coerce", utc=True)
            coords = r.get("coordinates") or {}
            rows.append(
                {
                    "datetime": dt,
                    "parameter": r.get("parameter"),
                    "value": r.get("value"),
                    "unit": r.get("unit"),
                    "latitude": coords.get("latitude"),
                    "longitude": coords.get("longitude"),
                    "source": "OpenAQ",
                }
            )

        meta = js.get("meta") or {}
        found = len(results)
        page = params["page"]
        total_found = meta.get("found", 0)
        if not found or (page * params["limit"] >= total_found):
            break
        params["page"] += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Pivot pollutants to columns; align names to pm2_5 etc.
    piv = (
        df.pivot_table(index="datetime", columns="parameter", values="value", aggfunc="mean")
        .rename(columns={"pm25": "pm2_5"})
        .reset_index()
    )
    # Add one representative lat/lon if available
    if {"latitude", "longitude"}.issubset(df.columns):
        lat0 = df["latitude"].dropna().iloc[0] if df["latitude"].notna().any() else None
        lon0 = df["longitude"].dropna().iloc[0] if df["longitude"].notna().any() else None
        piv["latitude"] = lat0
        piv["longitude"] = lon0
    return piv


def fetch_openaq_v3_measurements(*args, **kwargs):
    """Backward-compatible alias."""
    return fetch_openaq_v3(*args, **kwargs)

# ---------------- PurpleAir ----------------


def fetch_purpleair_sensors(lat: float, lon: float, *args, radius_m: int = 10000, token: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Query PurpleAir v1 sensors in a bounding box around (lat, lon).
    Returns current outdoor sensor readings; tolerant to extra args.
    """
    key = token or os.getenv("PURPLEAIR_API_KEY")
    if not key:
        raise RuntimeError("PURPLEAIR_API_KEY is not set. Please add it to .env")

    # approximate degree deltas for radius
    dlat = radius_m / 111000.0
    dlon = radius_m / (111000.0 * max(0.1, math.cos(math.radians(lat))))

    params = {
        "fields": "sensor_index,name,latitude,longitude,pm2.5_atm,pm2.5,humidity,temperature,pressure,last_seen",
        "nwlng": lon - dlon,
        "nwlat": lat + dlat,
        "selng": lon + dlon,
        "selat": lat - dlat,
        "location_type": 0,  # outdoor
    }
    headers = {"X-API-Key": key}

    js = _req_json("https://api.purpleair.com/v1/sensors", params=params, headers=headers)
    fields = js.get("fields", []) or []
    data = js.get("data", []) or []
    rows: List[Dict[str, Any]] = []
    for row in data:
        rec = {fields[i]: row[i] for i in range(min(len(fields), len(row)))}
        dt = pd.to_datetime(rec.get("last_seen"), unit="s", utc=True, errors="coerce")
        pm = rec.get("pm2.5_atm")
        if pm is None:
            pm = rec.get("pm2.5")
        rows.append(
            {
                "datetime": dt,
                "latitude": rec.get("latitude"),
                "longitude": rec.get("longitude"),
                "pm2_5": pm,
                "humidity": rec.get("humidity"),
                "temp": rec.get("temperature"),
                "pressure": rec.get("pressure"),
                "sensor_index": rec.get("sensor_index"),
                "name": rec.get("name"),
            }
        )
    return pd.DataFrame(rows)

# ---------------- WAQI ----------------


def fetch_waqi_city(city: str, token: Optional[str] = None, **kwargs) -> pd.DataFrame:
    key = token or os.getenv("WAQI_API_KEY")
    if not key:
        raise RuntimeError("WAQI_API_KEY is not set. Please add it to .env")
    url = f"https://api.waqi.info/feed/{city}/"
    js = _req_json(url, params={"token": key})
    # WAQI may return {"status":"error","data":"Unknown station"} (string)
    if not isinstance(js, dict) or js.get("status") != "ok" or not isinstance(js.get("data"), dict):
        return pd.DataFrame()
    data = js.get("data", {}) or {}
    iaqi = data.get("iaqi", {}) or {}
    dt = pd.to_datetime(((data.get("time") or {}).get("s")), errors="coerce")
    row: Dict[str, Any] = {"datetime": dt, "aqi": data.get("aqi"), "dominientpol": data.get("dominentpol")}
    for k, v in iaqi.items():
        try:
            row[k] = v.get("v")
        except Exception:
            pass
    return pd.DataFrame([row])


def fetch_waqi_current(city: str, token: Optional[str] = None, **kwargs):
    """Backward-compatible alias."""
    return fetch_waqi_city(city, token=token, **kwargs)

# ---------------- Visual Crossing ----------------


def fetch_visualcrossing_weather(lat: float, lon: float, start: str, end: str, key: Optional[str] = None, **kwargs) -> pd.DataFrame:
    api = key or os.getenv("VISUAL_CROSSING_API_KEY")
    if not api:
        raise RuntimeError("VISUAL_CROSSING_API_KEY is not set. Please add it to .env")
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    loc = f"{lat},{lon}"
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": api,
        "contentType": "json",
        "startDateTime": pd.to_datetime(start).strftime("%Y-%m-%d"),
        "endDateTime": pd.to_datetime(end).strftime("%Y-%m-%d"),
    }
    js = _req_json(url + loc, params=params)
    rows: List[Dict[str, Any]] = []
    for day in js.get("days", []) or []:
        for h in day.get("hours", []) or []:
            dt = pd.to_datetime(h.get("datetimeEpoch"), unit="s", utc=True, errors="coerce")
            rows.append(
                {
                    "datetime": dt,
                    "temp": h.get("temp"),
                    "humidity": h.get("humidity"),
                    "wind_speed": h.get("windspeed"),
                    "precip": h.get("precip"),
                    "latitude": lat,
                    "longitude": lon,
                }
            )
    return pd.DataFrame(rows)
