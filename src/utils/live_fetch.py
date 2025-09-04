from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def _utc_iso(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc).isoformat()


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _coalesce_series(df: pd.DataFrame, names: List[str]) -> pd.Series:
    """Combine columns by first non-null numeric among possible names."""
    s = None
    for name in names:
        if name in df.columns:
            cand = pd.to_numeric(df[name], errors="coerce")
            s = cand if s is None else s.combine_first(cand)
    if s is None:
        # Build an all-NA series of the correct length
        s = pd.Series([pd.NA] * len(df))
    return s


def _median_of(df: pd.DataFrame, names: List[str]) -> Optional[float]:
    s = _coalesce_series(df, names)
    val = pd.to_numeric(s, errors="coerce").median()
    try:
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _request_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20) -> Dict[str, Any]:
    r = requests.get(url, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -------------------------- Provider calls --------------------------

def call_openweathermap_air(lat: float, lon: float, api_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    raw = _request_json(url)
    row: Dict[str, Any] = {
        "provider": "openweathermap",
        "lat": lat,
        "lon": lon,
        "datetime": _utc_iso(),
    }
    try:
        item = (raw.get("list") or [{}])[0]
        comps = item.get("components") or {}
        row.update(
            {
                "pm2_5": _to_float(comps.get("pm2_5")),
                "pm10": _to_float(comps.get("pm10")),
                "no2": _to_float(comps.get("no2")),
                "o3": _to_float(comps.get("o3")),
                "so2": _to_float(comps.get("so2")),
                "co": _to_float(comps.get("co")),
                "aqi": _to_float((item.get("main") or {}).get("aqi")),  # 1..5
            }
        )
    except Exception:
        pass
    return row, raw


def call_visualcrossing_current(lat: float, lon: float, api_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}?unitGroup=metric&include=current&contentType=json&key={api_key}"
    )
    raw = _request_json(url)
    row: Dict[str, Any] = {
        "provider": "visualcrossing",
        "lat": lat,
        "lon": lon,
        "datetime": _utc_iso(),
    }
    try:
        cc = raw.get("currentConditions") or {}
        row.update(
            {
                "temp": _to_float(cc.get("temp")),
                "humidity": _to_float(cc.get("humidity")),
                "wind_speed": _to_float(cc.get("windspeed")),
                "precip": _to_float(cc.get("precip")),
            }
        )
    except Exception:
        pass
    return row, raw


def call_waqi_geo(lat: float, lon: float, token: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
    raw = _request_json(url)
    row: Dict[str, Any] = {
        "provider": "waqi",
        "lat": lat,
        "lon": lon,
        "datetime": _utc_iso(),
    }
    try:
        data = raw.get("data") or {}
        iaqi = data.get("iaqi") or {}
        # iaqi fields look like {"pm25":{"v": 100}}
        def v(key: str) -> Optional[float]:
            try:
                return _to_float((iaqi.get(key) or {}).get("v"))
            except Exception:
                return None

        row.update(
            {
                "aqi": _to_float(data.get("aqi")),
                "pm25": v("pm25"),
                "pm10": v("pm10"),
                "no2": v("no2"),
                "o3": v("o3"),
                "so2": v("so2"),
                "co": v("co"),
                "temp": v("t"),
                "humidity": v("h"),
                "wind_speed": v("w"),
            }
        )
    except Exception:
        pass
    return row, raw


def call_purpleair_bbox(lat: float, lon: float, api_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # A small bounding box around the point
    dlat = 0.4
    dlon = 0.4
    params = {
        "fields": "name,latitude,longitude,pm2.5,pm2.5_10minute,pm2.5_30minute,pm2.5_60minute,last_seen",
        "nwlng": lon - dlon,
        "nwlat": lat + dlat,
        "selng": lon + dlon,
        "selat": lat - dlat,
    }
    url = "https://api.purpleair.com/v1/sensors?" + "&".join(f"{k}={v}" for k, v in params.items())
    raw = _request_json(url, headers={"X-API-Key": api_key})
    row: Dict[str, Any] = {
        "provider": "purpleair",
        "lat": lat,
        "lon": lon,
        "datetime": _utc_iso(),
    }
    try:
        fields = raw.get("fields") or []
        data = raw.get("data") or []
        name_to_idx = {f: i for i, f in enumerate(fields)}
        # pick the 10-minute average if present, else 60-minute, else instant
        picks: List[Optional[float]] = []
        for rec in data:
            pm10m = rec[name_to_idx.get("pm2.5_10minute")] if "pm2.5_10minute" in name_to_idx else None
            pm60m = rec[name_to_idx.get("pm2.5_60minute")] if "pm2.5_60minute" in name_to_idx else None
            pminst = rec[name_to_idx.get("pm2.5")] if "pm2.5" in name_to_idx else None
            picks.append(_to_float(pm10m if pm10m is not None else (pm60m if pm60m is not None else pminst)))
        picks = [p for p in picks if p is not None]
        row["pm2.5"] = float(pd.Series(picks).median()) if picks else None
    except Exception:
        pass
    return row, raw


# -------------------------- Bundle + merge --------------------------

CANON_MAP = {
    "pm2_5": ["pm2_5", "pm25", "pm2.5"],
    "pm10": ["pm10"],
    "no2": ["no2", "NO2"],
    "o3": ["o3", "O3"],
    "so2": ["so2", "SO2"],
    "co": ["co", "CO"],
    "aqi": ["aqi", "AQI"],
    "temp": ["temp", "temperature"],
    "humidity": ["humidity", "rh"],
    "wind_speed": ["wind_speed", "windspeed", "wind"],
    "precip": ["precip", "precipitation"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # build canonical columns even if sources used different names
    for target, names in CANON_MAP.items():
        out[target] = _coalesce_series(out, names)
    return out


def fetch_live_bundle(city: str, lat: float, lon: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch live from OWM, VisualCrossing, WAQI, PurpleAir, normalize, and add a merged row."""
    city = city.strip()
    raw: Dict[str, Any] = {"city": city, "lat": lat, "lon": lon, "providers": {}}
    rows: List[Dict[str, Any]] = []

    # Read API keys from env (caller promised to set them)
    OWM = os.getenv("OPENWEATHERMAP_API_KEY")
    VC = os.getenv("VISUAL_CROSSING_API_KEY")
    WAQI = os.getenv("WAQI_TOKEN")
    PA = os.getenv("PURPLEAIR_API_KEY")

    # Each provider may fail independently; we keep going
    if OWM:
        try:
            row, rj = call_openweathermap_air(lat, lon, OWM)
            row.update({"city": city})
            rows.append(row)
            raw["providers"]["openweathermap"] = rj
        except Exception as e:
            raw["providers"]["openweathermap_error"] = str(e)

    if VC:
        try:
            row, rj = call_visualcrossing_current(lat, lon, VC)
            row.update({"city": city})
            rows.append(row)
            raw["providers"]["visualcrossing"] = rj
        except Exception as e:
            raw["providers"]["visualcrossing_error"] = str(e)

    if WAQI:
        try:
            row, rj = call_waqi_geo(lat, lon, WAQI)
            row.update({"city": city})
            rows.append(row)
            raw["providers"]["waqi"] = rj
        except Exception as e:
            raw["providers"]["waqi_error"] = str(e)

    if PA:
        try:
            row, rj = call_purpleair_bbox(lat, lon, PA)
            row.update({"city": city})
            rows.append(row)
            raw["providers"]["purpleair"] = rj
        except Exception as e:
            raw["providers"]["purpleair_error"] = str(e)

    if not rows:
        # Return empty normalized frame so callers never crash
        df = pd.DataFrame([{"provider": "none", "city": city, "lat": lat, "lon": lon, "datetime": _utc_iso()}])
        df = _normalize_columns(df)
        return df, raw

    df = pd.DataFrame(rows)
    df = _normalize_columns(df)

    # Build a merged (median) row across providers
    merged = {
        "provider": "merged",
        "city": city,
        "lat": lat,
        "lon": lon,
        "datetime": _utc_iso(),
    }
    for target, names in CANON_MAP.items():
        merged[target] = _median_of(df, names)

    # Stack merged row last
    out = pd.concat([df, pd.DataFrame([merged])], ignore_index=True)
    return out, raw



# --- Back-compat shims for older app/main.py imports --------------------------
from types import SimpleNamespace as _LF_NS

def _lf_maybe_num(x):
    try:
        import pandas as pd
        if x is None or (hasattr(pd, "isna") and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None

def _lf_aqi_from_pm25(pm):
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
            return round((Ih-Il)/(Ch-Cl)*(pm-Cl)+Il)
    return None

def livepoint_to_df(point):
    \"\"\"Compatibility: object/dict -> single-row DataFrame.\"\"\"
    import pandas as pd
    if hasattr(point, '__dict__'):
        d = dict(point.__dict__)
    elif isinstance(point, dict):
        d = dict(point)
    else:
        return pd.DataFrame()
    # make fetched_at serializable if present
    fa = d.get("fetched_at")
    if hasattr(fa, "isoformat"):
        d["fetched_at"] = fa.isoformat()
    return pd.DataFrame([d])

def fetch_live_point(city, lat, lon):
    \"\"\"
    Compatibility wrapper for older UI code.
    Returns (obj_with_attrs, df, raw_bundle).
    The first value exposes attributes like .aqi, .pm2_5, .fetched_at, etc.
    \"\"\"
    import pandas as pd
    from datetime import datetime, timezone

    try:
        # existing modern function in this module:
        df, raw = fetch_live_bundle(city, lat, lon)
    except Exception:
        df, raw = pd.DataFrame(), {}

    aqi = None
    pm25 = None
    pm10 = None
    no2 = None
    o3 = None
    so2 = None
    co = None

    if isinstance(df, pd.DataFrame) and not df.empty:
        row = df.iloc[0]
        pm25 = _lf_maybe_num(row.get("pm2_5"))
        pm10 = _lf_maybe_num(row.get("pm10"))
        no2  = _lf_maybe_num(row.get("no2"))
        o3   = _lf_maybe_num(row.get("o3"))
        so2  = _lf_maybe_num(row.get("so2"))
        co   = _lf_maybe_num(row.get("co"))
        # try direct AQI columns first, then fall back to PM2.5-derived AQI
        aqi = (_lf_maybe_num(row.get("aqi"))
               or _lf_maybe_num(row.get("waqi_aqi"))
               or _lf_aqi_from_pm25(pm25))

    obj = _LF_NS(
        city=str(city),
        lat=float(lat),
        lon=float(lon),
        aqi=aqi,
        pm2_5=pm25,
        pm10=pm10,
        no2=no2,
        o3=o3,
        so2=so2,
        co=co,
        fetched_at=datetime.now(timezone.utc),
        source="live_bundle"
    )
    return obj, df, raw
# ----------------------------------------------------------------------------- 
