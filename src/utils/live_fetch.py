from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pandas as pd
import requests

from .config import get_config

OWM_AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


@dataclass
class LivePoint:
    city: str
    lat: float
    lon: float
    fetched_at: datetime
    pm2_5: float
    pm10: float
    no2: float
    o3: float
    so2: float
    co: float
    aqi: int
    temp: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    precip: Optional[float] = None


def _owm_get(url: str, lat: float, lon: float) -> Dict[str, Any]:
    cfg = get_config()
    key = getattr(cfg, "OPENWEATHERMAP_API_KEY", None)
    if not key:
        raise RuntimeError("OPENWEATHERMAP_API_KEY is missing.")
    resp = requests.get(url, params={"lat": lat, "lon": lon, "appid": key},
                        timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_live_point(city: str, lat: float, lon: float) -> LivePoint:
    air = _owm_get(OWM_AIR_URL, lat, lon)
    comp = air["list"][0]["components"]
    aqi = int(air["list"][0]["main"]["aqi"])

    wx = _owm_get(OWM_WEATHER_URL, lat, lon)
    main = wx.get("main", {})
    wind = wx.get("wind", {})
    rain = wx.get("rain", {})

    fetched = datetime.now(timezone.utc)
    return LivePoint(
        city=city,
        lat=lat,
        lon=lon,
        fetched_at=fetched,
        pm2_5=float(comp.get("pm2_5", float("nan"))),
        pm10=float(comp.get("pm10", float("nan"))),
        no2=float(comp.get("no2", float("nan"))),
        o3=float(comp.get("o3", float("nan"))),
        so2=float(comp.get("so2", float("nan"))),
        co=float(comp.get("co", float("nan"))),
        aqi=aqi,
        temp=main.get("temp"),
        humidity=main.get("humidity"),
        wind_speed=wind.get("speed"),
        precip=rain.get("1h", 0.0),
    )


def livepoint_to_df(p: LivePoint) -> pd.DataFrame:
    row = {
        "datetime": p.fetched_at.isoformat(),
        "city": p.city,
        "lat": p.lat,
        "lon": p.lon,
        "pm2_5": p.pm2_5,
        "pm10": p.pm10,
        "no2": p.no2,
        "o3": p.o3,
        "so2": p.so2,
        "co": p.co,
        "aqi": p.aqi,
        "temp": p.temp,
        "humidity": p.humidity,
        "wind_speed": p.wind_speed,
        "precip": p.precip,
    }
    return pd.DataFrame([row])
