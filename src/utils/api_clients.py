"""HTTP clients for external data sources.

This module encapsulates the logic required to query various third‑party APIs
used in the Urban Air Quality Intelligence System.  Each function returns
either JSON or pandas DataFrame objects representing the data returned by the
service.  API keys are pulled from the environment via the `get_config`
function.

Note
----
These functions make real HTTP requests.  Make sure to set your API
keys in a `.env` file before calling them.  For long date ranges or
high‑resolution data (e.g. hourly), be mindful of API rate limits.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from .config import get_config


def _unix_timestamp(date_str: str) -> int:
    """Convert an ISO date string (YYYY‑MM‑DD) to a Unix timestamp (seconds)."""
    return int(dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def fetch_openweathermap_air_pollution(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch historical air pollution data from OpenWeatherMap.

    OpenWeatherMap provides hourly air quality data for up to 5 days
    historically.  The start and end dates must be expressed as Unix
    timestamps.  The returned DataFrame contains columns for each pollutant
    (pm2_5, pm10, o3, no2, so2, co) and a `datetime` column.

    Parameters
    ----------
    latitude, longitude : float
        Geographic coordinates of the location.
    start_date, end_date : str
        ISO date strings (YYYY‑MM‑DD) defining the inclusive date range.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by datetime with pollutant concentrations.
    """
    cfg = get_config()
    api_key = cfg.OPENWEATHERMAP_API_KEY
    if not api_key:
        raise RuntimeError(
            "OPENWEATHERMAP_API_KEY is not set. Please add it to your .env file."
        )
    start_ts = _unix_timestamp(start_date)
    end_ts = _unix_timestamp(end_date)
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": latitude,
        "lon": longitude,
        "start": start_ts,
        "end": end_ts,
        "appid": api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("list", [])
    records = []
    for entry in data:
        dt_ts = entry.get("dt")
        components = entry.get("components", {})
        record = {
            "datetime": dt.datetime.utcfromtimestamp(dt_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "o3": components.get("o3"),
            "no2": components.get("no2"),
            "so2": components.get("so2"),
            "co": components.get("co"),
        }
        records.append(record)
    df = pd.DataFrame(records)
    return df


def fetch_openaq_measurements(
    city: Optional[str] = None,
    coordinates: Optional[Tuple[float, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    limit: int = 10000,
) -> pd.DataFrame:
    """Retrieve air quality measurements from OpenAQ.

    OpenAQ provides data for numerous pollutants across the globe.  You can
    filter by city name or by coordinates.  If both are provided, the
    coordinates take precedence.  Date filters must be ISO strings.

    Parameters
    ----------
    city : str, optional
        Name of the city (e.g. "Vellore").  Ignored if coordinates are given.
    coordinates : tuple of float, optional
        A `(latitude, longitude)` pair.  If provided, `radius` defaults to
        10000 metres.
    start_date, end_date : str, optional
        ISO date strings (YYYY‑MM‑DD) specifying the inclusive range.
    parameters : list of str, optional
        List of pollutant codes to retrieve.  Defaults to PM2.5, PM10,
        NO2, O3, CO, SO2.
    limit : int
        Maximum number of records to return (default 10 000).

    Returns
    -------
    pandas.DataFrame
        Measurements with columns: datetime, parameter, value, unit, location, city.
    """
    cfg = get_config()
    token = cfg.OPENAQ_API_KEY
    headers = {"x-api-key": token} if token else {}
    base_url = "https://api.openaq.org/v2/measurements"
    # Default parameters
    if parameters is None:
        parameters = ["pm25", "pm10", "no2", "o3", "co", "so2"]
    params: Dict[str, str] = {
        "limit": limit,
        "page": 1,
        "offset": 0,
        "sort": "asc",
        "parameter": parameters,
        "order_by": "datetime",
    }
    if start_date:
        params["date_from"] = start_date
    if end_date:
        params["date_to"] = end_date
    if coordinates is not None:
        lat, lon = coordinates
        params["coordinates"] = f"{lat},{lon}"
        params["radius"] = 10000  # 10 km radius
    elif city:
        params["city"] = city
    results = []
    resp = requests.get(base_url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for item in data.get("results", []):
        results.append(
            {
                "datetime": item.get("date", {}).get("utc"),
                "parameter": item.get("parameter"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "location": item.get("location"),
                "city": item.get("city"),
                "latitude": item.get("coordinates", {}).get("latitude"),
                "longitude": item.get("coordinates", {}).get("longitude"),
            }
        )
    df = pd.DataFrame(results)
    return df


def fetch_purpleair_sensors(
    nwlat: float,
    nwlng: float,
    selat: float,
    selng: float,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch PurpleAir sensor data within a bounding box.

    PurpleAir provides near‑real‑time readings from low‑cost sensors.  You
    must supply your API key in `.env`.  The bounding box is defined by
    northwest and southeast coordinates (lat, lon).  By default the
    function requests common fields such as PM2.5 and temperature.

    Parameters
    ----------
    nwlat, nwlng : float
        Latitude and longitude of the north‑west corner of the bounding box.
    selat, selng : float
        Latitude and longitude of the south‑east corner of the bounding box.
    fields : list of str, optional
        PurpleAir sensor fields to request.  See the API docs for full list.

    Returns
    -------
    pandas.DataFrame
        Sensor measurements with sensor metadata.
    """
    cfg = get_config()
    api_key = cfg.PURPLEAIR_API_KEY
    if not api_key:
        raise RuntimeError(
            "PURPLEAIR_API_KEY is not set. Please add it to your .env file."
        )
    base_url = "https://api.purpleair.com/v1/sensors"
    if fields is None:
        # See https://api.purpleair.com for available fields.  The following
        # fields are commonly used: pm2.5 mass concentration (pm2.5_atm),
        # temperature, humidity, pressure.
        fields = ["pm2.5_atm", "temperature", "humidity", "pressure"]
    params = {
        "fields": ",".join(fields),
        "nwlat": nwlat,
        "nwlng": nwlng,
        "selat": selat,
        "selng": selng,
    }
    headers = {"X-API-Key": api_key}
    resp = requests.get(base_url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    sensor_fields = data.get("fields", [])
    results = []
    for sensor in data.get("data", []):
        record = dict(zip(sensor_fields, sensor))
        results.append(record)
    df = pd.DataFrame(results)
    return df


def fetch_waqi_current(
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Fetch current Air Quality Index (AQI) and pollutant concentrations from WAQI.

    The WAQI API returns the most recent AQI value for the specified coordinates.
    This function extracts the general AQI, dominant pollutant and individual
    pollutant readings.

    Parameters
    ----------
    latitude, longitude : float
        Geographic coordinates.

    Returns
    -------
    pandas.DataFrame
        A single row DataFrame with columns: aqi, dominentpol, pm25, pm10, no2, o3, co, so2.
    """
    cfg = get_config()
    token = cfg.WAQI_API_KEY
    if not token:
        raise RuntimeError(
            "WAQI_API_KEY is not set. Please add it to your .env file."
        )
    url = f"https://api.waqi.info/feed/geo:{latitude};{longitude}/"
    params = {"token": token}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"WAQI API error: {data.get('data')}")
    d = data.get("data", {})
    iaqi = d.get("iaqi", {})
    record = {
        "aqi": d.get("aqi"),
        "dominentpol": d.get("dominentpol"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
    }
    df = pd.DataFrame([record])
    return df


def fetch_visualcrossing_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly: bool = True,
) -> pd.DataFrame:
    """Fetch weather data from Visual Crossing.

    Visual Crossing provides historical and forecast weather data.  This
    function returns temperature, humidity, wind speed and precipitation for
    each hour (or day if `hourly=False`).  Date strings should be in
    YYYY‑MM‑DD format.

    Parameters
    ----------
    latitude, longitude : float
        Coordinates of the location.
    start_date, end_date : str
        ISO date strings defining the inclusive range.
    hourly : bool, default True
        Whether to return hourly records; if False daily summaries are
        returned.

    Returns
    -------
    pandas.DataFrame
        Weather observations with datetime, temperature, humidity, wind and precipitation.
    """
    cfg = get_config()
    api_key = cfg.VISUAL_CROSSING_API_KEY
    if not api_key:
        raise RuntimeError(
            "VISUAL_CROSSING_API_KEY is not set. Please add it to your .env file."
        )
    unit_group = "metric"
    include = "hours" if hourly else "days"
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
        f"timeline/{latitude},{longitude}/{start_date}/{end_date}"
    )
    params = {
        "unitGroup": unit_group,
        "include": include,
        "key": api_key,
        "contentType": "json",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records = []
    if hourly:
        for day in data.get("days", []):
            date_str = day.get("datetime")
            for hour in day.get("hours", []):
                dt_str = f"{date_str} {hour.get('datetime')}"
                records.append(
                    {
                        "datetime": dt_str,
                        "temp": hour.get("temp"),
                        "humidity": hour.get("humidity"),
                        "wind_speed": hour.get("windspeed"),
                        "precip": hour.get("precip"),
                    }
                )
    else:
        for day in data.get("days", []):
            records.append(
                {
                    "datetime": day.get("datetime"),
                    "temp": day.get("temp"),
                    "humidity": day.get("humidity"),
                    "wind_speed": day.get("windspeed"),
                    "precip": day.get("precip"),
                }
            )
    df = pd.DataFrame(records)
    return df
