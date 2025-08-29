"""Command‑line tool to collect air quality and weather data.

This script orchestrates calls to the various API clients and writes the
returned data into CSV files under the `data/raw/` directory.  It can be
executed standalone from the project root and accepts a handful of
command‑line arguments to specify the target location and date range.

Example
-------
    python src/data/collect_air_quality.py --city "Vellore" --latitude 12.9165 --longitude 79.1325 \
        --start-date "2024-08-17" --end-date "2024-08-24"

If you omit the city name the script will use coordinates only.  Both
`start-date` and `end-date` are inclusive and must be provided.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import pandas as pd

from src.utils.api_clients import (
    fetch_openaq_v3_measurements,
    fetch_openweathermap_air_pollution,
    fetch_purpleair_sensors,
    fetch_visualcrossing_weather,
    fetch_waqi_current,
)


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect air quality and weather data for a given location and date range."
    )
    parser.add_argument("--city", type=str, help="Name of the city (optional)")
    parser.add_argument(
        "--latitude", type=float, required=True, help="Latitude of the location"
    )
    parser.add_argument(
        "--longitude", type=float, required=True, help="Longitude of the location"
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    return parser.parse_args(args)


def ensure_directory(path: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save a DataFrame to CSV in the specified directory."""
    ensure_directory(output_dir)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} records)")


def compute_bounding_box(lat: float, lon: float, delta: float = 0.1) -> Tuple[float, float, float, float]:
    """Compute a simple bounding box around a point for PurpleAir queries.

    The bounding box is defined as (nwlat, nwlng, selat, selng).  The
    `delta` parameter controls the size of the box in degrees.  A value of
    0.1 (~11 km) is a reasonable default for city‑scale queries.
    """
    nwlat = lat + delta
    nwlng = lon - delta
    selat = lat - delta
    selng = lon + delta
    return nwlat, nwlng, selat, selng


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    city = args.city
    lat = args.latitude
    lon = args.longitude
    start_date = args.start_date
    end_date = args.end_date

    output_dir = os.path.join("data", "raw")

    # Fetch OpenWeatherMap pollution history
    try:
        owm_df = fetch_openweathermap_air_pollution(lat, lon, start_date, end_date)
        fname = f"{city or 'coord'}_openweathermap_{start_date}_{end_date}.csv"
        save_dataframe(owm_df, output_dir, fname)
    except Exception as e:
        print(f"[ERROR] Failed to fetch OpenWeatherMap air pollution data: {e}")

    # Fetch OpenAQ measurements
    try:
        openaq_df = fetch_openaq_v3_measurements(
            city=city, coordinates=(lat, lon), start_date=start_date, end_date=end_date
        )
        fname = f"{city or 'coord'}_openaq_{start_date}_{end_date}.csv"
        save_dataframe(openaq_df, output_dir, fname)
    except Exception as e:
        print(f"[ERROR] Failed to fetch OpenAQ data: {e}")

    # Fetch PurpleAir sensor readings within bounding box
    try:
        bbox = compute_bounding_box(lat, lon, delta=0.15)
        purpleair_df = fetch_purpleair_sensors(*bbox)
        fname = f"{city or 'coord'}_purpleair_{start_date}_{end_date}.csv"
        save_dataframe(purpleair_df, output_dir, fname)
    except Exception as e:
        print(f"[ERROR] Failed to fetch PurpleAir data: {e}")

    # Fetch WAQI current AQI (single record)
    try:
        waqi_df = fetch_waqi_current(lat, lon)
        fname = f"{city or 'coord'}_waqi_current.csv"
        save_dataframe(waqi_df, output_dir, fname)
    except Exception as e:
        print(f"[ERROR] Failed to fetch WAQI data: {e}")

    # Fetch Visual Crossing weather history
    try:
        wx_df = fetch_visualcrossing_weather(lat, lon, start_date, end_date, hourly=True)
        fname = f"{city or 'coord'}_visualcrossing_{start_date}_{end_date}.csv"
        save_dataframe(wx_df, output_dir, fname)
    except Exception as e:
        print(f"[ERROR] Failed to fetch Visual Crossing weather data: {e}")


if __name__ == "__main__":
    main()
