"""Feature engineering utilities for air quality datasets.

This module defines functions to merge raw sensor and weather data into
machine‑learning‑ready tables.  It includes logic for computing derived
features (e.g. hourly averages, daily means) and generating target labels
for classification and regression tasks.

The functions in this file are designed to operate on pandas DataFrames
produced by the data collection scripts.  All timestamps are assumed to
be in UTC and formatted as ISO strings.  You may need to adjust the
conversion if your raw data contains timezone information.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def standardise_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    """Ensure a DataFrame has a proper datetime index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with a datetime column.
    column : str, default "datetime"
        Name of the column containing ISO timestamps.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by pandas.Timestamp objects.
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    df = df.set_index(column)
    df = df.sort_index()
    return df


def compute_aqi_pm25(value: float) -> float:
    """Convert PM2.5 concentration (µg/m³) to AQI based on EPA breakpoints.

    The US EPA defines breakpoints for calculating the Air Quality Index
    from PM2.5 concentrations.  This function implements a linear
    interpolation between breakpoints.  See
    https://forum.airnowtech.org/t/the-aqi-equation/169 for details.

    Parameters
    ----------
    value : float
        PM2.5 concentration in micrograms per cubic metre.

    Returns
    -------
    float
        The corresponding AQI value.
    """
    # Define breakpoints for PM2.5 (µg/m³) and corresponding AQI values.
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.0, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= value <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
    return np.nan


def aqi_category(aqi: float) -> str:
    """Categorise an AQI value into descriptive buckets."""
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def merge_and_feature_engineer(
    pollution_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    temporal_resolution: str = "H",
) -> pd.DataFrame:
    """Merge pollutant measurements with weather data and compute features.

    The function aligns both DataFrames on a common time index (rounded
    according to `temporal_resolution`), aggregates measurements within
    each time bin and computes derived features such as AQI and its
    categorical class.

    Parameters
    ----------
    pollution_df : pandas.DataFrame
        DataFrame with columns including 'datetime', pollutant names and optionally 'aqi'.
    weather_df : pandas.DataFrame
        DataFrame with columns including 'datetime', 'temp', 'humidity', etc.
    temporal_resolution : str, default 'H'
        Pandas offset alias for resampling (e.g. 'H' for hourly, 'D' for daily).

    Returns
    -------
    pandas.DataFrame
        Merged and feature‑engineered DataFrame ready for modelling.
    """
    # Standardise datetime indices
    p_df = standardise_datetime(pollution_df)
    w_df = standardise_datetime(weather_df)

    # Resample pollutant data to the target resolution (e.g. hourly)
    pollutant_columns = [col for col in p_df.columns if col not in {"location", "city", "unit", "parameter"}]
    # For OpenAQ data, there may be multiple parameters; pivot to wide format
    if "parameter" in p_df.columns and "value" in p_df.columns:
        pivot_df = p_df.pivot_table(
            values="value",
            index=p_df.index,
            columns="parameter",
            aggfunc="mean",
        )
        pollutant_resampled = pivot_df.resample(temporal_resolution).mean()
    else:
        pollutant_resampled = p_df[pollutant_columns].resample(temporal_resolution).mean()

    # Resample weather data
    weather_resampled = w_df.resample(temporal_resolution).mean()

    # Merge on datetime
    merged = pollutant_resampled.join(weather_resampled, how="outer")
    merged = merged.reset_index().rename(columns={"index": "datetime"})

    # Compute AQI from PM2.5 if available
    if "pm25" in merged.columns:
        merged["aqi"] = merged["pm25"].apply(compute_aqi_pm25)
        merged["aqi_category"] = merged["aqi"].apply(aqi_category)
    elif "pm2_5" in merged.columns:
        merged["aqi"] = merged["pm2_5"].apply(compute_aqi_pm25)
        merged["aqi_category"] = merged["aqi"].apply(aqi_category)
    else:
        merged["aqi"] = np.nan
        merged["aqi_category"] = "Unknown"

    return merged
