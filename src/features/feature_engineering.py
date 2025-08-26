from __future__ import annotations

import numpy as np
import pandas as pd

POLLUTANT_COLS = ["pm2_5", "pm10", "o3", "no2", "so2", "co"]
WEATHER_COLS = ["temp", "humidity", "wind_speed", "precip"]


def aqi_category(aqi: float) -> str | float:
    """Simple AQI category mapping (EPA-like)."""
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def compute_aqi_from_pm25(pm25: pd.Series) -> pd.Series:
    """Compute AQI from PM2.5 using EPA-style breakpoints."""
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    def calc(x):
        if pd.isna(x):
            return np.nan
        for Cl, Ch, Il, Ih in bps:
            if x <= Ch:
                return (Ih - Il) / (Ch - Cl) * (x - Cl) + Il
        return 500.0

    return pm25.apply(calc)


def standardise_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a tz-naive UTC index named 'datetime', sorted.
    """
    df = df.copy()
    if "datetime" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'datetime' column.")
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    # Make tz-naive (UTC) so joins/filters are comparable
    dt = dt.dt.tz_convert(None)
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    return df


def merge_and_feature_engineer(
    pollution_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    temporal_resolution: str = "1h",
) -> pd.DataFrame:
    """
    Merge pollutant + weather on datetime, resampled to temporal_resolution.
    Returns columns: datetime, pollutants, weather, (latitude,longitude if available),
    and computed AQI + category (if missing).
    """
    # --- standardise and coerce numerics ---
    p_df = standardise_datetime(pollution_df)
    w_df = standardise_datetime(weather_df)

    for c in POLLUTANT_COLS + ["latitude", "longitude"]:
        if c in p_df.columns:
            p_df[c] = pd.to_numeric(p_df[c], errors="coerce")

    for c in WEATHER_COLS + ["latitude", "longitude"]:
        if c in w_df.columns:
            w_df[c] = pd.to_numeric(w_df[c], errors="coerce")

    # --- choose columns and resample ---
    pollutant_columns = [c for c in POLLUTANT_COLS if c in p_df.columns]
    keep_loc_cols = [c for c in ["latitude", "longitude"] if c in p_df.columns]

    pollutant_resampled = p_df[
        [c for c in pollutant_columns + keep_loc_cols if c in p_df.columns]
    ].resample(temporal_resolution).mean()

    weather_cols = [c for c in WEATHER_COLS if c in w_df.columns]
    # coerce to numeric already done; just resample
    weather_resampled = w_df[weather_cols].resample(temporal_resolution).mean()

    # --- join without column overlap issues (we excluded weather lat/lon) ---
    merged = pollutant_resampled.join(weather_resampled, how="outer")

    # if we didn't have lat/lon in pollutants, try to derive from either frame
    if not keep_loc_cols:
        for src in (p_df, w_df):
            for c in ["latitude", "longitude"]:
                if c in src.columns:
                    series = src[c].resample(temporal_resolution).mean()
                    if c not in merged.columns:
                        merged[c] = series
                    else:
                        merged[c] = merged[c].fillna(series)
        for c in ["latitude", "longitude"]:
            if c in merged.columns:
                merged[c] = merged[c].interpolate(limit_direction="both")

    # Fill weather gaps for charts
    for c in weather_cols:
        merged[c] = merged[c].interpolate(limit_direction="both").ffill().bfill()

    # --- AQI ---
    if "aqi" not in merged.columns:
        if "pm2_5" in merged.columns:
            merged["aqi"] = compute_aqi_from_pm25(merged["pm2_5"])
        elif "pm10" in merged.columns:
            # rough mapping if pm2_5 missing
            merged["aqi"] = compute_aqi_from_pm25(merged["pm10"] * 0.5)
        else:
            merged["aqi"] = np.nan

    merged["aqi_category"] = merged["aqi"].apply(aqi_category)

    merged = merged.reset_index()

    # order columns nicely if present
    preferred = (
        ["datetime"]
        + [c for c in POLLUTANT_COLS if c in merged.columns]
        + ["latitude", "longitude"]
        + [c for c in WEATHER_COLS if c in merged.columns]
        + ["aqi", "aqi_category"]
    )
    cols = [c for c in preferred if c in merged.columns] + [
        c for c in merged.columns if c not in preferred
    ]
    return merged[cols]
