from __future__ import annotations
import numpy as np
import pandas as pd

NUM_SAFE_DEFAULTS = {
    "pm2_5": np.nan, "pm10": np.nan, "no2": np.nan, "o3": np.nan, "so2": np.nan, "co": np.nan,
    "aqi": np.nan, "temp": np.nan, "humidity": np.nan, "wind_speed": np.nan,
    "population": 0.0, "pop_density_per_km2": 0.0,
    "pct_elderly": 0.0, "pct_children": 0.0, "respiratory_illness_rate_per_100k": 0.0,
    "latitude": np.nan, "longitude": np.nan,
}

CAT_SAFE_DEFAULTS = {
    "aqi_category": "Unknown",
    "_category": "unknown",
}

def coerce_none_like(df: pd.DataFrame) -> pd.DataFrame:
    """Turn string 'None', '', 'nan' to np.nan, and cast numerics."""
    if df is None or df.empty:
        return df
    out = df.replace({"None": np.nan, "": np.nan, "nan": np.nan})
    for col, val in NUM_SAFE_DEFAULTS.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col, val in CAT_SAFE_DEFAULTS.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    # ensure datetime if present
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out

def fill_missing_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NAs with safe values that won't break charts/tables."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col, default in NUM_SAFE_DEFAULTS.items():
        if col in out.columns:
            # do NOT invent pollutant values – leave as NaN for honesty;
            # just avoid breaking visuals by not introducing strings.
            out[col] = out[col]  # keep NaNs, charts handle them
    for col, default in CAT_SAFE_DEFAULTS.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)
    return out
