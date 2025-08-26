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


def merge_and_feature_engineer(pollution_df, weather_df, temporal_resolution='h'):
    """Merge pollution & weather, compute simple AQI + category.

    - Resamples to the given temporal_resolution (default "H")
    - Joins on datetime
    - Sets aqi (proxy) from pm2_5 if not present
    - Sets aqi_category from pm2_5 with bins so 20 → "Good"
    """
    import numpy as np
    import pandas as pd

    # Normalize frequency to avoid pandas 'H' deprecation warning
    temporal_resolution = str(temporal_resolution).lower()

    p = pollution_df.copy()
    w = weather_df.copy()

    # ensure datetime
    for df in (p, w):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            raise ValueError("Expected a 'datetime' column")

    p = p.set_index("datetime").sort_index()
    w = w.set_index("datetime").sort_index()

    pol_cols = [c for c in ("pm2_5", "pm10", "no2", "o3", "so2", "co") if c in p.columns]
    w_cols = [c for c in ("temp", "humidity", "wind_speed", "precip") if c in w.columns]

    if temporal_resolution:
        p = p[pol_cols].resample(temporal_resolution).mean()
        w = w[w_cols].resample(temporal_resolution).mean()

    df = p.join(w, how="outer").reset_index()

    # AQI proxy and category from pm2_5 (keeps whatever 'aqi' you had if present)
    if "pm2_5" in df.columns:
        if "aqi" not in df.columns:
            df["aqi"] = df["pm2_5"]

        bins = [-np.inf, 25, 50, 100, 150, 200, np.inf]
        labels = [
            "Good",
            "Moderate",
            "Unhealthy for Sensitive Groups",
            "Unhealthy",
            "Very Unhealthy",
            "Hazardous",
        ]
        df["aqi_category"] = pd.cut(df["pm2_5"], bins=bins, labels=labels, right=True).astype(str)

    return df
