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

def drop_empty_columns(df: pd.DataFrame, keep: list[str] | None = None) -> pd.DataFrame:
    # Drop columns that are entirely NaN or the literal strings 'None' or ''.
    if df is None or df.empty:
        return df
    tmp = df.replace({"None": np.nan, "": np.nan})
    mask = tmp.notna().any(axis=0)
    out = tmp.loc[:, mask]
    if keep:
        keep = [c for c in keep if c in tmp.columns and c not in out.columns]
        if keep:
            out = pd.concat([out, tmp[keep]], axis=1)
    return out

def has_enough_points(df: pd.DataFrame, cols: list[str], min_points: int = 10) -> bool:
    # True if df has at least `min_points` non-null rows across provided columns.
    if df is None or df.empty:
        return False
    subset = [c for c in cols if c in df.columns]
    if not subset:
        return False
    return df[subset].dropna().shape[0] >= min_points

def drop_mostly_empty_columns(df: pd.DataFrame, thresh: float = 0.95, keep: list[str] | None = None) -> pd.DataFrame:
    """Drop columns where fraction of NaN/empty >= thresh. Keep any in `keep`."""
    if df is None or df.empty:
        return df
    tmp = df.replace({"None": np.nan, "": np.nan})
    frac_nan = tmp.isna().mean()
    keep = keep or []
    cols = [c for c in tmp.columns if (frac_nan[c] < thresh) or (c in keep)]
    return tmp[cols]

def recent_nonnull_window(df: pd.DataFrame, ycol: str, xcol: str = "datetime", days: int = 14) -> pd.DataFrame:
    """Return the last `days` of rows where ycol has data; falls back to all."""
    if df is None or df.empty or ycol not in df or xcol not in df:
        return df
    dff = df.dropna(subset=[ycol]).copy()
    if dff.empty:
        return df
    cutoff = pd.to_datetime(dff[xcol]).max() - pd.Timedelta(days=days)
    return df[pd.to_datetime(df[xcol]) >= cutoff].copy()

def backfill_pm_from_forecast(proc_df: pd.DataFrame, fcst_df: pd.DataFrame, pm_col: str = "pm2_5") -> pd.DataFrame:
    """If `pm_col` has many NaNs, fill missing using forecast `yhat` on nearest hour."""
    if proc_df is None or proc_df.empty or fcst_df is None or fcst_df.empty:
        return proc_df
    if "datetime" not in proc_df or "ds" not in fcst_df or "yhat" not in fcst_df:
        return proc_df
    out = proc_df.copy()
    out["__dt_hour"] = pd.to_datetime(out["datetime"]).dt.floor("h")
    f = fcst_df.copy()
    f["__dt_hour"] = pd.to_datetime(f["ds"]).dt.floor("h")
    f = f[["__dt_hour", "yhat"]].drop_duplicates("__dt_hour", keep="last")
    out = out.merge(f, on="__dt_hour", how="left")
    if pm_col not in out:
        out[pm_col] = np.nan
    need_fill = out[pm_col].isna()
    out.loc[need_fill, pm_col] = out.loc[need_fill, "yhat"]
    out = out.drop(columns=["__dt_hour", "yhat"])
    return out
