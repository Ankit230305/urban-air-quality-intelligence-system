import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.live_fetch import fetch_live_point, livepoint_to_df


# ---------- helpers ----------
def slug_of(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_csv_safe(path: Path,
                  parse_dates: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as exc:
        st.warning(f"Failed to read {path.name}: {exc}")
        return None


def as_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_concat(parts: List[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def metric_row(aqi: float, pm25: float, pm10: float) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("AQI (OWM scale 1–5)", f"{aqi:.0f}")
    c2.metric("PM2.5 (µg/m³)", f"{pm25:.1f}")
    c3.metric("PM10 (µg/m³)", f"{pm10:.1f}")


# ---------- sidebar ----------
st.set_page_config(page_title="Urban AQ Intelligence", layout="wide")

st.sidebar.header("City & Dates")
city = st.sidebar.selectbox(
    "City",
    ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Kolkata",
     "Vizag", "Vellore"],
    index=1,
)
coords = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Vizag": (17.6868, 83.2185),
    "Vellore": (12.9165, 79.1325),
}
lat, lon = coords[city]

st.sidebar.caption(
    "Tip: Use the **Run pipeline** shell script for historical windows:\n"
    "`bin/run_city_pipeline.sh \"CITY\" lat lon YYYY-MM-DD YYYY-MM-DD`"
)

# ---------- tabs ----------
tabs = st.tabs(
    [
        "Live",
        "EDA",
        "Patterns",
        "Forecast",
        "Anomalies",
        "Health",
    ]
)

slug = slug_of(city)
paths = {
    "processed": Path("data/processed") / f"{slug}__features_plus_demo.csv",
    "seasonal": Path("reports") / f"seasonal_{slug}.csv",
    "assoc": Path("reports") / f"assoc_rules_{slug}.csv",
    "patterns": Path("reports") / f"patterns_{slug}.md",
    "forecast": Path("models") / "forecast_pm25.csv",
    "anoms": Path("data/processed") / f"{slug}__anomalies.csv",
    "health": Path("data/processed") / f"{slug}__health.csv",
    "live": Path("data/processed") / f"{slug}__live.csv",
    "metrics": Path("models") / f"supervised_metrics_{slug}.json",
}

# ---------- Live tab ----------
with tabs[0]:
    st.subheader(f"Live snapshot — {city}")
    c1, c2, c3, c4 = st.columns(4)
    lat_i = c1.number_input("Latitude", value=float(lat), format="%.4f")
    lon_i = c2.number_input("Longitude", value=float(lon), format="%.4f")
    fetch_btn = c3.button("Fetch live now")
    append_btn = c4.checkbox("Append to live CSV", value=True)

    if fetch_btn:
        lp = fetch_live_point(city, lat_i, lon_i)
        df_live_now = livepoint_to_df(lp)
        metric_row(
            aqi=float(df_live_now["aqi"].iloc[0]),
            pm25=float(df_live_now["pm2_5"].iloc[0]),
            pm10=float(df_live_now["pm10"].iloc[0]),
        )
        st.dataframe(df_live_now, use_container_width=True)

        if append_btn:
            if paths["live"].exists():
                old = pd.read_csv(paths["live"])
                out = safe_concat([old, df_live_now])
                out = out.drop_duplicates(subset=["datetime"], keep="last")
            else:
                out = df_live_now
            paths["live"].parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(paths["live"], index=False)
            st.success(f"Updated {paths['live']} (rows={len(out)})")

    df_hist = load_csv_safe(paths["live"], parse_dates=["datetime"])
    if df_hist is not None and not df_hist.empty:
        st.markdown("**Live history**")
        st.dataframe(df_hist.tail(200), use_container_width=True)
        st.plotly_chart(
            px.line(
                df_hist.sort_values("datetime"),
                x="datetime",
                y=["pm2_5", "pm10", "no2", "o3", "so2", "co"],
                title="Recent pollutant levels (live feed)",
            ),
            use_container_width=True,
        )
    else:
        st.info("No live CSV yet. Click **Fetch live now** to create it.")

# ---------- EDA tab ----------
with tabs[1]:
    st.subheader(f"EDA — {city}")
    df = load_csv_safe(paths["processed"], parse_dates=["datetime"])
    if df is None or df.empty:
        st.info("No processed file found yet. Run the pipeline for this city.")
    else:
        df = as_numeric(
            df,
            ["pm2_5", "pm10", "no2", "o3", "so2", "co",
             "aqi", "temp", "humidity", "wind_speed", "precip"],
        )
        st.markdown("**Sample (last 500 rows)**")
        st.dataframe(df.tail(500), use_container_width=True)

        st.markdown("**Summary**")
        c1, c2 = st.columns(2)
        c1.dataframe(df.describe(include="all"), use_container_width=True)
        miss = df.isna().mean().sort_values(ascending=False) * 100.0
        c2.plotly_chart(
            px.bar(
                miss.head(20),
                title="Missing values (%) — top 20",
                labels={"value": "missing %", "index": "column"},
            ),
            use_container_width=True,
        )

        st.markdown("**Distributions**")
        feat_cols = ["pm2_5", "pm10", "no2", "o3", "so2", "co", "aqi"]
        for c in feat_cols:
            if c in df.columns:
                st.plotly_chart(
                    px.histogram(df, x=c, nbins=40, title=f"Histogram: {c}"),
                    use_container_width=True,
                )

        st.markdown("**Correlation heatmap (numeric cols)**")
        num = df.select_dtypes(include=[np.number]).copy()
        if not num.empty:
            corr = num.corr(numeric_only=True)
            st.plotly_chart(
                px.imshow(
                    corr,
                    title="Correlation heatmap",
                    color_continuous_scale="RdBu",
                    zmin=-1, zmax=1,
                ),
                use_container_width=True,
            )

# ---------- Patterns tab ----------
with tabs[2]:
    st.subheader("Seasonality, clustering & association rules")
    seas = load_csv_safe(paths["seasonal"])
    assoc = load_csv_safe(paths["assoc"])
    pat_md = paths["patterns"].read_text() if paths["patterns"].exists() else ""

    if seas is not None and not seas.empty:
        st.markdown("**Daily seasonality (summary)**")
        st.dataframe(seas, use_container_width=True)
    else:
        st.info("No seasonal CSV yet.")

    if assoc is not None and not assoc.empty:
        st.markdown("**Association rules (top)**")
        st.dataframe(assoc.head(2000), use_container_width=True)
    else:
        st.info("No association rules CSV yet.")

    if pat_md:
        st.markdown("**Pattern notes**")
        st.markdown(pat_md)
    else:
        st.info("No patterns markdown yet.")

# ---------- Forecast tab ----------
with tabs[3]:
    st.subheader("7-day PM2.5 forecast")
    fdf = load_csv_safe(paths["forecast"], parse_dates=["ds"])
    if fdf is not None and not fdf.empty:
        st.dataframe(fdf.tail(200), use_container_width=True)
        if {"ds", "yhat"}.issubset(fdf.columns):
            st.plotly_chart(
                px.line(fdf, x="ds", y="yhat", title="PM2.5 forecast"),
                use_container_width=True,
            )
    else:
        st.info("No forecast file yet.")

# ---------- Anomalies tab ----------
with tabs[4]:
    st.subheader("Anomalies")
    adf = load_csv_safe(paths["anoms"], parse_dates=["datetime"])
    if adf is not None and not adf.empty:
        st.dataframe(adf.tail(500), use_container_width=True)
        if "score" in adf.columns:
            st.plotly_chart(
                px.line(adf, x="datetime", y="score",
                        title="Anomaly scores over time"),
                use_container_width=True,
            )
    else:
        st.info("No anomalies file yet.")

# ---------- Health tab ----------
with tabs[5]:
    st.subheader("Health risk")
    hdf = load_csv_safe(paths["health"], parse_dates=["datetime"])
    if hdf is not None and not hdf.empty:
        st.dataframe(hdf.tail(500), use_container_width=True)
        ycol = "risk_index" if "risk_index" in hdf.columns else "aqi"
        if ycol in hdf.columns:
            st.plotly_chart(
                px.line(hdf, x="datetime", y=ycol,
                        title=f"Health metric: {ycol}"),
                use_container_width=True,
            )
    else:
        st.info("No health file yet.")
