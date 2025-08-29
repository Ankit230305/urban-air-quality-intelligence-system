from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.utils.live_fetch import fetch_live_point, livepoint_to_df
from src.utils.clean import coerce_none_like, fill_missing_for_display
from src.utils.paths import resolve_processed, resolve_forecast_path

st.set_page_config(page_title="Urban AQI", page_icon="🌆", layout="wide")
st.title("Urban Air Quality Intelligence System")

DATA_DIR = Path("data")
PROC = DATA_DIR / "processed"
REPORTS = Path("reports")
MODELS = Path("models")


# ---------- helpers ----------
def load_csv_safe(p: Path, parse_dates=None) -> pd.DataFrame | None:
    try:
        if not p.exists():
            return None
        return pd.read_csv(p, parse_dates=parse_dates)
    except Exception:
        return None


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def synthetic_processed(city: str) -> pd.DataFrame:
    rng = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=7 * 24, freq="h")
    rs = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "datetime": rng,
            "city": city,
            "pm2_5": np.clip(rs.normal(35, 12, len(rng)), 5, 150),
            "pm10": np.clip(rs.normal(70, 25, len(rng)), 10, 200),
            "no2": np.clip(rs.normal(18, 6, len(rng)), 1, 60),
            "o3": np.clip(rs.normal(25, 10, len(rng)), 1, 120),
            "so2": np.clip(rs.normal(8, 3, len(rng)), 1, 40),
            "co": np.clip(rs.normal(400, 120, len(rng)), 100, 1200),
            "temp": np.clip(rs.normal(298, 4, len(rng)), 285, 310),
            "humidity": np.clip(rs.normal(55, 15, len(rng)), 15, 95),
            "wind_speed": np.clip(rs.normal(2.5, 1, len(rng)), 0, 8),
            "latitude": 0.0,
            "longitude": 0.0,
        }
    )


# ---------- sidebar ----------
with st.sidebar:
    st.header("Controls")
    city = st.text_input("City", value="Hyderabad")
    lat = st.number_input("Latitude", value=17.3850, format="%.6f")
    lon = st.number_input("Longitude", value=78.4867, format="%.6f")
    st.caption("Run the pipeline from the terminal to regenerate files.")

tabs = st.tabs(
    ["Overview", "EDA", "Patterns", "Forecasts", "Anomalies", "Health", "Models", "Live Now"]
)
slug = city.lower().replace(" ", "_")

paths = {
    "processed": PROC / f"{slug}_features_plus_demo.csv",  # preferred (UI will resolve fallbacks)
    "anoms": PROC / f"{slug}_anomalies.csv",
    "health": PROC / f"{slug}_health.csv",
    "forecast": MODELS / f"forecast_pm25_{slug}.csv",  # UI will fallback to generic if missing
    "seasonal": REPORTS / f"seasonal_{slug}.csv",
    "assoc": REPORTS / f"assoc_rules_{slug}.csv",
    "patterns": REPORTS / f"patterns_{slug}.md",
    "live": PROC / f"{slug}__live.csv",
}

# ---------- Overview ----------
with tabs[0]:
    st.subheader("Overview")
    fdf = resolve_processed(city)
    if fdf is None or fdf.empty:
        st.warning("No processed file found. Showing sample data for demo.")
        fdf = synthetic_processed(city)

    fdf = coerce_none_like(fdf)
    fdf = ensure_numeric(
        fdf,
        [
            "pm2_5",
            "pm10",
            "no2",
            "o3",
            "so2",
            "co",
            "aqi",
            "temp",
            "humidity",
            "wind_speed",
            "latitude",
            "longitude",
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    display_aqi = np.nan
    if "aqi" in fdf and fdf["aqi"].notna().any():
        display_aqi = float(fdf["aqi"].dropna().tail(1).mean())
    elif "pm2_5" in fdf and fdf["pm2_5"].notna().any():
        display_aqi = float(fdf["pm2_5"].tail(24).mean())  # proxy for display

    c1.metric("Current AQI (or PM2.5 proxy)", "—" if np.isnan(display_aqi) else f"{display_aqi:.1f}")
    c2.metric("Rows", f"{len(fdf):,}")
    if "datetime" in fdf.columns:
        c3.metric("Last timestamp", str(pd.to_datetime(fdf["datetime"]).max()))
    c4.metric("Last anomaly", "See Anomalies tab")

    st.dataframe(fdf.tail(300), use_container_width=True)

# ---------- EDA ----------
with tabs[1]:
    st.subheader("EDA")
    df = resolve_processed(city)
    used_synth = False
    if df is None or df.empty:
        st.info("Sample data shown (run pipeline to see real data).")
        df = synthetic_processed(city)
        used_synth = True

    df = coerce_none_like(df)
    df = ensure_numeric(
        df,
        [
            "pm2_5",
            "pm10",
            "no2",
            "o3",
            "so2",
            "co",
            "temp",
            "humidity",
            "wind_speed",
            "latitude",
            "longitude",
        ],
    )

    if "datetime" in df and "pm2_5" in df:
        st.plotly_chart(
            px.line(df, x="datetime", y="pm2_5", title="PM2.5 over time"),
            use_container_width=True,
        )

    num_cols = [c for c in ["pm2_5", "pm10", "no2", "o3", "so2", "co"] if c in df.columns]
    if num_cols:
        st.plotly_chart(
            px.histogram(df, x=num_cols, marginal="rug", title="Pollutant distributions", barmode="overlay"),
            use_container_width=True,
        )

    if {"pm2_5", "temp"}.issubset(df.columns):
        st.plotly_chart(
            px.scatter(df, x="temp", y="pm2_5", trendline="ols", title="PM2.5 vs Temperature"),
            use_container_width=True,
        )
    if {"pm2_5", "humidity"}.issubset(df.columns):
        st.plotly_chart(
            px.scatter(df, x="humidity", y="pm2_5", trendline="ols", title="PM2.5 vs Humidity"),
            use_container_width=True,
        )

    if "datetime" in df and "pm2_5" in df:
        tmp = df[["datetime", "pm2_5"]].dropna().copy()
        tmp["weekday"] = pd.to_datetime(tmp["datetime"]).dt.dayofweek
        tmp["hour"] = pd.to_datetime(tmp["datetime"]).dt.hour
        pivot = tmp.pivot_table(index="weekday", columns="hour", values="pm2_5", aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", title="PM2.5 seasonality (weekday × hour)"),
                        use_container_width=True)

    corr_cols = [c for c in num_cols + ["temp", "humidity", "wind_speed"] if c in df.columns]
    if corr_cols:
        corr = df[corr_cols].corr(numeric_only=True).fillna(0)
        st.plotly_chart(px.imshow(corr, text_auto=False, title="Correlation heatmap"),
                        use_container_width=True)

    if {"latitude", "longitude", "pm2_5"}.issubset(df.columns):
        mp = df.dropna(subset=["latitude", "longitude"]).tail(1000).copy()
        if not mp.empty:
            st.plotly_chart(
                px.scatter_mapbox(
                    mp,
                    lat="latitude",
                    lon="longitude",
                    color="pm2_5",
                    color_continuous_scale="Turbo",
                    zoom=9,
                    height=400,
                    title="Locations colored by PM2.5",
                ).update_layout(mapbox_style="open-street-map"),
                use_container_width=True,
            )

# ---------- Patterns ----------
with tabs[2]:
    st.subheader("Seasonality, clustering & association rules")
    seas = load_csv_safe(paths["seasonal"])
    assoc = load_csv_safe(paths["assoc"])
    pat_md = paths["patterns"].read_text() if paths["patterns"].exists() else ""

    if seas is not None and not seas.empty:
        st.markdown("**Daily seasonality (summary)**")
        st.dataframe(seas, use_container_width=True)
    else:
        st.info("No seasonal CSV yet. If processed data exists, run patterns step.")

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

# ---------- Forecasts ----------
with tabs[3]:
    st.subheader("7-day PM2.5 forecast")
    fpath = resolve_forecast_path(slug)
    fdf = load_csv_safe(fpath, parse_dates=["ds"])
    if fdf is not None and not fdf.empty:
        st.dataframe(fdf.tail(200), use_container_width=True)
        if {"ds", "yhat"}.issubset(fdf.columns):
            st.plotly_chart(px.line(fdf, x="ds", y="yhat", title="PM2.5 forecast"),
                            use_container_width=True)
    else:
        st.info("No forecast file yet.")

# ---------- Anomalies ----------
with tabs[4]:
    st.subheader("Anomalies")
    adf = load_csv_safe(paths["anoms"], parse_dates=["datetime"])
    if adf is not None and not adf.empty:
        st.dataframe(adf.tail(500), use_container_width=True)
        if "score" in adf.columns:
            st.plotly_chart(px.line(adf, x="datetime", y="score", title="Anomaly scores over time"),
                            use_container_width=True)
    else:
        st.info("No anomalies file yet.")

# ---------- Health ----------
with tabs[5]:
    st.subheader("Health risk")
    hdf = load_csv_safe(paths["health"], parse_dates=["datetime"])
    if hdf is not None and not hdf.empty:
        st.dataframe(hdf.tail(500), use_container_width=True)
        ycol = (
            "health_risk_score"
            if "health_risk_score" in hdf.columns
            else ("risk_index" if "risk_index" in hdf.columns else "aqi")
        )
        if ycol in hdf.columns:
            st.plotly_chart(px.line(hdf, x="datetime", y=ycol, title=f"Health metric: {ycol}"),
                            use_container_width=True)
        band = hdf["health_risk_band"].iloc[-1] if "health_risk_band" in hdf.columns else "Unknown"
        recs = {
            "Low": "Normal activities. Keep windows open.",
            "Moderate": "Sensitive groups should limit prolonged outdoor exertion.",
            "High": "Wear a mask outdoors; reduce outdoor cardio; use air purifier.",
            "Very High": "Avoid outdoor activity; N95 mask if stepping out.",
            "Unknown": "No band available; follow general caution.",
        }
        st.info(f"Current risk band: **{band}** — {recs.get(band, recs['Unknown'])}")
    else:
        st.info("No health file yet.")

# ---------- Models ----------
with tabs[6]:
    st.subheader("Model Scores")
    mfile = MODELS / f"supervised_metrics_{slug}.json"
    if not mfile.exists():
        mfile = MODELS / "supervised_metrics.json"
    if mfile.exists():
        import json
        metrics = json.loads(mfile.read_text())
        if isinstance(metrics, dict):
            rows = []
            for model_name, m in metrics.items():
                if isinstance(m, dict):
                    row = {"model": model_name}
                    for k, v in m.items():
                        if isinstance(v, (int, float, str)):
                            row[k] = v
                    rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.json(metrics)
        else:
            st.json(metrics)
    else:
        st.info("No supervised metrics file found yet. Train models to populate this tab.")

# ---------- Live Now ----------
with tabs[7]:
    st.subheader("Live Now")
    if st.button("Fetch live snapshot"):
        try:
            lp = fetch_live_point(city, float(lat), float(lon))
            df_live = livepoint_to_df(lp)
            st.success(f"Fetched at {lp.fetched_at.isoformat()}")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("PM2.5 (µg/m³)", f"{lp.pm2_5:.1f}")
            m2.metric("PM10 (µg/m³)", f"{lp.pm10:.1f}")
            m3.metric("NO₂ (µg/m³)", f"{lp.no2:.1f}")
            m4.metric("O₃ (µg/m³)", f"{lp.o3:.1f}")
            m5.metric("SO₂ (µg/m³)", f"{lp.so2:.1f}")
            m6.metric("CO (µg/m³)", f"{lp.co:.1f}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temp (K)", f"{(lp.temp if lp.temp is not None else float('nan')):.1f}")
            c2.metric("Humidity (%)", f"{(lp.humidity if lp.humidity is not None else float('nan')):.0f}")
            c3.metric("Wind (m/s)", f"{(lp.wind_speed if lp.wind_speed is not None else float('nan')):.1f}")
            c4.metric("OWM AQI (1–5)", f"{lp.aqi}")
            st.dataframe(df_live, use_container_width=True)
            if st.checkbox("Append this snapshot to live CSV", value=True):
                paths["live"].parent.mkdir(parents=True, exist_ok=True)
                if paths["live"].exists():
                    base = pd.read_csv(paths["live"])
                    df_live = pd.concat([base, df_live], ignore_index=True)
                df_live.to_csv(paths["live"], index=False)
                st.caption(f"Saved to {paths['live']}")
        except RuntimeError as exc:
            st.error(
                "Live fetch failed.\n\n"
                "Tips:\n"
                "• Ensure OPENWEATHERMAP_API_KEY is set in .env or your shell.\n"
                "• Check network connectivity.\n"
                f"• Details: {exc}"
            )
