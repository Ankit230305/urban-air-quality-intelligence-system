from __future__ import annotations

import streamlit as st
import pandas as pd

from src.data.live import live_row

st.set_page_config(
    page_title="Urban AQI — Live",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Live Air Quality (Current)")
st.caption(
    "Pulls current pollutants and weather via OpenWeatherMap "
    "(and optional WAQI AQI)."
)

with st.sidebar:
    st.subheader("Location")
    city = st.text_input("City", value="Mumbai")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat = st.number_input("Latitude", value=19.0760, format="%.6f")
    with col_lon:
        lon = st.number_input("Longitude", value=72.8777, format="%.6f")
    refresh = st.button("Fetch now")

if refresh:
    df = live_row(city, float(lat), float(lon))
    if df.empty:
        st.error(
            "No live data. Check your API keys in the environment "
            "(OPENWEATHERMAP_API_KEY, optional WAQI_TOKEN)."
        )
    else:
        top = df.iloc[0]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("PM2.5 (µg/m³)", f"{top.get('pm2_5', float('nan')):.1f}")
        m2.metric("PM10 (µg/m³)", f"{top.get('pm10', float('nan')):.1f}")
        m3.metric("NO₂ (µg/m³)", f"{top.get('no2', float('nan')):.1f}")
        m4.metric("O₃ (µg/m³)", f"{top.get('o3', float('nan')):.1f}")
        m5.metric("SO₂ (µg/m³)", f"{top.get('so2', float('nan')):.1f}")
        m6.metric("CO (µg/m³)", f"{top.get('co', float('nan')):.1f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temp (°C)", f"{top.get('temp', float('nan')):.1f}")
        c2.metric("Humidity (%)", f"{top.get('humidity', float('nan')):.0f}")
        c3.metric(
            "Wind (m/s)", f"{top.get('wind_speed', float('nan')):.1f}"
        )
        c4.metric("OWM AQI (1–5)", f"{top.get('aqi_owm', float('nan'))}")

        if pd.notna(top.get("aqi_waqi", float("nan"))):
            st.metric("WAQI (US AQI)", int(top["aqi_waqi"]))

        st.divider()
        st.subheader("Raw live payload")
        st.dataframe(df, use_container_width=True)
else:
    st.info("Choose a city/coords in the left sidebar and click **Fetch now**.")
