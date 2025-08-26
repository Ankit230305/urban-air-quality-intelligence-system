#!/usr/bin/env python3
import os, sys
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

def slug_of(city: str) -> str:
    return city.lower().replace(" ", "_")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not key:
        sys.exit("ERROR: Missing OPENWEATHERMAP_API_KEY")

    aq = requests.get(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        params={"lat": args.lat, "lon": args.lon, "appid": key},
        timeout=30,
    ).json()

    wx = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": args.lat, "lon": args.lon, "appid": key, "units": "metric"},
        timeout=30,
    ).json()

    comp = aq["list"][0]["components"]
    aqi = aq["list"][0]["main"]["aqi"]
    tm = datetime.fromtimestamp(aq["list"][0]["dt"], tz=timezone.utc).replace(tzinfo=None)

    row = {
        "datetime": tm.isoformat(sep=" "),
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "no2": comp.get("no2"),
        "o3": comp.get("o3"),
        "so2": comp.get("so2"),
        "co": comp.get("co"),
        "aqi": aqi,
        "temp": wx["main"]["temp"],
        "humidity": wx["main"]["humidity"],
        "wind_speed": wx["wind"]["speed"],
        "precip": (wx.get("rain", {}).get("1h", 0.0) or
                   wx.get("snow", {}).get("1h", 0.0) or 0.0),
        "city": args.city,
    }
    df = pd.DataFrame([row])

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    slug = slug_of(args.city)
    fplus = outdir / f"{slug}__features_plus_demo.csv"

    if fplus.exists():
        existing = pd.read_csv(fplus, parse_dates=["datetime"])
        for c in ["pm2_5","pm10","no2","o3","so2","co","aqi","temp","humidity","wind_speed","precip"]:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        # align to existing columns (keep any demographics cols)
        for c in existing.columns:
            if c not in df.columns:
                df[c] = pd.NA
        df = df.reindex(columns=existing.columns)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(fplus, index=False)
        print(f"✅ Appended 1 live row to {fplus} (rows={len(combined)})")
    else:
        df.to_csv(fplus, index=False)
        print(f"✅ Wrote {fplus} (rows=1)")

if __name__ == "__main__":
    main()
