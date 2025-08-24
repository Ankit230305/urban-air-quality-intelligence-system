import argparse
from pathlib import Path
import pandas as pd
import numpy as np

POLLUTANTS = ["pm2_5","pm10","no2","o3","so2","co"]

def parse_args():
    ap = argparse.ArgumentParser(description="Simple health risk scoring & advisories")
    ap.add_argument("--input-file", required=True, help="features CSV with datetime & pollutants")
    ap.add_argument("--demographics", required=False, default="data/external/demographics_india.csv")
    ap.add_argument("--city", required=False, default=None)
    ap.add_argument("--output", required=True, help="where to write health CSV")
    return ap.parse_args()

def aqi_category_from_pm25(pm25):
    # US EPA style
    if pd.isna(pm25): return "Unknown"
    x = float(pm25)
    if x <= 12: return "Good"
    if x <= 35.4: return "Moderate"
    if x <= 55.4: return "Unhealthy (SG)"
    if x <= 150.4: return "Unhealthy"
    if x <= 250.4: return "Very Unhealthy"
    return "Hazardous"

def risk_from_pm25(pm25, elderly=8.0):
    # heuristic risk 0..1; bump with elderly share
    if pd.isna(pm25): return 0.3
    x = max(0.0, min(500.0, float(pm25)))
    base = x / 150.0  # saturate ~ Unhealthy
    return float(min(1.0, base * (1.0 + elderly/100.0)))

def main():
    args = parse_args()
    df = pd.read_csv(args.input_file, parse_dates=["datetime"])
    try:
        df["datetime"] = df["datetime"].dt.tz_convert(None)
    except Exception:
        try:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        except Exception:
            pass

    # coerce numerics
    for c in POLLUTANTS + ["temp","humidity","wind_speed","precip","aqi","population","pop_density_per_km2","pct_elderly","pct_children","respiratory_illness_rate_per_100k","latitude","longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure aqi_category exists
    if "aqi_category" not in df.columns:
        df["aqi_category"] = df["pm2_5"].apply(aqi_category_from_pm25) if "pm2_5" in df.columns else "Unknown"

    # demographics join (optional)
    if args.city and Path(args.demographics).exists():
        d = pd.read_csv(args.demographics)
        # simple single-row city match
        row = d.loc[d["city"].str.lower() == args.city.lower()].head(1)
        if not row.empty:
            for c in ["population","pop_density_per_km2","pct_elderly","pct_children","respiratory_illness_rate_per_100k"]:
                if c in d.columns:
                    df[c] = df[c].fillna(row.iloc[0].get(c))

    elderly = df["pct_elderly"].fillna(8.0).iloc[0] if "pct_elderly" in df.columns and not df.empty else 8.0
    df["health_risk_score"] = df.get("pm2_5", pd.Series(index=df.index, dtype=float)).apply(lambda x: risk_from_pm25(x, elderly))
    def band(x):
        if pd.isna(x): return "Unknown"
        if x < 0.25: return "Low"
        if x < 0.5: return "Moderate"
        if x < 0.75: return "High"
        return "Very High"
    df["health_risk_band"] = df["health_risk_score"].apply(band)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"✅ health file saved: {args.output} (rows={len(df)})")

if __name__ == "__main__":
    main()
