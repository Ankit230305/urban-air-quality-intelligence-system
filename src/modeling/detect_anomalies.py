import argparse
from pathlib import Path
import pandas as pd
import numpy as np

POLLUTANTS = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]


def parse_args():
    ap = argparse.ArgumentParser(description="Detect pollution spikes via rolling z-score")
    ap.add_argument("--input-file", required=True, help="Processed features CSV with 'datetime'")
    ap.add_argument("--output", required=True, help="Where to write anomalies CSV")
    ap.add_argument("--target", default="pm2_5", help="Column to detect anomalies on (default: pm2_5)")
    ap.add_argument("--window", type=int, default=24, help="Rolling window length (hours)")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold for anomaly")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file, parse_dates=["datetime"])
    # normalize tz
    try:
        df["datetime"] = df["datetime"].dt.tz_convert(None)
    except Exception:
        try:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        except Exception:
            pass

    # coerce numerics
    for c in POLLUTANTS + ["temp", "humidity", "wind_speed", "precip", "aqi", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("datetime").copy()
    # drop rows where all pollutants are missing
    if set(POLLUTANTS).intersection(df.columns):
        df = df.dropna(subset=list(set(POLLUTANTS).intersection(df.columns)), how="all")

    target = args.target
    if target not in df.columns:
        # nothing to do, write empty with header
        pd.DataFrame(columns=["datetime", target, "z_score", "is_anomaly"]).to_csv(args.output, index=False)
        print(f"⚠️ {target} not in columns; wrote empty anomalies file {args.output}")
        return

    # rolling mean/std (centered)
    window = max(3, int(args.window))
    roll_mean = df[target].rolling(window=window, min_periods=1, center=True).mean()
    roll_std = df[target].rolling(window=window, min_periods=1, center=True).std(ddof=0).replace(0, np.nan)

    z = (df[target] - roll_mean) / roll_std
    z = z.fillna(0.0)
    is_anom = (z.abs() >= float(args.z_thresh))

    out_cols = ["datetime", target, "temp", "humidity", "wind_speed", "precip", "latitude", "longitude"]
    out_cols = [c for c in out_cols if c in df.columns]
    out = df[out_cols].copy()
    out["z_score"] = z.values
    out["is_anomaly"] = is_anom.values

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"✅ anomalies saved: {args.output} (rows={len(out)})")


if __name__ == "__main__":
    main()
