import argparse
from pathlib import Path

import pandas as pd
from prophet import Prophet

def make_naive(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--city", default="")
    ap.add_argument("--target", default="pm2_5")
    ap.add_argument("--periods", type=int, default=168)
    ap.add_argument("--freq", default="h", help="future frequency (default: 'h' hourly)")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = args.city.lower().replace(" ", "_") if args.city else ""

    df = pd.read_csv(args.input_file, parse_dates=["datetime"]).sort_values("datetime")
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found in {args.input_file}")

    ts = df[["datetime", args.target]].rename(columns={"datetime": "ds", args.target: "y"}).copy()
    ts["ds"] = make_naive(ts["ds"])
    ts["y"] = pd.to_numeric(ts["y"], errors="coerce")
    ts = ts.dropna(subset=["ds", "y"])
    if ts.empty or len(ts) < 20:
        raise SystemExit("Not enough data to train Prophet (need at least ~20 rows).")

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(ts)

    freq = (args.freq or "h").lower()
    future = m.make_future_dataframe(periods=args.periods, freq=freq, include_history=True)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    out_file = outdir / (f"forecast_pm25_{slug}.csv" if slug else "forecast_pm25.csv")
    fcst.to_csv(out_file, index=False)
    print(f"✅ Saved forecast to {out_file} (rows={len(fcst)})")

if __name__ == "__main__":
    main()
