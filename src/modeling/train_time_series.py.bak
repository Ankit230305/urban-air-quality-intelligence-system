import argparse
from pathlib import Path

import pandas as pd
from prophet import Prophet


def main():
    parser = argparse.ArgumentParser(description="Train Prophet 7-day forecast for PM2.5")
    parser.add_argument("--input-file", required=True, help="Path to processed features CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to write model/forecast")
    parser.add_argument("--target", default="pm2_5", help="Target column (default: pm2_5)")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_col = args.target

    # Load data
    df = pd.read_csv(in_path)

    # Ensure datetime exists and strip any timezone, then aggregate to daily means
    if "datetime" not in df.columns:
        raise SystemExit("Input file must contain a 'datetime' column")

    ts = pd.to_datetime(df["datetime"], errors="coerce")
    # robust timezone stripping
    try:
        ts = ts.dt.tz_convert(None)
    except Exception:
        pass
    try:
        ts = ts.dt.tz_localize(None)
    except Exception:
        pass

    if target_col not in df.columns:
        # fallback if the column was named pm25
        if "pm25" in df.columns:
            target_col = "pm25"
        else:
            raise SystemExit(f"Target column '{args.target}' not found in {in_path.name}")

    work = pd.DataFrame({"datetime": ts, target_col: pd.to_numeric(df[target_col], errors="coerce")})
    work = work.dropna(subset=[target_col]).set_index("datetime").sort_index()

    # Resample to daily mean to avoid gaps/irregular hourly cadence
    daily = work.resample("D").mean(numeric_only=True).dropna().reset_index()

    if len(daily) < 7:
        raise SystemExit(f"Not enough history after resampling (got {len(daily)} daily points)")

    ts_df = daily.rename(columns={"datetime": "ds", target_col: "y"})

    # Fit Prophet
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(ts_df)

    # 7-day ahead daily forecast
    future = m.make_future_dataframe(periods=7, freq="D")
    forecast = m.predict(future)

    # Save outputs
    forecast_out = out_dir / "forecast_pm25.csv"
    model_out = out_dir / "prophet_pm25_model.pkl"

    forecast.to_csv(forecast_out, index=False)
    try:
        # Prophet models are picklable
        import joblib
        joblib.dump(m, model_out)
    except Exception:
        # If joblib fails due to environment differences, at least keep forecast CSV
        pass

    print(f"✅ Saved forecast to {forecast_out} (rows={len(forecast)})")
    print(f"📦 Model saved to {model_out} (if joblib succeeded)")


if __name__ == "__main__":
    main()
