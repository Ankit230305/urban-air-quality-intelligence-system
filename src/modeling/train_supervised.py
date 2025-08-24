import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

def make_aqi_cat(aqi):
    try:
        aqi = float(aqi)
    except Exception:
        return np.nan
    if aqi < 51: return "Good"
    if aqi < 101: return "Moderate"
    if aqi < 201: return "Unhealthy"
    if aqi < 301: return "Very Unhealthy"
    return "Hazardous"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--city", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file, parse_dates=["datetime"]).sort_values("datetime")
    # features to use (whatever is available)
    X_cols = [c for c in ["temp","humidity","wind_speed","precip","pm10","no2","o3","so2","co"] if c in df.columns]
    if "pm2_5" not in df.columns or len(X_cols) == 0:
        raise SystemExit("Insufficient columns. Need pm2_5 and at least one of: " + ", ".join(["temp","humidity","wind_speed","precip","pm10","no2","o3","so2","co"]))

    # --- Coerce numeric ---
    for col in X_cols + ["pm2_5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Impute features robustly: ffill/bfill then median for any remaining ---
    X_all = df[X_cols].copy()
    X_all = X_all.ffill().bfill()
    X_all = X_all.fillna(X_all.median(numeric_only=True))

    # --- Interpolate pm2_5 target and drop remaining NaN ---
    y_all = df["pm2_5"].interpolate(limit_direction="both")
    mask = y_all.notna() & X_all.notna().all(axis=1)
    X = X_all.loc[mask].to_numpy()
    y = y_all.loc[mask].to_numpy(dtype=float)

    if len(y) < 10:
        raise SystemExit(f"Not enough clean rows after preprocessing: {len(y)}")

    # Temporal split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=max(1, int(0.2*len(y))), shuffle=False)

    # ------- Regression: predict pm2_5 -------
    reg = RandomForestRegressor(n_estimators=300, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Filter any NaNs just in case
    valid = np.isfinite(y_test) & np.isfinite(y_pred)
    mae  = mean_absolute_error(y_test[valid], y_pred[valid])
    mse = mean_squared_error(y_test[valid], y_pred[valid]); rmse = mse**0.5
    r2   = r2_score(y_test[valid], y_pred[valid])

    joblib.dump(reg, outdir / f"rf_regressor_{args.city.lower().replace(' ','_')}.joblib")

    # ------- Classification: AQI category (optional if aqi exists) -------
    acc = f1 = np.nan
    if "aqi" in df.columns:
        df["aqi_cat"] = df["aqi"].apply(make_aqi_cat)
        cls_mask = mask & df["aqi_cat"].notna()
        # Use the same X preprocessed rows aligned via mask
        Xc = X_all.loc[cls_mask]
        yc = df.loc[cls_mask, "aqi_cat"]
        if len(yc) > 20 and yc.nunique() > 1:
            Xc = Xc.to_numpy()
            # temporal split
            Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=max(1, int(0.2*len(yc))), shuffle=False)
            clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
            clf.fit(Xc_train, yc_train)
            yc_pred = clf.predict(Xc_test)
            acc = accuracy_score(yc_test, yc_pred)
            f1  = f1_score(yc_test, yc_pred, average="weighted")
            joblib.dump(clf, outdir / f"rf_classifier_{args.city.lower().replace(' ','_')}.joblib")

    # Report
    rep = outdir / f"supervised_report_{args.city.lower().replace(' ','_')}.md"
    with open(rep, "w") as f:
        f.write(f"# Supervised Models – {args.city}\n\n")
        f.write("## Regression (PM2.5)\n")
        f.write(f"- MAE: {mae:.3f}\n- RMSE: {rmse:.3f}\n- R²: {r2:.3f}\n\n")
        f.write("## Classification (AQI category)\n")
        f.write(f"- Accuracy: {acc}\n- F1 (weighted): {f1}\n")

    print(f"✅ Saved models+report to {outdir}")

if __name__ == "__main__":
    main()
