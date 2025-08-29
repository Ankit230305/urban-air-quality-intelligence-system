import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _save_metrics(metrics: dict, city: str) -> None:
    """Persist supervised model metrics next to models/."""
    slug = (city or "").lower().replace(" ", "_")
    out = Path("models") / (
        f"supervised_metrics_{slug}.json" if slug else "supervised_metrics.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(metrics, f, indent=2)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

POLLUTANTS = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]
WEATHER = ["temp", "humidity", "wind_speed", "precip"]


def slug_of(city: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in city.lower())


def aqi_category(pm25: float) -> str:
    if pm25 < 51:
        return "Good"
    if pm25 < 101:
        return "Moderate"
    if pm25 < 201:
        return "Unhealthy"
    if pm25 < 301:
        return "Very Unhealthy"
    return "Hazardous"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("datetime")
    for c in POLLUTANTS + WEATHER + ["aqi", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "datetime" in df.columns:
        df["hour"] = df["datetime"].dt.hour
        df["dow"] = df["datetime"].dt.dayofweek
    else:
        df["hour"] = 0
        df["dow"] = 0

    if "pm2_5" in df.columns:
        df["pm2_5_lag1"] = df["pm2_5"].shift(1)
        df["pm2_5_lag3"] = df["pm2_5"].shift(3)
        df["pm2_5_roll6h"] = df["pm2_5"].rolling(6, min_periods=1).mean()

    return df


def safe_feature_list(df: pd.DataFrame, target: str) -> list:
    base = ["hour", "dow", "temp", "humidity", "wind_speed", "precip", "pm2_5_lag1", "pm2_5_roll6h"]
    feats = [c for c in base + POLLUTANTS if (c in df.columns and c != target)]
    seen, out = set(), []
    for c in feats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    for c in ["hour", "temp", "humidity", "wind_speed", "precip"]:
        if c in df.columns and c not in out:
            out.append(c)
    return out


def numeric_only(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    keep = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    return X[keep]


def fill_and_prune(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    drop_cols = [c for c in X.columns if X[c].isna().all()]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)
    return X


def split_time_ordered(X: pd.DataFrame, y: pd.Series, min_test: int = 1):
    n = len(X)
    if n <= min_test:
        return X.iloc[:0], X, y.iloc[:0], y
    test_size = max(int(round(n * 0.2)), min_test)
    split_idx = n - test_size
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--target", default="pm2_5")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    city_slug = slug_of(args.city)

    df = pd.read_csv(args.input_file, parse_dates=["datetime"]).sort_values("datetime")
    df = build_features(df)

    metrics = {"city": args.city}
    reg_results = {}

    if args.target not in df.columns:
        metrics["regression"] = {}
        (outdir / f"supervised_metrics_{city_slug}.json").write_text(json.dumps(metrics, indent=2))
        _save_metrics(metrics, args.city)
        return

    feats = safe_feature_list(df, args.target)
    reg_df = df.dropna(subset=[args.target]).copy()
    y = pd.to_numeric(reg_df[args.target], errors="coerce")
    reg_df = reg_df.loc[y.notna()]
    y = y.loc[y.notna()]

    X_raw = numeric_only(reg_df, feats)
    feats = list(X_raw.columns)
    X_imp = fill_and_prune(X_raw)
    y = y.loc[X_imp.index]

    X_train, X_test, y_train, y_test = split_time_ordered(X_imp, y, min_test=1)
    preds_store = {}

    if len(X_train) and len(X_test):
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        ypr = rf.predict(X_test)
        reg_results["RandomForestRegressor"] = {
            "MAE": float(mean_absolute_error(y_test, ypr)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, ypr))),
            "R2": float(r2_score(y_test, ypr)),
        }
        preds_store["rf"] = (y_test, ypr)
        pd.DataFrame(
            {"feature": list(X_train.columns), "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False).to_csv(
            outdir / f"feature_importance_rf_{city_slug}.csv", index=False
        )
        joblib.dump(
            {"model": rf, "features": list(X_train.columns)},
            outdir / f"rf_regressor_{city_slug}.pkl",
        )

        try:
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(X_train, y_train)
            ypg = gbr.predict(X_test)
            reg_results["GradientBoostingRegressor"] = {
                "MAE": float(mean_absolute_error(y_test, ypg)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, ypg))),
                "R2": float(r2_score(y_test, ypg)),
            }
            preds_store["gbr"] = (y_test, ypg)
            pd.DataFrame(
                {"feature": list(X_train.columns), "importance": gbr.feature_importances_}
            ).sort_values("importance", ascending=False).to_csv(
                outdir / f"feature_importance_gbr_{city_slug}.csv", index=False
            )
            joblib.dump(
                {"model": gbr, "features": list(X_train.columns)},
                outdir / f"gbr_regressor_{city_slug}.pkl",
            )
        except ValueError:
            hgb = HistGradientBoostingRegressor(random_state=42)
            hgb.fit(X_train, y_train)
            yph = hgb.predict(X_test)
            reg_results["HistGradientBoostingRegressor"] = {
                "MAE": float(mean_absolute_error(y_test, yph)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, yph))),
                "R2": float(r2_score(y_test, yph)),
            }
            preds_store["gbr"] = (y_test, yph)
            joblib.dump(
                {"model": hgb, "features": list(X_train.columns)},
                outdir / f"hgb_regressor_{city_slug}.pkl",
            )

        best_key = min(reg_results, key=lambda k: reg_results[k]["RMSE"])
        y_true, y_pred = preds_store["rf" if best_key.startswith("RandomForest") else "gbr"]
        pred_df = reg_df.loc[y_true.index, ["datetime"]].copy()
        pred_df["pm2_5_true"] = y_true.values
        pred_df["pm2_5_pred"] = y_pred
        pred_df.to_csv(Path("data/processed") / f"{city_slug}_predictions.csv", index=False)

    metrics["regression"] = reg_results

    cls_results = {}
    if "pm2_5" in df.columns and len(df) >= 5:
        cls_df = df.dropna(subset=["pm2_5"]).copy()
        cls_df["aqi_cat"] = cls_df["pm2_5"].apply(aqi_category)
        cls_feats_raw = [c for c in list(X_imp.columns) if c != "pm2_5"]
        Xc_raw = numeric_only(cls_df, cls_feats_raw)
        Xc_imp = fill_and_prune(Xc_raw)
        yc = cls_df["aqi_cat"].loc[Xc_imp.index]

        Xtr, Xte, ytr, yte = split_time_ordered(Xc_imp, yc, min_test=1)

        if len(Xtr) and len(Xte) and len(set(ytr)) >= 2:
            lr = LogisticRegression(max_iter=200)
            lr.fit(Xtr, ytr)
            yhat = lr.predict(Xte)
            cls_results["LogisticRegression"] = {
                "accuracy": float(accuracy_score(yte, yhat)),
                "f1_macro": float(f1_score(yte, yhat, average="macro")),
            }
            joblib.dump(
                {"model": lr, "features": list(Xtr.columns)},
                outdir / f"lr_classifier_{city_slug}.pkl",
            )
            cm = confusion_matrix(
                yte, yhat, labels=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"]
            )
            pd.DataFrame(
                cm,
                index=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"],
                columns=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"],
            ).to_csv(outdir / f"confusion_matrix_{city_slug}.csv")

            rfc = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            rfc.fit(Xtr, ytr)
            yhat = rfc.predict(Xte)
            cls_results["RandomForestClassifier"] = {
                "accuracy": float(accuracy_score(yte, yhat)),
                "f1_macro": float(f1_score(yte, yhat, average="macro")),
            }
            joblib.dump(
                {"model": rfc, "features": list(Xtr.columns)},
                outdir / f"rfc_classifier_{city_slug}.pkl",
            )
            cm = confusion_matrix(
                yte, yhat, labels=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"]
            )
            pd.DataFrame(
                cm,
                index=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"],
                columns=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"],
            ).to_csv(outdir / f"confusion_matrix_{city_slug}.csv")

    if cls_results:
        metrics["classification"] = cls_results

    (outdir / f"supervised_metrics_{city_slug}.json").write_text(json.dumps(metrics, indent=2))
    _save_metrics(metrics, args.city)


if __name__ == "__main__":
    main()
