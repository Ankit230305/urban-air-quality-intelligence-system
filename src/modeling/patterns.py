# Pattern discovery: daily seasonality, clustering, and simple association rules
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _slugify(city: str) -> str:
    return city.lower().replace(" ", "_")


def _coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime is parsed and tz-naive (UTC-normalized then strip tz)."""
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        # If tz-aware, normalize to UTC then drop tz to keep joiners happy
        try:
            if getattr(df["datetime"].dt, "tz", None) is not None:
                df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try:
                df["datetime"] = df["datetime"].dt.tz_localize(None)
            except Exception:
                pass
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric for pollutants/weather & AQI, non-numeric -> NaN."""
    for c in [
        "pm2_5",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
        "temp",
        "humidity",
        "wind_speed",
        "precip",
        "aqi",
    ]:
        if c in df.columns:
            # try soft first
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # if still object, hard-coerce
            if df[c].dtype == "object":
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily means of all numeric columns."""
    if df.empty or "datetime" not in df.columns:
        return pd.DataFrame()
    df = _coerce_datetime(df).sort_values("datetime").set_index("datetime")
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    daily = num.resample("D").mean(numeric_only=True)
    daily = daily.dropna(how="all").reset_index()
    return daily


def run_clustering(daily: pd.DataFrame, clusters: int = 3) -> pd.DataFrame:
    """
    KMeans on available pollutant+weather columns; labels only for rows with all features present.
    """
    from sklearn.cluster import KMeans

    if daily.empty:
        return daily.assign(cluster=np.nan)

    feat_cols = [
        c
        for c in [
            "pm2_5",
            "pm10",
            "no2",
            "o3",
            "so2",
            "co",
            "temp",
            "humidity",
            "wind_speed",
            "precip",
            "aqi",
        ]
        if c in daily.columns
    ]
    if not feat_cols:
        return daily.assign(cluster=np.nan)

    X = daily[feat_cols].copy()
    mask = X.notna().all(axis=1)
    X_use = X[mask]
    if len(X_use) < 2:
        out = daily.copy()
        out["cluster"] = np.nan
        return out

    k = max(1, min(clusters, len(X_use)))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_use.to_numpy())

    out = daily.copy()
    out["cluster"] = np.nan
    out.loc[mask, "cluster"] = labels
    return out


def mine_associations(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Association mining between discretized weather/pollutants:
    - Discretize into terciles (low/med/high) using qcut (duplicates drop)
    - One-hot encode ternary items; run apriori + association_rules
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except Exception:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    if daily.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    cols = [
        c
        for c in [
            "pm2_5",
            "pm10",
            "no2",
            "o3",
            "so2",
            "co",
            "temp",
            "humidity",
            "wind_speed",
            "precip",
        ]
        if c in daily.columns
    ]
    if not cols:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    df = daily[cols].dropna()
    if df.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    # discretize to terciles and one-hot encode as boolean “items”
    items = {}
    for c in df.columns:
        try:
            q = pd.qcut(df[c], 3, labels=["low", "med", "high"], duplicates="drop")
        except Exception:
            q = pd.Series(["med"] * len(df), index=df.index)
        for lev in pd.unique(q):
            if pd.isna(lev):
                continue
            items[f"{c}={lev}"] = q == lev

    trans = pd.DataFrame(items).fillna(False)
    if trans.sum().sum() == 0:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    freq = apriori(trans, min_support=0.2, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    out = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    out["consequents"] = out["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    out.sort_values(["lift", "confidence", "support"], ascending=False, inplace=True)
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description="Pattern discovery: daily seasonality, clustering, association rules"
    )
    ap.add_argument(
        "--input-file",
        required=True,
        help="Processed features CSV with datetime + pollutants + weather",
    )
    ap.add_argument("--outdir", default="reports", help="Directory to write outputs")
    ap.add_argument("--city", default="City", help="City name (for filenames)")
    ap.add_argument(
        "--clusters", type=int, default=3, help="KMeans cluster count (capped to valid rows)"
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(args.city)

    df = pd.read_csv(args.input_file, parse_dates=["datetime"])
    df = _coerce_numeric(df)

    daily = build_daily(df)
    daily = run_clustering(daily, clusters=args.clusters)

    daily_path = outdir / f"seasonal_{slug}.csv"
    daily.to_csv(daily_path, index=False)

    rules = mine_associations(daily)
    rules_path = outdir / f"assoc_rules_{slug}.csv"
    rules.to_csv(rules_path, index=False)

    md_path = outdir / f"patterns_{slug}.md"
    with md_path.open("w") as f:
        f.write(f"# Pattern discovery — {args.city}\n\n")
        f.write(f"- Daily rows: **{len(daily)}**\n")
        if "cluster" in daily.columns and daily["cluster"].notna().any():
            f.write(
                f"- Clusters present: **{int(pd.Series(daily['cluster'].dropna()).nunique())}**\n"
            )
        else:
            f.write("- Clusters present: *(not enough complete rows to compute)*\n")
        f.write(f"- Association rules: **{len(rules)}**\n")
        if not rules.empty:
            f.write("\nTop 5 rules by lift:\n\n")
            for _, r in rules.head(5).iterrows():
                f.write(
                    f"- **{r['antecedents']} ⇒ {r['consequents']}** "
                    f"(support {r['support']:.2f}, conf {r['confidence']:.2f}, lift {r['lift']:.2f})\n"
                )

    print(f"✅ Saved: {daily_path}, {rules_path}, {md_path}")


if __name__ == "__main__":
    main()
