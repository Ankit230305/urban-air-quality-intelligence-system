import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

def seasonal_trends(df: pd.DataFrame) -> pd.DataFrame:
    g = df.set_index("datetime").copy()
    out = []
    # daily means
    d = g.resample("D").mean(numeric_only=True).reset_index()
    d["dow"] = d["datetime"].dt.dayofweek
    d["month"] = d["datetime"].dt.month
    return d

def do_clustering(daily: pd.DataFrame, features, k=3):
    X = daily[features].dropna().copy()
    if X.empty:
        return None, None, None
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(Z)
    sil = silhouette_score(Z, labels) if len(set(labels))>1 else np.nan
    return labels, sil, km

def discretize_for_rules(df: pd.DataFrame, cols_bins: dict):
    disc = {}
    for col, bins in cols_bins.items():
        if col not in df: continue
        disc[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    return pd.DataFrame(disc, index=df.index)

def mine_rules(df: pd.DataFrame):
    # example: rules between weather and pm2_5 levels
    use = df[["pm2_5","temp","humidity","wind_speed","precip"]].dropna()
    if use.empty:
        return pd.DataFrame()
    # discretize to 3 bins each
    binned = use.apply(lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
    # one-hot encode bins
    ohe = pd.get_dummies(binned.astype(int), prefix=binned.columns, columns=binned.columns)
    freq = apriori(ohe, min_support=0.1, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    rules = rules.sort_values("lift", ascending=False)
    return rules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--clusters", type=int, default=3)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file, parse_dates=["datetime"]).sort_values("datetime")
    daily = seasonal_trends(df)

    # clustering on daily pollutant+weather means
    feat = [c for c in ["pm2_5","pm10","no2","o3","so2","co","temp","humidity","wind_speed","precip"] if c in daily]
    labels, sil, km = do_clustering(daily, feat, k=args.clusters)
    if labels is not None:
        daily["cluster"] = labels

    # association rules
    rules = mine_rules(df)

    # save artifacts
    daily_path = Path(args.outdir) / f"seasonal_{args.city.lower().replace(' ','_')}.csv"
    daily.to_csv(daily_path, index=False)

    rules_path = Path(args.outdir) / f"assoc_rules_{args.city.lower().replace(' ','_')}.csv"
    rules.to_csv(rules_path, index=False)

    # short report
    report_path = Path(args.outdir) / f"patterns_{args.city.lower().replace(' ','_')}.md"
    with open(report_path, "w") as f:
        f.write(f"# Pattern Discovery – {args.city}\n\n")
        f.write(f"- Features used for clustering: {feat}\n")
        if labels is not None:
            f.write(f"- KMeans(k={args.clusters}) silhouette: {sil:.3f}\n")
            cts = daily["cluster"].value_counts().to_dict()
            f.write(f"- Cluster sizes: {cts}\n")
        else:
            f.write("- Clustering skipped (insufficient data).\n")
        f.write(f"- Association rules saved: {rules_path.name} (rows={len(rules)})\n")

    print(f"✅ Saved: {daily_path}, {rules_path}, {report_path}")

if __name__ == "__main__":
    main()
