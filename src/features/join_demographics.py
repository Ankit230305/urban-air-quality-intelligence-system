import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features", required=True, help="processed features CSV with a 'datetime' column"
    )
    ap.add_argument("--demographics", default="data/external/demographics_india.csv")
    ap.add_argument("--city", required=True, help="City name to match")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    f = pd.read_csv(args.features, parse_dates=["datetime"])
    d = pd.read_csv(args.demographics)

    # match by 'city' first, fallback to district contains
    row = None
    if "city" in d.columns:
        m = d["city"].str.strip().str.lower() == args.city.strip().lower()
        if m.any():
            row = d[m].iloc[0]
    if row is None and "district" in d.columns:
        # crude fallback (works when city == district name)
        m = d["district"].str.strip().str.lower().str.contains(args.city.strip().lower())
        if m.any():
            row = d[m].iloc[0]

    if row is None:
        print(
            f"[WARN] No demographics match for city='{args.city}' in {args.demographics}. Copying features unchanged."
        )
        f.to_csv(args.output, index=False)
        return

    # broadcast demographics across all rows
    for c in [
        "population",
        "pop_density_per_km2",
        "pct_elderly",
        "pct_children",
        "respiratory_illness_rate_per_100k",
    ]:
        if c in d.columns:
            f[c] = row.get(c)

    f.to_csv(args.output, index=False)
    print(f"✅ wrote {args.output} with demographics columns")


if __name__ == "__main__":
    main()
