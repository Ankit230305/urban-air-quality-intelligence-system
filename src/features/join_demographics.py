from __future__ import annotations
from pathlib import Path
import pandas as pd

DEMOS = Path("data/external/city_demographics.csv")

def _slug(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

def run(input_file: str, city: str, output_file: str) -> None:
    """Join demographics by city slug; fill safe defaults if file/rows missing."""
    df = pd.read_csv(input_file)
    slug = _slug(city)
    # load demos
    if DEMOS.exists():
        dem = pd.read_csv(DEMOS)
        # allow both 'city' and 'slug' columns
        if "slug" not in dem.columns:
            dem["slug"] = dem.get("city", "").astype(str).str.lower().str.replace(" ", "_", regex=False)
        dem = dem.drop_duplicates("slug")
        demo_row = dem[dem["slug"] == slug]
        if demo_row.empty:
            demo_row = pd.DataFrame([{
                "slug": slug, "_category": "unknown", "population": 0,
                "pop_density_per_km2": 0.0, "pct_elderly": 0.0, "pct_children": 0.0,
                "respiratory_illness_rate_per_100k": 0.0
            }])
    else:
        demo_row = pd.DataFrame([{
            "slug": slug, "_category": "unknown", "population": 0,
            "pop_density_per_km2": 0.0, "pct_elderly": 0.0, "pct_children": 0.0,
            "respiratory_illness_rate_per_100k": 0.0
        }])

    df["slug"] = slug
    out = pd.merge(df, demo_row, on="slug", how="left", suffixes=("", ""))
    out.to_csv(output_file, index=False)
    print(f"✅ demographics joined → {output_file} (rows={len(out)})")
