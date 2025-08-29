#!/usr/bin/env python3

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse
from pathlib import Path

import pandas as pd

from src.utils.live_fetch import fetch_live_point, livepoint_to_df

def slug_of(name: str) -> str:
    return name.lower().replace(" ", "_")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    args = p.parse_args()

    lp = fetch_live_point(args.city, args.lat, args.lon)
    df_new = livepoint_to_df(lp)

    out = Path("data/processed") / f"{slug_of(args.city)}__live.csv"
    if out.exists():
        df_old = pd.read_csv(out)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["datetime"], keep="last")
    else:
        df = df_new

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved/updated: {out} (rows={len(df)})")

if __name__ == "__main__":
    main()
