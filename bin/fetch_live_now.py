#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.utils.live_fetch import fetch_live_point, livepoint_to_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a live snapshot now.")
    ap.add_argument("--city", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument(
        "--out",
        default=None,
        help="Optional CSV to append to (default: data/processed/<city>__live.csv)",
    )
    args = ap.parse_args()

    out = (
        Path(args.out)
        if args.out
        else Path("data/processed") / f"{args.city.lower().replace(' ', '_')}__live.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        lp = fetch_live_point(args.city, args.lat, args.lon)
        df = livepoint_to_df(lp)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return

    if out.exists():
        df0 = pd.read_csv(out)
        df = pd.concat([df0, df], ignore_index=True)

    df.to_csv(out, index=False)
    print(f"✅ Live snapshot saved to {out} (rows={len(df)})")


if __name__ == "__main__":
    main()
