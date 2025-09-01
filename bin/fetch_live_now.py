#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone, timezone

# Make 'src' imports work when called from anywhere
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.live_fetch import fetch_live_bundle, slugify  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch live air & weather from providers.")
    p.add_argument("--city", required=True, help="City name (e.g. Mumbai)")
    p.add_argument("--lat", required=True, type=float)
    p.add_argument("--lon", required=True, type=float)
    p.add_argument("--outdir", default="data/live", help="Output folder (default: data/live)")
    args = p.parse_args()

    df, raw = fetch_live_bundle(args.city, args.lat, args.lon)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = slugify(args.city)
    csv_path = outdir / f"{slug}_live_{ts}.csv"
    json_path = outdir / f"{slug}_live_{ts}.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(raw, indent=2))

    print(f"✅ wrote {csv_path}  ({len(df)} rows)")
    print(f"✅ wrote {json_path}")


if __name__ == "__main__":
    main()
