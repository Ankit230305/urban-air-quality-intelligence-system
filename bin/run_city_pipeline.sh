#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 5 ]; then
  echo "Usage: $0 \"City Name\" LAT LON YYYY-MM-DD YYYY-MM-DD"
  exit 1
fi

CITY="$1"; LAT="$2"; LON="$3"; START="$4"; END="$5"
SLUG=$(echo "$CITY" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')
export PYTHONPATH="${PYTHONPATH:-$PWD}"

python -m src.data.collect_air_quality --city "$CITY" --lat "$LAT" --lon "$LON" --start "$START" --end "$END"

python - <<'PY' "$CITY" "$SLUG" "$START" "$END"
import sys, pandas as pd
from pathlib import Path
from src.features.feature_engineering import merge_and_feature_engineer

city, slug, start, end = sys.argv[1:5]
raw = Path("data/raw")
proc = Path("data/processed"); proc.mkdir(parents=True, exist_ok=True)

poll_file = sorted(raw.glob(f"{city}_openweathermap_{start}_{end}.csv"))[-1]
wx_file   = sorted(raw.glob(f"{city}_visualcrossing_{start}_{end}.csv"))[-1]

p = pd.read_csv(poll_file, parse_dates=["datetime"])
w = pd.read_csv(wx_file, parse_dates=["datetime"])

df = merge_and_feature_engineer(p, w).sort_values("datetime")
out = proc / f"{slug}_features.csv"
df.to_csv(out, index=False)
print(f"WROTE {out} {df.shape}")
PY

python -m src.features.join_demographics \
  --features "data/processed/${SLUG}_features.csv" \
  --demographics "data/external/demographics_india.csv" \
  --city "$CITY" \
  --output "data/processed/${SLUG}_features_plus_demo.csv"

python -m src.modeling.patterns \
  --input-file "data/processed/${SLUG}_features_plus_demo.csv" \
  --city "$CITY" \
  --outdir reports \
  --clusters 3 || true

python -m src.modeling.train_time_series \
  --input-file "data/processed/${SLUG}_features_plus_demo.csv" \
  --output-dir models \
  --target pm2_5 || true

python -m src.modeling.detect_anomalies \
  --input-file "data/processed/${SLUG}_features_plus_demo.csv" \
  --output "data/processed/${SLUG}_anomalies.csv"

python -m src.modeling.health_risk \
  --input-file "data/processed/${SLUG}_features_plus_demo.csv" \
  --demographics "data/external/demographics_india.csv" \
  --city "$CITY" \
  --output "data/processed/${SLUG}_health.csv"

echo "OK: built city=${CITY} slug=${SLUG}"
