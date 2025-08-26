#!/usr/bin/env bash
set -euo pipefail
CITY_LATLON=("Delhi 28.6139 77.2090"
             "Mumbai 19.0760 72.8777"
             "Bengaluru 12.9716 77.5946"
             "Hyderabad 17.3850 78.4867"
             "Chennai 13.0827 80.2707"
             "Kolkata 22.5726 88.3639"
             "Vizag 17.6868 83.2185"
             "Vellore 12.9165 79.1325")
START="${1:-2024-08-17}"
END="${2:-2024-08-24}"
for row in "${CITY_LATLON[@]}"; do
  set -- $row
  CITY="$1"; LAT="$2"; LON="$3"
  echo "=== Building $CITY ==="
  bash bin/run_city_pipeline.sh "$CITY" "$LAT" "$LON" "$START" "$END"
done
