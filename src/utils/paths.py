from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clean import coerce_none_like


def resolve_processed(city: str):
    """Return a DataFrame using the best available processed file for a city."""
    slug = city.lower().replace(" ", "_")
    cands = [
        Path("data/processed") / f"{slug}_features_plus_demo.csv",
        Path("data/processed") / f"{slug}__features.csv",
        Path("data/processed") / f"{slug}_features.csv",
    ]
    for p in cands:
        if p.exists():
            try:
                df = pd.read_csv(p, parse_dates=["datetime"])
                return coerce_none_like(df)
            except Exception:
                pass
    return None

def resolve_forecast_path(slug: str) -> Path:
    """Prefer city-specific forecast; fallback to generic."""
    p_city = Path("models") / f"forecast_pm25_{slug}.csv"
    p_gen = Path("models") / "forecast_pm25.csv"
    return p_city if p_city.exists() else p_gen
