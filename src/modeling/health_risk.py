"""Estimate respiratory health risks based on pollutant exposure.

This script computes a simple risk score for respiratory illness based on
pollutant concentrations and returns a categorical risk level (Low,
Moderate, High).  The model implemented here is not medically
validated; it serves as a placeholder until domain‑specific models can
be integrated.  The intent is to demonstrate how health impact scores
could be layered on top of pollution data.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Compute respiratory health risk scores from pollution data."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="CSV file with pollutant concentration columns (pm2_5/pm25, pm10, no2, o3, co, so2)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the health risk CSV",
    )
    return parser.parse_args(args)


def compute_risk_score(row: pd.Series) -> Tuple[float, str]:
    """Compute a risk score and category from pollutant concentrations.

    A simple logistic function is used to transform the linear combination
    of pollutants into a probability between 0 and 1.  Categories are
    derived from the score.
    """
    # Extract concentrations (fallback to 0 if missing)
    pm25 = row.get("pm2_5", row.get("pm25", 0.0)) or 0.0
    pm10 = row.get("pm10", 0.0) or 0.0
    no2 = row.get("no2", 0.0) or 0.0
    o3 = row.get("o3", 0.0) or 0.0
    co = row.get("co", 0.0) or 0.0
    so2 = row.get("so2", 0.0) or 0.0
    # Linear model coefficients (arbitrary but increasing with concentration)
    z = 0.03 * pm25 + 0.02 * pm10 + 0.015 * no2 + 0.01 * o3 + 0.005 * co + 0.01 * so2 - 5
    score = 1 / (1 + np.exp(-z))  # sigmoid transformation
    # Categorise risk
    if score < 0.33:
        category = "Low"
    elif score < 0.66:
        category = "Moderate"
    else:
        category = "High"
    return score, category


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    input_file = args.input_file
    output_file = args.output_file

    df = pd.read_csv(input_file)
    # Compute risk for each row
    scores = []
    categories = []
    for _, row in df.iterrows():
        score, cat = compute_risk_score(row)
        scores.append(score)
        categories.append(cat)
    df["health_risk_score"] = scores
    df["health_risk_category"] = categories
    # Save to file
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved health risk data to {output_file}")


if __name__ == "__main__":
    main()
