"""Detect anomalies in air quality time series.

This script identifies abnormal pollution events (spikes) using a
z‑score method.  It reads a processed data file containing PM2.5
concentrations and flags observations that deviate significantly from
the rolling mean.  The results are written to a new CSV with an
`is_anomaly` boolean column.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Detect anomalies in PM2.5 time series using z-score."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="CSV file with datetime and PM2.5 data.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output CSV with anomaly flags.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Rolling window size (in samples) for computing the mean and std.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Z-score threshold beyond which a point is considered an anomaly.",
    )
    return parser.parse_args(args)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    input_file = args.input_file
    output_file = args.output_file
    window = args.window
    thresh = args.threshold

    df = pd.read_csv(input_file)
    if "pm2_5" in df.columns:
        col = "pm2_5"
    elif "pm25" in df.columns:
        col = "pm25"
    else:
        raise ValueError("Input must contain a 'pm2_5' or 'pm25' column")

    # Ensure dataframe is sorted by datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Compute rolling mean and std
    rolling_mean = df[col].rolling(window=window, min_periods=1, center=True).mean()
    rolling_std = df[col].rolling(window=window, min_periods=1, center=True).std()
    z_scores = (df[col] - rolling_mean) / rolling_std
    df["z_score"] = z_scores
    df["is_anomaly"] = z_scores.abs() > thresh

    # Save results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved anomalies to {output_file} (found {df['is_anomaly'].sum()} anomalies)")


if __name__ == "__main__":
    main()
