"""Tests for the feature engineering module."""

import pandas as pd

from src.features.feature_engineering import merge_and_feature_engineer


def test_merge_and_feature_engineer():
    # Create synthetic pollutant and weather data
    pollution_data = pd.DataFrame(
        {
            "datetime": ["2024-08-24 00:00:00", "2024-08-24 01:00:00"],
            "pm2_5": [10.0, 20.0],
            "pm10": [20.0, 40.0],
            "no2": [5.0, 6.0],
        }
    )
    weather_data = pd.DataFrame(
        {
            "datetime": ["2024-08-24 00:00:00", "2024-08-24 01:00:00"],
            "temp": [25.0, 24.5],
            "humidity": [60, 65],
            "wind_speed": [5.0, 4.5],
        }
    )
    result = merge_and_feature_engineer(pollution_data, weather_data, temporal_resolution="H")
    # Expect 2 rows
    assert len(result) == 2
    # Check that AQI and category exist
    assert "aqi" in result.columns
    assert "aqi_category" in result.columns
    # Check that categories match expected
    expected_categories = ["Good", "Good"]
    assert result["aqi_category"].tolist() == expected_categories
