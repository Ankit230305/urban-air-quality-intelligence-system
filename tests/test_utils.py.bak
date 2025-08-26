"""Unit tests for utility functions."""
import pytest

from src.utils.config import get_config
from src.features.feature_engineering import compute_aqi_pm25, aqi_category

defdef test_load_config(tmp_path, monkeypatch):
    """Ensure that get_config picks up variables from a .env file."""
    # Create a temporary .env file
    env_path = tmp_path / ".env"
    env_path.write_text("OPENWEATHERMAP_API_KEY=abc123\n")
    # Change working directory to temp path
    monkeypatch.chdir(tmp_path)
    cfg = get_config()
    assert cfg.OPENWEATHERMAP_API_KEY == "abc123"

@pytest.mark.parametrize(
    "value,expected_aqi",
    [
        (5, 20),
        (25, 74),
        (40, 128),
        (100, 174),
        (200, 265),
    ],
)

defdef test_compute_aqi_pm25(value, expected_aqi):
    aqi = compute_aqi_pm25(value)
    assert isinstance(aqi, float)
    # Check that the computed AQI is roughly equal to expected (±5)
    assert abs(aqi - expected_aqi) < 5

defdef test_aqi_category():
    assert aqi_category(25) == "Good"
    assert aqi_category(75) == "Moderate"
    assert aqi_category(125) == "Unhealthy for Sensitive Groups"
    assert aqi_category(175) == "Unhealthy"
    assert aqi_category(250) == "Very Unhealthy"
    assert aqi_category(350) == "Hazardous"
