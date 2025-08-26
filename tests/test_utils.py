import pytest

from src.utils.config import get_config

# If your project exposes these, keep the imports below.
# Otherwise, comment them out or adjust to your module names.
try:
    from src.utils.aqi import compute_aqi_pm25, aqi_category  # type: ignore
except Exception:  # pragma: no cover
    compute_aqi_pm25 = None  # fallback for CI if module missing
    aqi_category = None


def test_load_config(tmp_path, monkeypatch):
    """get_config should pick values from a .env file in CWD."""
    env_path = tmp_path / ".env"
    env_path.write_text("OPENWEATHERMAP_API_KEY=abc123\n")
    monkeypatch.chdir(tmp_path)
    cfg = get_config()
    assert getattr(cfg, "OPENWEATHERMAP_API_KEY", None) == "abc123"


@pytest.mark.parametrize(
    "value,expected_floor,expected_ceiling",
    [
        (5, 0, 50),
        (25, 0, 100),
        (40, 50, 150),
        (100, 100, 200),
        (200, 150, 300),
    ],
)
def test_compute_aqi_pm25_range(value, expected_floor, expected_ceiling):
    if compute_aqi_pm25 is None:
        pytest.skip("compute_aqi_pm25 not available in this build")
    aqi = float(compute_aqi_pm25(value))
    assert expected_floor <= aqi <= expected_ceiling


def test_aqi_category_returns_string():
    if aqi_category is None:
        pytest.skip("aqi_category not available in this build")
    assert isinstance(aqi_category(25), str)
    assert isinstance(aqi_category(250), str)
