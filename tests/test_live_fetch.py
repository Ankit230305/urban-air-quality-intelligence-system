from types import SimpleNamespace
import pandas as pd
import pytest


def test_fetch_live_point_parses(monkeypatch):
    # mock config
    mod = __import__("src.utils.live_fetch", fromlist=["_owm_get", "fetch_live_point", "livepoint_to_df"])
    lf = mod

    def fake_get(url, lat, lon):
        if "air_pollution" in url:
            return {
                "list": [
                    {"main": {"aqi": 3},
                     "components": {"pm2_5": 23.0, "pm10": 48.0,
                                    "no2": 10.0, "o3": 40.0, "so2": 5.0, "co": 300.0}}
                ]
            }
        else:
            return {"main": {"temp": 300.0, "humidity": 50},
                    "wind": {"speed": 2.0},
                    "rain": {"1h": 0.0}}

    monkeypatch.setattr(lf, "_owm_get", fake_get)

    p = lf.fetch_live_point("TestCity", 12.3, 45.6)
    df = lf.livepoint_to_df(p)
    assert df.loc[0, "city"] == "TestCity"
    assert df.loc[0, "pm2_5"] == 23.0
    assert df.loc[0, "aqi"] == 3


def test_missing_key_raises(monkeypatch):
    # simulate config without key by monkeypatching _owm_get to raise
    mod = __import__("src.utils.live_fetch", fromlist=["_owm_get", "fetch_live_point"])
    lf = mod

    def raise_key(*args, **kwargs):
        raise RuntimeError("Missing OpenWeatherMap API key.")

    monkeypatch.setattr(lf, "_owm_get", raise_key)
    with pytest.raises(RuntimeError):
        lf.fetch_live_point("City", 0.0, 0.0)
