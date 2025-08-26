"""Configuration management for the Urban Air Quality Intelligence System.

This module centralises all configuration handling.  It loads environment
variables from a `.env` file using python‑dotenv and provides a simple
interface for accessing API keys and other parameters throughout the
project.  Using a single source for configuration makes it easy to
customise your deployment without changing code.

Usage:

    from src.utils.config import get_config

    config = get_config()
    api_key = config.OPENWEATHERMAP_API_KEY
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Dataclass for storing configuration values.

    Fields are typed as Optional[str] because not all keys are required at
    runtime.  If a key is missing it will be `None`, and the calling
    function should handle that case appropriately.  Environment
    variables are loaded at instantiation time.
    """

    OPENWEATHERMAP_API_KEY: Optional[str] = None
    OPENAQ_API_KEY: Optional[str] = None
    PURPLEAIR_API_KEY: Optional[str] = None
    WAQI_API_KEY: Optional[str] = None
    VISUAL_CROSSING_API_KEY: Optional[str] = None
    CPCB_API_KEY: Optional[str] = None
    GOOGLE_MAPS_API_KEY: Optional[str] = None

    def __post_init__(self) -> None:
        # Load variables from the environment after dataclass initialisation
        self.OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
        self.OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")
        self.PURPLEAIR_API_KEY = os.environ.get("PURPLEAIR_API_KEY")
        self.WAQI_API_KEY = os.environ.get("WAQI_API_KEY")
        self.VISUAL_CROSSING_API_KEY = os.environ.get("VISUAL_CROSSING_API_KEY")
        self.CPCB_API_KEY = os.environ.get("CPCB_API_KEY")
        self.GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")


_CONFIG_CACHE: Optional[Config] = None


def get_config() -> Config:
    """Load the configuration from the environment.

    On the first call this function loads variables from `.env` in the
    project root (if present) and stores a single instance of the
    `Config` dataclass.  Subsequent calls return the cached instance.

    Returns
    -------
    Config
        A dataclass containing all environment variables as attributes.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        # Load environment variables from .env if it exists.  `load_dotenv`
        # silently does nothing if the file is absent.
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
        _CONFIG_CACHE = Config()
    return _CONFIG_CACHE
