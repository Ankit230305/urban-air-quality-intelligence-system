from __future__ import annotations
import os
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

def get_config():
    """Load config.
    If a .env exists in the *current working dir*, it overrides os.environ
    (matches unit test expectation). Otherwise, keep existing environment.
    """
    dotenv_path = Path.cwd() / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)
    # else: do not override; use whatever is in the environment
    return SimpleNamespace(
        OPENWEATHERMAP_API_KEY=os.getenv('OPENWEATHERMAP_API_KEY'),
        WAQI_TOKEN=os.getenv('WAQI_TOKEN'),
        VISUALCROSSING_API_KEY=os.getenv('VISUALCROSSING_API_KEY'),
    )
