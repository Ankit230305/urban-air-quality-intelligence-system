import sys
from pathlib import Path

# Add <repo>/src to sys.path so "from src..." imports work in CI and locally
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    p = str(SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
