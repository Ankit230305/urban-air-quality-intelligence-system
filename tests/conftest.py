import sys
from pathlib import Path

# Make '<repo>/' importable so 'import src....' works
ROOT = Path(__file__).resolve().parents[1]
p = str(ROOT)
if p not in sys.path:
    sys.path.insert(0, p)
