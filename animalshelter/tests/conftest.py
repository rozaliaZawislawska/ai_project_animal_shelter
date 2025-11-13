import sys
from pathlib import Path

# Ensure `src` is on sys.path so tests can import the package without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))