"""Vercel entrypoint: exposes the FastAPI app defined in src/api.py."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from api import app  # noqa: E402,F401
