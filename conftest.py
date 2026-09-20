"""Pytest bootstrap: make the project-root packages (app, rag) importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
