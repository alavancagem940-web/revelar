#!/usr/bin/env python3
"""Revelar — aprimoramento automático de fotos e vídeos (Python + CustomTkinter)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":1"


def main() -> None:
    from ui import run_app

    run_app()


if __name__ == "__main__":
    main()
