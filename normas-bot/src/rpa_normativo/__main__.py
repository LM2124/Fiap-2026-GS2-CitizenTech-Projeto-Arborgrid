"""Permite execucao via python -m rpa_normativo a partir de normas-bot/src."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
