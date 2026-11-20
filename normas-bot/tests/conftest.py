"""Configuracao de caminho de importacao para os testes do pacote rpa_normativo."""

from __future__ import annotations

import sys
from pathlib import Path

RPA_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = RPA_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
