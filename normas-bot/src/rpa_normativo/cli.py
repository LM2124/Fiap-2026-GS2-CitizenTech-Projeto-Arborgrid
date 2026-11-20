"""Interface de linha de comando (CLI) do robo RPA normativo ArborGrid SP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .executor import executar_rpa

_RPA_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _RPA_ROOT.parent


def construir_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        prog="rpa_normativo",
        description="Robo RPA de verificacao e sincronizacao da base normativa do ArborGrid SP.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Usa amostras locais em amostras/fontes/ em vez de baixar da internet.",
    )
    parser.add_argument(
        "--simular-mudanca",
        action="append",
        default=[],
        metavar="FONTE_ID",
        help="Simula alteracao de conteudo lendo <id>_alterada.<ext> (demonstra regra R4).",
    )
    parser.add_argument(
        "--simular-falha",
        action="append",
        default=[],
        metavar="FONTE_ID",
        help="Simula timeout repetido de rede na fonte informada (demonstra regra R5).",
    )
    parser.add_argument(
        "--fontes",
        type=Path,
        default=_RPA_ROOT / "fontes.yaml",
        help="Caminho para o arquivo fontes.yaml.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=_REPO_ROOT / "webapp" / "data" / "knowledge_base.jsonl",
        help="Caminho para a base de conhecimento JSONL de origem.",
    )
    parser.add_argument(
        "--amostras-dir",
        type=Path,
        default=_RPA_ROOT / "amostras" / "fontes",
        help="Diretorio com amostras locais para modo --offline.",
    )
    parser.add_argument(
        "--saida-dir",
        type=Path,
        default=_RPA_ROOT / "saida",
        help="Diretorio de gravacao da base sincronizada e relatorio.",
    )
    parser.add_argument(
        "--estado-dir",
        type=Path,
        default=_RPA_ROOT / "estado",
        help="Diretorio de persistencia do estado da ultima execucao.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=_RPA_ROOT / "logs",
        help="Diretorio de gravacao dos arquivos de log.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada CLI retornando codigo de saida 0 ou 2."""
    parser = construir_parser()
    args = parser.parse_args(argv)

    resultado = executar_rpa(
        caminho_fontes=args.fontes,
        caminho_base=args.base,
        amostras_dir=args.amostras_dir,
        saida_dir=args.saida_dir,
        estado_dir=args.estado_dir,
        logs_dir=args.logs_dir,
        offline=args.offline,
        simular_mudanca=set(args.simular_mudanca),
        simular_falha=set(args.simular_falha),
    )
    return resultado.codigo_saida
