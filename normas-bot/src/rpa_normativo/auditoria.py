"""Configuracao de logging estruturado e trilha de auditoria sem dados pessoais."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class _FormatadorIso8601(logging.Formatter):
    """Formata timestamp em ISO-8601 com fuso horario local (ex.: 2026-11-19T21:30:00-03:00)."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created).astimezone()
        return dt.isoformat(timespec="seconds")


def criar_logger_execucao(logs_dir: Path, agora: datetime | None = None) -> tuple[logging.Logger, Path]:
    """Cria logger dedicado escrevendo em logs/rpa_normativo_AAAAMMDD.log e no console."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ref = agora or datetime.now().astimezone()
    caminho_log = logs_dir / f"rpa_normativo_{ref.strftime('%Y%m%d')}.log"

    nome_logger = f"rpa_normativo.{ref.strftime('%Y%m%d%H%M%S%f')}"
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatador = _FormatadorIso8601("%(asctime)s %(levelname)s %(message)s")

    handler_arquivo = logging.FileHandler(caminho_log, encoding="utf-8")
    handler_arquivo.setFormatter(formatador)
    logger.addHandler(handler_arquivo)

    handler_console = logging.StreamHandler()
    handler_console.setFormatter(formatador)
    logger.addHandler(handler_console)

    return logger, caminho_log


def fechar_logger(logger: logging.Logger) -> None:
    """Encerra e remove handlers do logger para liberar descritores de arquivo."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _formatar_valor(valor: Any) -> str:
    """Formata valores chave=valor evitando quebras de linha e aspas desnecessarias."""
    texto = str(valor).replace("\n", " ").replace("\r", " ").strip()
    if " " in texto or "=" in texto or texto == "":
        texto_limpo = texto.replace('"', "'")
        return f'"{texto_limpo}"'
    return texto_limpo if (texto_limpo := texto) else '""'


def registrar_evento(
    logger: logging.Logger,
    exec_id: str,
    fonte: str,
    etapa: str,
    status: str,
    nivel: int = logging.INFO,
    **campos: Any,
) -> None:
    """Registra uma linha estruturada de auditoria com campos padrao."""
    partes = [
        f"exec={exec_id}",
        f"fonte={fonte}",
        f"etapa={etapa}",
        f"status={status}",
    ]
    for chave, valor in campos.items():
        if valor is not None:
            partes.append(f"{chave}={_formatar_valor(valor)}")
    logger.log(nivel, " ".join(partes))
