"""Modelos de dados, constantes de regras de negocio e tipos de excecao do RPA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATUS_FONTE_FECHADA = "FONTE_FECHADA"
STATUS_FONTE_INTERNA = "FONTE_INTERNA"
STATUS_SEM_ALTERACAO = "SEM_ALTERACAO"
STATUS_CONFIRMADA = "CONFIRMADA"
STATUS_REVISAO_HUMANA = "REVISAO_HUMANA"
STATUS_ERRO_REDE = "ERRO_REDE"
STATUS_ERRO_HTTP = "ERRO_HTTP"
STATUS_ERRO_CONTEUDO = "ERRO_CONTEUDO"
STATUS_NAO_VERIFICADO = "NAO_VERIFICADO"

EXCECAO_NEGOCIO = "negocio"
EXCECAO_SISTEMA = "sistema"

STATUS_SUCESSO_FONTE = frozenset(
    {
        STATUS_FONTE_FECHADA,
        STATUS_FONTE_INTERNA,
        STATUS_SEM_ALTERACAO,
        STATUS_CONFIRMADA,
    }
)

STATUS_ERRO_SISTEMA = frozenset(
    {
        STATUS_ERRO_REDE,
        STATUS_ERRO_HTTP,
        STATUS_ERRO_CONTEUDO,
    }
)


@dataclass(frozen=True)
class TrechoConfig:
    """Configuracao de um trecho da base vinculado a uma fonte."""

    chunk_id: str
    fatos: tuple[str, ...]


@dataclass(frozen=True)
class FonteConfig:
    """Configuracao de uma fonte monitorada em fontes.yaml."""

    id: str
    nome: str
    tipo: str
    url: str | None = None
    url_verificada: bool = False
    motivo_sem_verificacao: str | None = None
    trechos: tuple[TrechoConfig, ...] = field(default_factory=tuple)


@dataclass
class ResultadoDownload:
    """Resultado bruto da etapa de obtencao de uma fonte (HTTP ou amostra offline)."""

    ok: bool
    conteudo: bytes = b""
    status_http: int | None = None
    tentativas: int = 0
    duracao_ms: int = 0
    codigo_erro: str | None = None
    mensagem_erro: str | None = None


@dataclass
class ResultadoFonte:
    """Resultado consolidado da avaliacao de uma fonte apos aplicar R1 a R7."""

    fonte_id: str
    nome: str
    tipo: str
    url: str | None
    url_verificada: bool
    status: str
    motivo: str
    tentativas: int = 0
    bytes_baixados: int = 0
    duracao_ms: int = 0
    sha256: str | None = None
    excecao_tipo: str | None = None
    trechos_ids: list[str] = field(default_factory=list)

    def para_dict(self) -> dict[str, Any]:
        """Converte o resultado da fonte para dicionario serializavel em JSON."""
        return {
            "fonte_id": self.fonte_id,
            "nome": self.nome,
            "tipo": self.tipo,
            "url": self.url,
            "url_verificada": self.url_verificada,
            "status": self.status,
            "motivo": self.motivo,
            "tentativas": self.tentativas,
            "bytes": self.bytes_baixados,
            "duracao_ms": self.duracao_ms,
            "sha256": self.sha256,
            "excecao_tipo": self.excecao_tipo,
            "trechos_ids": list(self.trechos_ids),
        }


@dataclass
class ResultadoExecucao:
    """Resumo completo de uma execucao do robo RPA."""

    execucao_id: str
    codigo_saida: int
    relatorio: dict[str, Any]
    arquivo_base_sincronizada: str
    arquivo_relatorio: str
    arquivo_estado: str
    arquivo_log: str
