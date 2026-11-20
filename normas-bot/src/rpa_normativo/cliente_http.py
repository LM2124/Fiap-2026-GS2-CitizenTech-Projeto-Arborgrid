"""Cliente HTTP resiliente (R5 e R6) e leitor de amostras locais para modo offline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import httpx

from .modelos import (
    STATUS_ERRO_CONTEUDO,
    STATUS_ERRO_HTTP,
    STATUS_ERRO_REDE,
    FonteConfig,
    ResultadoDownload,
)

USER_AGENT = "ArborGrid-RPA-Normativo/1.0 (+FIAP Global Solution 2026 CitizenTech)"
TIMEOUT_SEGUNDOS = 15.0
MAX_TENTATIVAS_REDE = 3
BACKOFF_SEGUNDOS = (1.0, 2.0, 4.0)


class ClienteFontes:
    """Obtencao de conteudo de fontes publicas via HTTP/HTTPS ou amostras locais."""

    def __init__(
        self,
        offline: bool = False,
        amostras_dir: Path | None = None,
        simular_mudanca: set[str] | None = None,
        simular_falha: set[str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        backoff_segundos: tuple[float, ...] = BACKOFF_SEGUNDOS,
        timeout_segundos: float = TIMEOUT_SEGUNDOS,
    ) -> None:
        self.offline = offline
        self.amostras_dir = amostras_dir
        self.simular_mudanca = simular_mudanca or set()
        self.simular_falha = simular_falha or set()
        self.transport = transport
        self.sleep_fn = sleep_fn
        self.backoff_segundos = backoff_segundos
        self.timeout_segundos = timeout_segundos

    def obter(self, fonte: FonteConfig) -> ResultadoDownload:
        """Obtem os bytes da fonte aplicando simulacao, modo offline ou HTTP com retry."""
        if fonte.id in self.simular_falha:
            return self._simular_falha_rede(fonte)

        if self.offline:
            return self._ler_amostra_local(fonte)

        return self._baixar_http(fonte)

    def _simular_falha_rede(self, fonte: FonteConfig) -> ResultadoDownload:
        inicio = time.perf_counter()
        erro_msg = f"Timeout simulado na fonte '{fonte.id}' para demonstracao da regra R5."
        for tentativa in range(1, MAX_TENTATIVAS_REDE + 1):
            if tentativa < MAX_TENTATIVAS_REDE:
                espera = self.backoff_segundos[min(tentativa - 1, len(self.backoff_segundos) - 1)]
                self.sleep_fn(espera)
        duracao_ms = max(1, int((time.perf_counter() - inicio) * 1000))
        return ResultadoDownload(
            ok=False,
            tentativas=MAX_TENTATIVAS_REDE,
            duracao_ms=duracao_ms,
            codigo_erro=STATUS_ERRO_REDE,
            mensagem_erro=erro_msg,
        )

    def _ler_amostra_local(self, fonte: FonteConfig) -> ResultadoDownload:
        inicio = time.perf_counter()
        if not self.amostras_dir:
            return ResultadoDownload(
                ok=False,
                tentativas=1,
                duracao_ms=1,
                codigo_erro=STATUS_ERRO_CONTEUDO,
                mensagem_erro="Diretorio de amostras offline nao configurado.",
            )

        ext = "pdf" if fonte.tipo == "pdf" else "html"
        if fonte.id in self.simular_mudanca:
            caminho = self.amostras_dir / f"{fonte.id}_alterada.{ext}"
        else:
            caminho = self.amostras_dir / f"{fonte.id}.{ext}"

        if not caminho.is_file():
            return ResultadoDownload(
                ok=False,
                tentativas=1,
                duracao_ms=max(1, int((time.perf_counter() - inicio) * 1000)),
                codigo_erro=STATUS_ERRO_CONTEUDO,
                mensagem_erro=f"Amostra local nao encontrada: {caminho.name}",
            )

        conteudo = caminho.read_bytes()
        duracao_ms = max(1, int((time.perf_counter() - inicio) * 1000))
        return ResultadoDownload(
            ok=True,
            conteudo=conteudo,
            status_http=200,
            tentativas=1,
            duracao_ms=duracao_ms,
        )

    def _baixar_http(self, fonte: FonteConfig) -> ResultadoDownload:
        if not fonte.url or not fonte.url.startswith("https://"):
            return ResultadoDownload(
                ok=False,
                tentativas=1,
                duracao_ms=1,
                codigo_erro=STATUS_ERRO_HTTP,
                mensagem_erro="URL ausente ou fora do protocolo HTTPS obrigatorio.",
            )

        inicio = time.perf_counter()
        ultimo_erro = "Erro desconhecido de rede."
        ultimo_status: int | None = None

        with httpx.Client(
            timeout=self.timeout_segundos,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            transport=self.transport,
        ) as cliente:
            for tentativa in range(1, MAX_TENTATIVAS_REDE + 1):
                try:
                    resp = cliente.get(fonte.url)
                    ultimo_status = resp.status_code

                    if 200 <= resp.status_code < 300:
                        duracao_ms = max(1, int((time.perf_counter() - inicio) * 1000))
                        return ResultadoDownload(
                            ok=True,
                            conteudo=resp.content,
                            status_http=resp.status_code,
                            tentativas=tentativa,
                            duracao_ms=duracao_ms,
                        )

                    if 400 <= resp.status_code < 500:
                        duracao_ms = max(1, int((time.perf_counter() - inicio) * 1000))
                        return ResultadoDownload(
                            ok=False,
                            status_http=resp.status_code,
                            tentativas=tentativa,
                            duracao_ms=duracao_ms,
                            codigo_erro=STATUS_ERRO_HTTP,
                            mensagem_erro=f"HTTP {resp.status_code} (sem nova tentativa, regra R6).",
                        )

                    ultimo_erro = f"HTTP {resp.status_code} do servidor remoto."
                except httpx.TimeoutException as exc:
                    ultimo_erro = f"Timeout apos {self.timeout_segundos}s: {type(exc).__name__}"
                except httpx.RequestError as exc:
                    ultimo_erro = f"Erro de conexao: {type(exc).__name__}"

                if tentativa < MAX_TENTATIVAS_REDE:
                    espera = self.backoff_segundos[min(tentativa - 1, len(self.backoff_segundos) - 1)]
                    self.sleep_fn(espera)

        duracao_ms = max(1, int((time.perf_counter() - inicio) * 1000))
        return ResultadoDownload(
            ok=False,
            status_http=ultimo_status,
            tentativas=MAX_TENTATIVAS_REDE,
            duracao_ms=duracao_ms,
            codigo_erro=STATUS_ERRO_REDE,
            mensagem_erro=ultimo_erro,
        )
