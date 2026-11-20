"""Aplicacao das regras de negocio R1 a R7 sobre fontes e trechos."""

from __future__ import annotations

from typing import Any

from .extrator import ErroExtracaoConteudo, extrair_e_hashear, normalizar_texto
from .modelos import (
    EXCECAO_NEGOCIO,
    EXCECAO_SISTEMA,
    STATUS_CONFIRMADA,
    STATUS_ERRO_CONTEUDO,
    STATUS_FONTE_FECHADA,
    STATUS_FONTE_INTERNA,
    STATUS_NAO_VERIFICADO,
    STATUS_REVISAO_HUMANA,
    STATUS_SEM_ALTERACAO,
    FonteConfig,
    ResultadoDownload,
    ResultadoFonte,
)


def _verificacao_trecho(
    status: str,
    fonte: FonteConfig,
    sha256: str | None,
    verificado_em: str,
    fatos_ausentes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "fonte_id": fonte.id,
        "url": fonte.url,
        "sha256": sha256,
        "verificado_em": verificado_em,
        "fatos_ausentes": list(fatos_ausentes or []),
    }


def avaliar_fonte(
    fonte: FonteConfig,
    download: ResultadoDownload | None,
    hash_anterior: str | None,
    verificado_em: str,
) -> tuple[ResultadoFonte, dict[str, dict[str, Any]], str | None]:
    """Avalia uma fonte aplicando R1 a R7 e retorna (resultado_fonte, verificacoes_por_chunk, novo_hash)."""
    trechos_ids = [t.chunk_id for t in fonte.trechos]

    # R1: fontes fechadas ou internas nao sao baixadas nem verificadas externamente
    if fonte.tipo in ("fechada", "interna"):
        status_r1 = STATUS_FONTE_FECHADA if fonte.tipo == "fechada" else STATUS_FONTE_INTERNA
        motivo_padrao = (
            "Norma tecnica de acesso fechado (ABNT/MTE), sem verificacao automatica."
            if fonte.tipo == "fechada"
            else "Norma interna do sistema ArborGrid SP, sem fonte externa."
        )
        motivo = fonte.motivo_sem_verificacao or motivo_padrao
        verificacoes = {
            t.chunk_id: _verificacao_trecho(status_r1, fonte, None, verificado_em)
            for t in fonte.trechos
        }
        res = ResultadoFonte(
            fonte_id=fonte.id,
            nome=fonte.nome,
            tipo=fonte.tipo,
            url=fonte.url,
            url_verificada=fonte.url_verificada,
            status=status_r1,
            motivo=motivo,
            trechos_ids=trechos_ids,
        )
        return res, verificacoes, None

    # R5 / R6 / R7 na etapa de download
    if download is None or not download.ok:
        codigo_erro = (download.codigo_erro if download and download.codigo_erro else STATUS_ERRO_CONTEUDO)
        motivo = (download.mensagem_erro if download and download.mensagem_erro else "Falha ao obter fonte.")
        verificacoes = {
            t.chunk_id: _verificacao_trecho(STATUS_NAO_VERIFICADO, fonte, hash_anterior, verificado_em)
            for t in fonte.trechos
        }
        res = ResultadoFonte(
            fonte_id=fonte.id,
            nome=fonte.nome,
            tipo=fonte.tipo,
            url=fonte.url,
            url_verificada=fonte.url_verificada,
            status=codigo_erro,
            motivo=motivo,
            tentativas=download.tentativas if download else 0,
            bytes_baixados=0,
            duracao_ms=download.duracao_ms if download else 0,
            sha256=hash_anterior,
            excecao_tipo=EXCECAO_SISTEMA,
            trechos_ids=trechos_ids,
        )
        return res, verificacoes, hash_anterior

    # R7: extracao de texto e calculo do hash SHA-256
    bytes_baixados = len(download.conteudo)
    try:
        texto_normalizado, sha256_atual = extrair_e_hashear(download.conteudo, fonte.tipo)
    except ErroExtracaoConteudo as exc:
        verificacoes = {
            t.chunk_id: _verificacao_trecho(STATUS_NAO_VERIFICADO, fonte, hash_anterior, verificado_em)
            for t in fonte.trechos
        }
        res = ResultadoFonte(
            fonte_id=fonte.id,
            nome=fonte.nome,
            tipo=fonte.tipo,
            url=fonte.url,
            url_verificada=fonte.url_verificada,
            status=STATUS_ERRO_CONTEUDO,
            motivo=str(exc),
            tentativas=download.tentativas,
            bytes_baixados=bytes_baixados,
            duracao_ms=download.duracao_ms,
            sha256=hash_anterior,
            excecao_tipo=EXCECAO_SISTEMA,
            trechos_ids=trechos_ids,
        )
        return res, verificacoes, hash_anterior

    # R2: hash igual ao da ultima execucao confirmada -> nao reextrai fatos
    if hash_anterior is not None and sha256_atual == hash_anterior:
        verificacoes = {
            t.chunk_id: _verificacao_trecho(STATUS_SEM_ALTERACAO, fonte, sha256_atual, verificado_em)
            for t in fonte.trechos
        }
        res = ResultadoFonte(
            fonte_id=fonte.id,
            nome=fonte.nome,
            tipo=fonte.tipo,
            url=fonte.url,
            url_verificada=fonte.url_verificada,
            status=STATUS_SEM_ALTERACAO,
            motivo="Conteudo inalterado em relacao a ultima verificacao confirmada (hash identico).",
            tentativas=download.tentativas,
            bytes_baixados=bytes_baixados,
            duracao_ms=download.duracao_ms,
            sha256=sha256_atual,
            trechos_ids=trechos_ids,
        )
        return res, verificacoes, sha256_atual

    # R7 (pagina sem o documento): se NENHUM fato monitorado aparece, o texto baixado nao e o documento
    # (endereco mudou, bloqueio ou pagina montada por JavaScript). Uma mudanca real na norma troca um
    # fato, nao todos: tratar isso como R4 mandaria a pessoa revisar uma norma que nao mudou.
    todos_fatos = [fato for trecho in fonte.trechos for fato in trecho.fatos]
    if len(todos_fatos) >= 2 and not any(normalizar_texto(f) in texto_normalizado for f in todos_fatos):
        verificacoes = {
            t.chunk_id: _verificacao_trecho(STATUS_NAO_VERIFICADO, fonte, hash_anterior, verificado_em)
            for t in fonte.trechos
        }
        res = ResultadoFonte(
            fonte_id=fonte.id,
            nome=fonte.nome,
            tipo=fonte.tipo,
            url=fonte.url,
            url_verificada=fonte.url_verificada,
            status=STATUS_ERRO_CONTEUDO,
            motivo=(
                f"Nenhum dos {len(todos_fatos)} fatos monitorados aparece no texto baixado: o endereco pode ter "
                "mudado, o portal pode ter bloqueado o acesso ou a pagina depende de JavaScript."
            ),
            tentativas=download.tentativas,
            bytes_baixados=bytes_baixados,
            duracao_ms=download.duracao_ms,
            sha256=hash_anterior,
            excecao_tipo=EXCECAO_SISTEMA,
            trechos_ids=trechos_ids,
        )
        return res, verificacoes, hash_anterior

    # R3 / R4: hash novo ou primeira execucao -> confere fatos esperados de cada trecho
    verificacoes = {}
    total_fatos_ausentes: list[str] = []
    for trecho in fonte.trechos:
        ausentes = [
            fato for fato in trecho.fatos if normalizar_texto(fato) not in texto_normalizado
        ]
        if ausentes:
            total_fatos_ausentes.extend(f"{trecho.chunk_id}:{fato}" for fato in ausentes)
            verificacoes[trecho.chunk_id] = _verificacao_trecho(
                STATUS_REVISAO_HUMANA,
                fonte,
                sha256_atual,
                verificado_em,
                fatos_ausentes=ausentes,
            )
        else:
            verificacoes[trecho.chunk_id] = _verificacao_trecho(
                STATUS_CONFIRMADA,
                fonte,
                sha256_atual,
                verificado_em,
            )

    if total_fatos_ausentes:
        # R4: mantem o hash confirmado anterior no estado para nao suprimir o alerta na proxima execucao
        res = ResultadoFonte(
            fonte_id=fonte.id,
            nome=fonte.nome,
            tipo=fonte.tipo,
            url=fonte.url,
            url_verificada=fonte.url_verificada,
            status=STATUS_REVISAO_HUMANA,
            motivo=f"Fatos ausentes detectados apos alteracao da fonte: {', '.join(total_fatos_ausentes)}",
            tentativas=download.tentativas,
            bytes_baixados=bytes_baixados,
            duracao_ms=download.duracao_ms,
            sha256=sha256_atual,
            excecao_tipo=EXCECAO_NEGOCIO,
            trechos_ids=trechos_ids,
        )
        return res, verificacoes, hash_anterior

    # R3: todos os fatos presentes -> confirma e grava o novo hash
    res = ResultadoFonte(
        fonte_id=fonte.id,
        nome=fonte.nome,
        tipo=fonte.tipo,
        url=fonte.url,
        url_verificada=fonte.url_verificada,
        status=STATUS_CONFIRMADA,
        motivo="Todos os fatos monitorados estao presentes no texto oficial.",
        tentativas=download.tentativas,
        bytes_baixados=bytes_baixados,
        duracao_ms=download.duracao_ms,
        sha256=sha256_atual,
        trechos_ids=trechos_ids,
    )
    return res, verificacoes, sha256_atual
