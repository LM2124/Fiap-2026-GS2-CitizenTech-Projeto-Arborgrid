"""Orquestrador da execucao em lote, persistencia de estado, saidas e auditoria."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

from .auditoria import criar_logger_execucao, fechar_logger, registrar_evento
from .cliente_http import ClienteFontes
from .modelos import (
    EXCECAO_NEGOCIO,
    EXCECAO_SISTEMA,
    STATUS_CONFIRMADA,
    STATUS_ERRO_CONTEUDO,
    STATUS_ERRO_SISTEMA,
    STATUS_FONTE_FECHADA,
    STATUS_FONTE_INTERNA,
    STATUS_NAO_VERIFICADO,
    STATUS_REVISAO_HUMANA,
    STATUS_SEM_ALTERACAO,
    STATUS_SUCESSO_FONTE,
    FonteConfig,
    ResultadoDownload,
    ResultadoExecucao,
    ResultadoFonte,
    TrechoConfig,
)
from .regras import avaliar_fonte

CAMPOS_ORIGINAIS_BASE = ("chunk_id", "titulo", "fonte", "categoria", "conteudo")


def carregar_fontes(caminho_yaml: Path) -> list[FonteConfig]:
    """Le fontes.yaml e devolve a lista tipada de fontes configuradas."""
    dados = yaml.safe_load(caminho_yaml.read_text(encoding="utf-8")) or {}
    fontes: list[FonteConfig] = []
    for item in dados.get("fontes", []):
        trechos = tuple(
            TrechoConfig(
                chunk_id=str(t["chunk_id"]),
                fatos=tuple(str(f) for f in t.get("fatos", [])),
            )
            for t in item.get("trechos", [])
        )
        fontes.append(
            FonteConfig(
                id=str(item["id"]),
                nome=str(item["nome"]),
                tipo=str(item["tipo"]),
                url=item.get("url"),
                url_verificada=bool(item.get("url_verificada", False)),
                motivo_sem_verificacao=item.get("motivo_sem_verificacao"),
                trechos=trechos,
            )
        )
    return fontes


def carregar_base_jsonl(caminho_base: Path) -> list[dict[str, Any]]:
    """Le a base JSONL original preservando ordem e conteudo exato dos trechos."""
    trechos: list[dict[str, Any]] = []
    with caminho_base.open("r", encoding="utf-8") as fp:
        for linha in fp:
            limpa = linha.strip()
            if limpa:
                trechos.append(json.loads(limpa))
    return trechos


def carregar_estado(caminho_estado: Path) -> dict[str, dict[str, Any]]:
    """Carrega hashes anteriores de estado/ultima_execucao.json se existir."""
    if not caminho_estado.is_file():
        return {}
    try:
        dados = json.loads(caminho_estado.read_text(encoding="utf-8"))
        return dados if isinstance(dados, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _agora_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def executar_rpa(
    caminho_fontes: Path,
    caminho_base: Path,
    amostras_dir: Path,
    saida_dir: Path,
    estado_dir: Path,
    logs_dir: Path,
    offline: bool = False,
    simular_mudanca: set[str] | None = None,
    simular_falha: set[str] | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ResultadoExecucao:
    """Executa uma rodada completa de verificacao das fontes normativas."""
    exec_id = uuid.uuid4().hex[:8]
    inicio_dt = datetime.now().astimezone()
    inicio_iso = inicio_dt.isoformat(timespec="seconds")
    t0 = time.perf_counter()

    logger, caminho_log = criar_logger_execucao(logs_dir, agora=inicio_dt)
    modo = "offline" if offline else "online"

    try:
        registrar_evento(
            logger,
            exec_id=exec_id,
            fonte="lote",
            etapa="inicio",
            status="INICIADO",
            modo=modo,
            base=caminho_base.name,
        )

        fontes = carregar_fontes(caminho_fontes)
        trechos_base = carregar_base_jsonl(caminho_base)
        caminho_estado = estado_dir / "ultima_execucao.json"
        estado_anterior = carregar_estado(caminho_estado)

        cliente = ClienteFontes(
            offline=offline,
            amostras_dir=amostras_dir,
            simular_mudanca=simular_mudanca,
            simular_falha=simular_falha,
            transport=transport,
            sleep_fn=sleep_fn,
        )

        resultados_fontes: list[ResultadoFonte] = []
        verificacoes_por_chunk: dict[str, dict[str, Any]] = {}
        novo_estado: dict[str, dict[str, Any]] = dict(estado_anterior)

        for fonte in fontes:
            verificado_em = _agora_iso()
            hash_anterior = (estado_anterior.get(fonte.id) or {}).get("sha256")
            download: ResultadoDownload | None = None

            try:
                if fonte.tipo not in ("fechada", "interna"):
                    download = cliente.obter(fonte)
                    registrar_evento(
                        logger,
                        exec_id=exec_id,
                        fonte=fonte.id,
                        etapa="download",
                        status="OK" if download.ok else (download.codigo_erro or "ERRO"),
                        nivel=logging.INFO if download.ok else logging.ERROR,
                        tentativa=download.tentativas,
                        bytes=len(download.conteudo),
                        ms=download.duracao_ms,
                        http=download.status_http,
                    )
                else:
                    registrar_evento(
                        logger,
                        exec_id=exec_id,
                        fonte=fonte.id,
                        etapa="download",
                        status="IGNORADO",
                        tipo=fonte.tipo,
                    )

                res_fonte, verif_fonte, hash_para_estado = avaliar_fonte(
                    fonte=fonte,
                    download=download,
                    hash_anterior=hash_anterior,
                    verificado_em=verificado_em,
                )

                if fonte.tipo not in ("fechada", "interna"):
                    if res_fonte.status == STATUS_ERRO_CONTEUDO and download and download.ok:
                        registrar_evento(
                            logger,
                            exec_id=exec_id,
                            fonte=fonte.id,
                            etapa="extracao",
                            status=STATUS_ERRO_CONTEUDO,
                            nivel=logging.ERROR,
                            detalhe=res_fonte.motivo,
                        )
                    elif download and download.ok:
                        registrar_evento(
                            logger,
                            exec_id=exec_id,
                            fonte=fonte.id,
                            etapa="extracao",
                            status="OK",
                            sha256=(res_fonte.sha256 or "")[:12],
                        )

                nivel_decisao = logging.INFO
                if res_fonte.excecao_tipo == EXCECAO_NEGOCIO:
                    nivel_decisao = logging.WARNING
                elif res_fonte.excecao_tipo == EXCECAO_SISTEMA:
                    nivel_decisao = logging.ERROR

                registrar_evento(
                    logger,
                    exec_id=exec_id,
                    fonte=fonte.id,
                    etapa="decisao",
                    status=res_fonte.status,
                    nivel=nivel_decisao,
                    trechos=len(fonte.trechos),
                    excecao=res_fonte.excecao_tipo or "nenhuma",
                )

            except Exception as exc:
                # Garante que nenhuma falha inesperada passe silenciosa nem aborte o lote
                res_fonte = ResultadoFonte(
                    fonte_id=fonte.id,
                    nome=fonte.nome,
                    tipo=fonte.tipo,
                    url=fonte.url,
                    url_verificada=fonte.url_verificada,
                    status=STATUS_ERRO_CONTEUDO,
                    motivo=f"Erro inesperado ao processar fonte: {type(exc).__name__}",
                    excecao_tipo=EXCECAO_SISTEMA,
                    trechos_ids=[t.chunk_id for t in fonte.trechos],
                )
                verif_fonte = {
                    t.chunk_id: {
                        "status": STATUS_NAO_VERIFICADO,
                        "fonte_id": fonte.id,
                        "url": fonte.url,
                        "sha256": hash_anterior,
                        "verificado_em": verificado_em,
                        "fatos_ausentes": [],
                    }
                    for t in fonte.trechos
                }
                hash_para_estado = hash_anterior
                registrar_evento(
                    logger,
                    exec_id=exec_id,
                    fonte=fonte.id,
                    etapa="decisao",
                    status=STATUS_ERRO_CONTEUDO,
                    nivel=logging.ERROR,
                    erro=type(exc).__name__,
                )

            resultados_fontes.append(res_fonte)
            verificacoes_por_chunk.update(verif_fonte)
            if hash_para_estado is not None or fonte.tipo in ("fechada", "interna"):
                novo_estado[fonte.id] = {
                    "sha256": hash_para_estado,
                    "status": res_fonte.status,
                    "verificado_em": verificado_em,
                }

        # Monta base sincronizada mantendo os 5 campos originais inalterados e na mesma ordem
        linhas_sincronizadas: list[dict[str, Any]] = []
        trechos_relatorio: list[dict[str, Any]] = []
        agora_gravacao = _agora_iso()

        for original in trechos_base:
            chunk_id = str(original["chunk_id"])
            verif = verificacoes_por_chunk.get(
                chunk_id,
                {
                    "status": STATUS_NAO_VERIFICADO,
                    "fonte_id": None,
                    "url": None,
                    "sha256": None,
                    "verificado_em": agora_gravacao,
                    "fatos_ausentes": [],
                },
            )
            registro = {campo: original[campo] for campo in CAMPOS_ORIGINAIS_BASE}
            registro["verificacao"] = verif
            linhas_sincronizadas.append(registro)
            trechos_relatorio.append(
                {
                    "chunk_id": chunk_id,
                    "fonte_id": verif["fonte_id"],
                    "status": verif["status"],
                    "fatos_ausentes": list(verif.get("fatos_ausentes") or []),
                    "verificado_em": verif["verificado_em"],
                }
            )

        # Persiste saidas e estado
        saida_dir.mkdir(parents=True, exist_ok=True)
        estado_dir.mkdir(parents=True, exist_ok=True)

        caminho_base_sync = saida_dir / "knowledge_base_synced.jsonl"
        with caminho_base_sync.open("w", encoding="utf-8") as fp:
            for item in linhas_sincronizadas:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

        caminho_estado.write_text(
            json.dumps(novo_estado, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        contagem_fontes = dict(Counter(r.status for r in resultados_fontes))
        contagem_trechos = dict(Counter(t["status"] for t in trechos_relatorio))

        excecoes_negocio = [
            {
                "fonte_id": r.fonte_id,
                "status": r.status,
                "motivo": r.motivo,
                "trechos_afetados": [
                    t["chunk_id"]
                    for t in trechos_relatorio
                    if t["fonte_id"] == r.fonte_id and t["status"] == STATUS_REVISAO_HUMANA
                ],
            }
            for r in resultados_fontes
            if r.excecao_tipo == EXCECAO_NEGOCIO
        ]
        excecoes_sistema = [
            {
                "fonte_id": r.fonte_id,
                "status": r.status,
                "motivo": r.motivo,
                "tentativas": r.tentativas,
            }
            for r in resultados_fontes
            if r.excecao_tipo == EXCECAO_SISTEMA
        ]

        tem_problema = bool(excecoes_negocio or excecoes_sistema) or any(
            r.status not in STATUS_SUCESSO_FONTE for r in resultados_fontes
        )
        codigo_saida = 2 if tem_problema else 0

        fim_iso = _agora_iso()
        duracao_ms = max(1, int((time.perf_counter() - t0) * 1000))

        relatorio = {
            "execucao_id": exec_id,
            "inicio": inicio_iso,
            "fim": fim_iso,
            "duracao_ms": duracao_ms,
            "modo": modo,
            "codigo_saida": codigo_saida,
            "total_fontes": len(resultados_fontes),
            "total_trechos": len(trechos_relatorio),
            "contagem_status_fontes": contagem_fontes,
            "contagem_status_trechos": contagem_trechos,
            "excecoes_negocio": excecoes_negocio,
            "excecoes_sistema": excecoes_sistema,
            "fontes": [r.para_dict() for r in resultados_fontes],
            "trechos": trechos_relatorio,
        }

        caminho_relatorio = saida_dir / "relatorio_execucao.json"
        caminho_relatorio.write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        registrar_evento(
            logger,
            exec_id=exec_id,
            fonte="lote",
            etapa="gravacao",
            status="OK",
            base_sync=caminho_base_sync.name,
            relatorio=caminho_relatorio.name,
            trechos=len(linhas_sincronizadas),
        )

        confirmadas = contagem_fontes.get(STATUS_CONFIRMADA, 0)
        sem_alteracao = contagem_fontes.get(STATUS_SEM_ALTERACAO, 0)
        revisao = contagem_fontes.get(STATUS_REVISAO_HUMANA, 0)
        erros = sum(contagem_fontes.get(s, 0) for s in STATUS_ERRO_SISTEMA)
        fechadas = contagem_fontes.get(STATUS_FONTE_FECHADA, 0) + contagem_fontes.get(
            STATUS_FONTE_INTERNA, 0
        )

        registrar_evento(
            logger,
            exec_id=exec_id,
            fonte="lote",
            etapa="fim",
            status="OK" if codigo_saida == 0 else "ATENCAO",
            nivel=logging.INFO if codigo_saida == 0 else logging.WARNING,
            codigo_saida=codigo_saida,
            confirmadas=confirmadas,
            sem_alteracao=sem_alteracao,
            revisao=revisao,
            erros=erros,
            fechadas=fechadas,
            ms=duracao_ms,
        )

        return ResultadoExecucao(
            execucao_id=exec_id,
            codigo_saida=codigo_saida,
            relatorio=relatorio,
            arquivo_base_sincronizada=str(caminho_base_sync),
            arquivo_relatorio=str(caminho_relatorio),
            arquivo_estado=str(caminho_estado),
            arquivo_log=str(caminho_log),
        )
    finally:
        fechar_logger(logger)
