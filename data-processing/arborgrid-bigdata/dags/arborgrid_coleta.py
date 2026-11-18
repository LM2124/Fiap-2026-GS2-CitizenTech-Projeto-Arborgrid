"""
DAG de coleta do ArborGrid - Global Solution FIAP 2026.2 (Big Data)

Fluxo:
    [coleta_clima, coleta_arvores, coleta_denuncias]  (em paralelo)
        -> verifica_arquivos

- Roda uma vez por dia (@daily), com catchup=True desde 10/11/2026 (primeiro lote da API ArborGrid).
- Cada coleta grava UM arquivo em chegada/<fonte>/<fonte>_<ds>.json (JSON Lines:
  um registro por linha). Reexecutar o mesmo dia sobrescreve o mesmo arquivo (idempotente).
- Credenciais ficam em Variables do Airflow e são lidas dentro das tarefas.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Variable, dag, task

log = logging.getLogger(__name__)

CHEGADA = Path(os.environ.get("ARBORGRID_CHEGADA", "/opt/airflow/chegada"))
FONTES = ["clima", "arvores", "denuncias"]


def gravar(registros: list[dict], fonte: str, ds: str) -> str:
    """Grava chegada/<fonte>/<fonte>_<ds>.json, um registro JSON por linha.
    Escreve primeiro num arquivo temporário e depois renomeia: nunca fica arquivo pela metade."""
    pasta = CHEGADA / fonte
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"{fonte}_{ds}.json"
    temporario = destino.with_name(destino.name + ".tmp")
    with open(temporario, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    temporario.replace(destino)
    log.info("%s: %d registros gravados em %s", fonte, len(registros), destino)
    return str(destino)


def avisar_falha(context) -> None:
    """on_failure_callback: deixa no log qual tarefa falhou e em qual data lógica."""
    ti = context.get("task_instance")
    log.error("[ArborGrid] FALHA: tarefa=%s ds=%s tentativa=%s",
              getattr(ti, "task_id", "?"), context.get("ds"), getattr(ti, "try_number", "?"))


@dag(
    dag_id="arborgrid_coleta",
    description="Coleta diária: clima, árvores (API ArborGrid) e denúncias",
    schedule="@daily",
    start_date=datetime(2026, 11, 10),
    catchup=True,
    max_active_runs=1,  # um dia por vez: poupa memória e respeita os limites das APIs públicas
    default_args={
        "owner": "arborgrid-bigdata",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "on_failure_callback": avisar_falha,
    },
    tags=["arborgrid", "gs2026", "bigdata"],
)
def arborgrid_coleta():

    @task
    def coleta_clima(**context) -> str:
        from arborgrid.fontes import coletar_clima
        ds = context["ds"]
        return gravar(coletar_clima(ds), "clima", ds)

    @task
    def coleta_arvores(**context) -> str:
        from arborgrid.fontes import coletar_arvores
        ds = context["ds"]
        # Lidas aqui dentro, e não no topo do arquivo, para não consultar o banco a cada leitura do DAG.
        url = Variable.get("arborgrid_api_url")
        token = Variable.get("arborgrid_api_token")
        return gravar(coletar_arvores(ds, url, token), "arvores", ds)

    @task
    def coleta_denuncias(**context) -> str:
        from arborgrid.fontes import gerar_denuncias
        ds = context["ds"]
        token = Variable.get("arborgrid_app_token")
        return gravar(gerar_denuncias(ds, token=token), "denuncias", ds)

    @task
    def verifica_arquivos(**context) -> None:
        ds = context["ds"]
        problemas = []
        for fonte in FONTES:
            arquivo = CHEGADA / fonte / f"{fonte}_{ds}.json"
            if not arquivo.exists():
                problemas.append(f"faltando: {arquivo}")
            elif arquivo.stat().st_size == 0:
                problemas.append(f"vazio: {arquivo}")
            else:
                with open(arquivo, encoding="utf-8") as f:
                    linhas = sum(1 for _ in f)
                log.info("OK %s (%d linhas)", arquivo, linhas)
        if problemas:
            raise FileNotFoundError("Arquivos do dia com problema: " + "; ".join(problemas))

    [coleta_clima(), coleta_arvores(), coleta_denuncias()] >> verifica_arquivos()


arborgrid_coleta()
