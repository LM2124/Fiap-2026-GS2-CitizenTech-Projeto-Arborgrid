"""Funções de coleta de cada fonte do ArborGrid.

Cada função devolve uma lista de registros (dicionários). Quem grava em disco é o DAG.
A resposta das APIs é guardada como veio; só acrescentamos metadados da coleta
(fonte, ds_coleta, coletado_em) para rastreabilidade na Bronze.
"""
from __future__ import annotations

import hashlib
import logging
import random
from datetime import date, datetime, timedelta, timezone

import requests

from arborgrid.config import (
    ARVORES_ENDPOINT, ARVORES_LIMITE_PAGINA, OPEN_METEO_URL, REGIOES, USER_AGENT, VARIAVEIS_CLIMA,
)

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": USER_AGENT}
FUSO_SP = timezone(timedelta(hours=-3))


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _meta(fonte: str, ds: str) -> dict:
    return {"fonte": fonte, "ds_coleta": ds, "coletado_em": _agora()}


# ---------------------------------------------------------------------------
# F1 - Open-Meteo: previsão hora a hora de ds até ds+2 (janela de 48h), por região
# ---------------------------------------------------------------------------
def coletar_clima(ds: str) -> list[dict]:
    inicio = date.fromisoformat(ds)
    params = {
        "latitude": ",".join(str(r["lat"]) for r in REGIOES),
        "longitude": ",".join(str(r["lon"]) for r in REGIOES),
        "hourly": ",".join(VARIAVEIS_CLIMA),
        "wind_speed_unit": "kmh",
        "timezone": "America/Sao_Paulo",
        "start_date": inicio.isoformat(),
        "end_date": (inicio + timedelta(days=2)).isoformat(),
    }
    resp = requests.get(OPEN_METEO_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    respostas = resp.json()
    if isinstance(respostas, dict):  # com um único ponto a API devolve objeto, não lista
        respostas = [respostas]
    if len(respostas) != len(REGIOES):
        raise ValueError(f"Open-Meteo devolveu {len(respostas)} pontos; esperado {len(REGIOES)}")

    meta = _meta("open-meteo", ds)
    # A API devolve na mesma ordem dos pontos pedidos
    return [
        {**meta, "region_key": r["region_key"], "lat_pedida": r["lat"], "lon_pedida": r["lon"], **resp_r}
        for r, resp_r in zip(REGIOES, respostas)
    ]


# ---------------------------------------------------------------------------
# F3 - API ArborGrid (Visão Computacional): inspeções do lote do dia
# ---------------------------------------------------------------------------
def coletar_arvores(ds: str, url_base: str, token: str) -> list[dict]:
    """Uma linha por PÁGINA da API, exatamente como veio (envelope + items[]).
    Dia sem lote -> a API devolve total 0 e items [], e isso é gravado normalmente."""
    if not url_base or not token:
        raise PermissionError("URL ou token da API ArborGrid ausentes (Variables arborgrid_api_url / arborgrid_api_token)")

    meta = _meta("api-arborgrid-inspections", ds)
    headers = {**HEADERS, "Authorization": f"Bearer {token}"}
    paginas, offset = [], 0
    while True:
        params = {"date": ds, "limit": ARVORES_LIMITE_PAGINA, "offset": offset}
        resp = requests.get(url_base.rstrip("/") + ARVORES_ENDPOINT, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        pagina = resp.json()
        paginas.append({**meta, "pagina": len(paginas) + 1, **pagina})
        offset += len(pagina.get("items", []))
        if not pagina.get("items") or offset >= pagina.get("total", 0):
            return paginas


# ---------------------------------------------------------------------------
# F4 - Denúncias dos moradores (SIMULADAS)
# ---------------------------------------------------------------------------
TIPOS_DENUNCIA = ["galho_na_fiacao", "arvore_inclinada", "arvore_seca", "fio_partido"]
_NOMES = ["Ana", "Bruno", "Carla", "Diego", "Elisa", "Fabio", "Gabriela", "Hugo",
          "Iara", "Joao", "Karen", "Lucas", "Marina", "Nicolas", "Olivia", "Pedro"]
_SOBRENOMES = ["Silva", "Souza", "Oliveira", "Santos", "Lima", "Costa", "Pereira", "Rocha"]


def _cidadaos_ficticios(n: int = 60) -> list[dict]:
    """Pool fixo de moradores FICTÍCIOS: o mesmo morador pode denunciar em dias diferentes."""
    rng = random.Random(2026)
    pool = []
    for i in range(n):
        nome, sobrenome = rng.choice(_NOMES), rng.choice(_SOBRENOMES)
        pool.append({
            "nome": f"{nome} {sobrenome}",
            "email": f"{nome.lower()}.{sobrenome.lower()}{i}@exemplo.com",
            "telefone": f"(11) 9{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}",
        })
    return pool


def gerar_denuncias(ds: str, token: str) -> list[dict]:
    """Simula a API de denúncias do app. Mesma ds -> mesmas denúncias (determinístico).

    - data_hora_relato: quando o morador registrou no celular;
    - data_hora_envio: quando chegou ao servidor (pode atrasar horas por falta de sinal);
    - latitude/longitude: GPS do celular do morador (dado pessoal).
    Alguns registros vêm com problemas de propósito, para as regras de qualidade da Prata.
    """
    if not token:
        raise PermissionError("Token do app ausente (Variable arborgrid_app_token)")

    semente = int(hashlib.sha256(f"denuncias-{ds}".encode()).hexdigest(), 16) % 2**32
    rng = random.Random(semente)
    moradores = _cidadaos_ficticios()
    dia = date.fromisoformat(ds)

    denuncias = []
    for i in range(rng.randint(8, 20)):
        r = rng.choice(REGIOES)
        relato = datetime(dia.year, dia.month, dia.day, rng.randint(6, 22),
                          rng.randint(0, 59), tzinfo=FUSO_SP)
        # 1 em cada 4 denúncias chega atrasada (sem sinal no momento do relato)
        atraso = timedelta(hours=rng.randint(1, 10)) if rng.random() < 0.25 else timedelta(seconds=rng.randint(2, 30))
        denuncias.append({
            "id_denuncia": f"DN-{ds.replace('-', '')}-{i:03d}",
            "data_hora_relato": relato.isoformat(),
            "data_hora_envio": (relato + atraso).isoformat(),
            "latitude": round(r["lat"] + rng.uniform(-0.008, 0.008), 6),
            "longitude": round(r["lon"] + rng.uniform(-0.008, 0.008), 6),
            "tipo": rng.choice(TIPOS_DENUNCIA),
            "descricao": "Relato enviado pelo app ArborGrid",
            "foto_url": f"https://app.arborgrid.example/fotos/{ds}/{i:03d}.jpg",
            "reportante": dict(rng.choice(moradores)),
        })

    # Registros problemáticos de propósito
    if len(denuncias) >= 3:
        denuncias[0]["latitude"] = None                                  # celular sem GPS
        denuncias[1]["tipo"] = "outro"                                   # tipo fora do catálogo
        relato_2 = datetime.fromisoformat(denuncias[2]["data_hora_relato"])
        denuncias[2]["data_hora_envio"] = (relato_2 - timedelta(hours=2)).isoformat()  # relógio errado
        denuncias.append(dict(denuncias[-1]))                            # reenvio duplicado (mesmo id)

    meta = _meta("app-arborgrid-denuncias", ds)
    return [{**meta, **d} for d in denuncias]
