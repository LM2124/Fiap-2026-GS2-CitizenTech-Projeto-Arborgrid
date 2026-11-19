"""Envia o lote da camada Ouro para o MVP ArborGrid (POST /api/v1/gold/risk-scores).

Uso (Windows / cmd):
    set ARBORGRID_API_TOKEN=<token da API>
    python enviar_ouro_mvp.py gold_risk_scores_20261117.json

O token NUNCA fica no código: vem da variável de ambiente ARBORGRID_API_TOKEN.
A URL pode ser trocada com ARBORGRID_API_URL (padrão: http://localhost).
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL_BASE = os.environ.get("ARBORGRID_API_URL", "http://localhost").rstrip("/")
ENDPOINT = "/api/v1/gold/risk-scores"


def main() -> int:
    if len(sys.argv) != 2:
        print("Uso: python enviar_ouro_mvp.py <arquivo_do_lote.json>")
        return 2
    token = os.environ.get("ARBORGRID_API_TOKEN")
    if not token:
        print("Defina a variável de ambiente ARBORGRID_API_TOKEN antes de rodar.")
        return 2

    with open(sys.argv[1], encoding="utf-8") as f:
        lote = json.load(f)
    print(f"Enviando {len(lote['records'])} registros (batch_id={lote['batch_id']}) para {URL_BASE}{ENDPOINT}")

    req = urllib.request.Request(
        URL_BASE + ENDPOINT,
        data=json.dumps(lote, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print("HTTP", resp.status)
            print(resp.read().decode("utf-8"))
            return 0
    except urllib.error.HTTPError as erro:
        print("HTTP", erro.code)
        print(erro.read().decode("utf-8"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
