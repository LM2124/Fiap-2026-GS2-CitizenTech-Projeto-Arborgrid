"""Testes automatizados 100% offline do robo RPA normativo ArborGrid SP."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import yaml

from rpa_normativo.executor import CAMPOS_ORIGINAIS_BASE, carregar_base_jsonl, executar_rpa

RPA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RPA_ROOT.parent
FONTES_YAML = RPA_ROOT / "fontes.yaml"
BASE_JSONL = REPO_ROOT / "webapp" / "data" / "knowledge_base.jsonl"
AMOSTRAS_DIR = RPA_ROOT / "amostras" / "fontes"


def _gerar_pdf_minimo(linhas: list[str]) -> bytes:
    """Gera em memoria um PDF 1.4 valido de uma pagina com fonte Helvetica."""
    cmds = ["BT", "/F1 11 Tf", "50 750 Td", "15 TL"]
    for linha in linhas:
        escapada = linha.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        cmds.append(f"({escapada}) Tj T*")
    cmds.append("ET")
    stream = "\n".join(cmds).encode("latin-1")

    objetos = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        + stream
        + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    ]
    saida = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objetos:
        offsets.append(len(saida))
        saida.extend(obj)
    xref_pos = len(saida)
    saida.extend(f"xref\n0 {len(objetos) + 1}\n".encode("ascii"))
    saida.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        saida.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    saida.extend(
        f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(saida)


def _dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "saida", tmp_path / "estado", tmp_path / "logs"


def _criar_cenario_mini(
    tmp_path: Path,
    fontes_lista: list[dict],
    trechos_lista: list[dict],
) -> tuple[Path, Path]:
    fontes_path = tmp_path / "fontes.yaml"
    fontes_path.write_text(
        yaml.safe_dump({"fontes": fontes_lista}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    base_path = tmp_path / "base.jsonl"
    with base_path.open("w", encoding="utf-8") as fp:
        for item in trechos_lista:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    return fontes_path, base_path


def test_sucesso_primeira_execucao_offline_e_integridade_base(tmp_path: Path) -> None:
    """1. Primeira execucao offline confirma todas as fontes abertas e preserva todos os trechos da base."""
    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res = executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
    )

    assert res.codigo_saida == 0
    assert res.relatorio["total_trechos"] == 25
    assert res.relatorio["contagem_status_fontes"]["CONFIRMADA"] == 17
    assert res.relatorio["contagem_status_fontes"]["FONTE_FECHADA"] == 1
    assert res.relatorio["contagem_status_fontes"]["FONTE_INTERNA"] == 1

    originais = carregar_base_jsonl(BASE_JSONL)
    sincronizados = carregar_base_jsonl(Path(res.arquivo_base_sincronizada))
    assert len(sincronizados) == len(originais) == 25

    for orig, sync in zip(originais, sincronizados):
        for campo in CAMPOS_ORIGINAIS_BASE:
            assert sync[campo] == orig[campo]
        assert "verificacao" in sync
        assert sync["verificacao"]["fatos_ausentes"] == []
        assert sync["verificacao"]["status"] in {
            "CONFIRMADA",
            "FONTE_FECHADA",
            "FONTE_INTERNA",
        }


def test_idempotencia_segunda_execucao_sem_alteracao(tmp_path: Path) -> None:
    """2. Segunda execucao seguida com mesmo conteudo resulta em SEM_ALTERACAO (R2)."""
    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res1 = executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
    )
    assert res1.codigo_saida == 0

    res2 = executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
    )
    assert res2.codigo_saida == 0
    assert res2.relatorio["contagem_status_fontes"]["SEM_ALTERACAO"] == 17
    assert res2.relatorio["contagem_status_fontes"].get("CONFIRMADA", 0) == 0


def test_decisao_r4_revisao_humana_preserva_conteudo_original(tmp_path: Path) -> None:
    """3. Mudanca com fato ausente gera REVISAO_HUMANA, lista fato ausente e mantem conteudo intacto."""
    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)

    # Primeira execucao normal
    executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
    )

    # Segunda execucao com amostra alterada da Enel Emergencia (telefone 0800 72 72 196 trocado)
    res_mudanca = executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
        simular_mudanca={"enel-emergencia"},
    )

    assert res_mudanca.codigo_saida == 2
    assert len(res_mudanca.relatorio["excecoes_negocio"]) == 1
    assert res_mudanca.relatorio["excecoes_sistema"] == []
    assert res_mudanca.relatorio["excecoes_negocio"][0]["fonte_id"] == "enel-emergencia"

    originais = {c["chunk_id"]: c for c in carregar_base_jsonl(BASE_JSONL)}
    sincronizados = {
        c["chunk_id"]: c for c in carregar_base_jsonl(Path(res_mudanca.arquivo_base_sincronizada))
    }
    enel = sincronizados["ENEL-EMERG-01"]
    assert enel["conteudo"] == originais["ENEL-EMERG-01"]["conteudo"]
    assert enel["verificacao"]["status"] == "REVISAO_HUMANA"
    assert "0800 72 72 196" in enel["verificacao"]["fatos_ausentes"]

    # Terceira execucao ainda com amostra alterada nao pode silenciar para SEM_ALTERACAO
    res_repetida = executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
        simular_mudanca={"enel-emergencia"},
    )
    assert res_repetida.codigo_saida == 2
    assert res_repetida.relatorio["contagem_status_fontes"]["REVISAO_HUMANA"] == 1


def test_decisao_r1_fonte_fechada_e_interna_nao_disparam_http(tmp_path: Path) -> None:
    """4. Fontes fechada e interna nao fazem download nem chamam o transporte HTTP."""
    chamadas: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        chamadas.append(str(request.url))
        return httpx.Response(500)

    fontes_path, base_path = _criar_cenario_mini(
        tmp_path,
        fontes_lista=[
            {
                "id": "abnt-teste",
                "nome": "ABNT Teste",
                "tipo": "fechada",
                "trechos": [{"chunk_id": "C-FECHADA", "fatos": ["NBR"]}],
            },
            {
                "id": "interna-teste",
                "nome": "Interna Teste",
                "tipo": "interna",
                "trechos": [{"chunk_id": "C-INTERNA", "fatos": ["LGPD"]}],
            },
        ],
        trechos_lista=[
            {
                "chunk_id": "C-FECHADA",
                "titulo": "Norma fechada",
                "fonte": "ABNT",
                "categoria": "norma",
                "conteudo": "Texto NBR.",
            },
            {
                "chunk_id": "C-INTERNA",
                "titulo": "Diretriz interna",
                "fonte": "ArborGrid",
                "categoria": "sistema",
                "conteudo": "Texto LGPD.",
            },
        ],
    )

    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res = executar_rpa(
        caminho_fontes=fontes_path,
        caminho_base=base_path,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=False,
        transport=httpx.MockTransport(handler),
    )

    assert chamadas == []
    assert res.codigo_saida == 0
    assert res.relatorio["contagem_status_fontes"]["FONTE_FECHADA"] == 1
    assert res.relatorio["contagem_status_fontes"]["FONTE_INTERNA"] == 1


def test_retry_r5_503_recupera_na_terceira_e_timeout_continua_lote(tmp_path: Path) -> None:
    """5. 503, 503, 200 confirma na 3a tentativa; 3 timeouts geram ERRO_REDE sem parar o lote."""
    tentativas_recupera = 0
    tentativas_timeout = 0
    esperas_registradas: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal tentativas_recupera, tentativas_timeout
        url = str(request.url)
        if "recupera" in url:
            tentativas_recupera += 1
            if tentativas_recupera < 3:
                return httpx.Response(503)
            return httpx.Response(200, text="<html><body>Canal 156 ativo</body></html>")
        if "timeout" in url:
            tentativas_timeout += 1
            raise httpx.ReadTimeout("Timeout simulado", request=request)
        return httpx.Response(200, text="<html><body>Canal 193 ativo</body></html>")

    fontes_path, base_path = _criar_cenario_mini(
        tmp_path,
        fontes_lista=[
            {
                "id": "fonte-timeout",
                "nome": "Fonte Timeout",
                "url": "https://exemplo.sp.gov.br/timeout",
                "tipo": "html",
                "trechos": [{"chunk_id": "C-TO", "fatos": ["199"]}],
            },
            {
                "id": "fonte-recupera",
                "nome": "Fonte Recupera",
                "url": "https://exemplo.sp.gov.br/recupera",
                "tipo": "html",
                "trechos": [{"chunk_id": "C-REC", "fatos": ["156"]}],
            },
            {
                "id": "fonte-ok",
                "nome": "Fonte OK",
                "url": "https://exemplo.sp.gov.br/ok",
                "tipo": "html",
                "trechos": [{"chunk_id": "C-OK", "fatos": ["193"]}],
            },
        ],
        trechos_lista=[
            {"chunk_id": "C-TO", "titulo": "T1", "fonte": "F1", "categoria": "c", "conteudo": "199"},
            {"chunk_id": "C-REC", "titulo": "T2", "fonte": "F2", "categoria": "c", "conteudo": "156"},
            {"chunk_id": "C-OK", "titulo": "T3", "fonte": "F3", "categoria": "c", "conteudo": "193"},
        ],
    )

    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res = executar_rpa(
        caminho_fontes=fontes_path,
        caminho_base=base_path,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=False,
        transport=httpx.MockTransport(handler),
        sleep_fn=esperas_registradas.append,
    )

    assert tentativas_timeout == 3
    assert tentativas_recupera == 3
    assert esperas_registradas == [1.0, 2.0, 1.0, 2.0]
    assert res.codigo_saida == 2

    por_id = {f["fonte_id"]: f for f in res.relatorio["fontes"]}
    assert por_id["fonte-timeout"]["status"] == "ERRO_REDE"
    assert por_id["fonte-recupera"]["status"] == "CONFIRMADA"
    assert por_id["fonte-recupera"]["tentativas"] == 3
    assert por_id["fonte-ok"]["status"] == "CONFIRMADA"

    trechos = {t["chunk_id"]: t for t in res.relatorio["trechos"]}
    assert trechos["C-TO"]["status"] == "NAO_VERIFICADO"
    assert trechos["C-REC"]["status"] == "CONFIRMADA"
    assert trechos["C-OK"]["status"] == "CONFIRMADA"


def test_r6_http_404_sem_retry(tmp_path: Path) -> None:
    """6. HTTP 404 gera ERRO_HTTP em uma unica tentativa, sem espera de retry."""
    chamadas = 0
    esperas: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chamadas
        chamadas += 1
        return httpx.Response(404)

    fontes_path, base_path = _criar_cenario_mini(
        tmp_path,
        fontes_lista=[
            {
                "id": "fonte-404",
                "nome": "Fonte 404",
                "url": "https://exemplo.sp.gov.br/404",
                "tipo": "html",
                "trechos": [{"chunk_id": "C-404", "fatos": ["156"]}],
            }
        ],
        trechos_lista=[
            {"chunk_id": "C-404", "titulo": "T", "fonte": "F", "categoria": "c", "conteudo": "156"}
        ],
    )

    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res = executar_rpa(
        caminho_fontes=fontes_path,
        caminho_base=base_path,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=False,
        transport=httpx.MockTransport(handler),
        sleep_fn=esperas.append,
    )

    assert chamadas == 1
    assert esperas == []
    assert res.codigo_saida == 2
    assert res.relatorio["fontes"][0]["status"] == "ERRO_HTTP"
    assert res.relatorio["trechos"][0]["status"] == "NAO_VERIFICADO"
    assert len(res.relatorio["excecoes_sistema"]) == 1


def test_r7_html_sem_texto_gera_erro_conteudo(tmp_path: Path) -> None:
    """7. HTML sem texto visivel (apenas script/style) gera ERRO_CONTEUDO."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><head><style>body{}</style><script>var x=1;</script></head><body></body></html>",
        )

    fontes_path, base_path = _criar_cenario_mini(
        tmp_path,
        fontes_lista=[
            {
                "id": "html-vazio",
                "nome": "HTML Vazio",
                "url": "https://exemplo.sp.gov.br/vazio",
                "tipo": "html",
                "trechos": [{"chunk_id": "C-VAZIO", "fatos": ["156"]}],
            }
        ],
        trechos_lista=[
            {"chunk_id": "C-VAZIO", "titulo": "T", "fonte": "F", "categoria": "c", "conteudo": "156"}
        ],
    )

    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res = executar_rpa(
        caminho_fontes=fontes_path,
        caminho_base=base_path,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=False,
        transport=httpx.MockTransport(handler),
    )

    assert res.codigo_saida == 2
    assert res.relatorio["fontes"][0]["status"] == "ERRO_CONTEUDO"
    assert res.relatorio["trechos"][0]["status"] == "NAO_VERIFICADO"


def test_r7_pagina_sem_nenhum_fato_e_erro_de_conteudo_nao_revisao(tmp_path: Path) -> None:
    """7b. Pagina com texto, mas sem nenhum dos fatos (casca de JavaScript, bloqueio): ERRO_CONTEUDO, nao R4."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body><p>Catalogo de Legislacao Municipal. Aguarde...</p></body></html>")

    fontes_path, base_path = _criar_cenario_mini(
        tmp_path,
        fontes_lista=[
            {
                "id": "casca-js",
                "nome": "Portal que depende de JavaScript",
                "url": "https://exemplo.sp.gov.br/lei",
                "tipo": "html",
                "trechos": [{"chunk_id": "C-LEI", "fatos": ["17.794/2022", "Subprefeituras"]}],
            }
        ],
        trechos_lista=[
            {"chunk_id": "C-LEI", "titulo": "T", "fonte": "F", "categoria": "c", "conteudo": "Lei 17.794/2022"}
        ],
    )

    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res = executar_rpa(
        caminho_fontes=fontes_path,
        caminho_base=base_path,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=False,
        transport=httpx.MockTransport(handler),
    )

    assert res.codigo_saida == 2
    fonte = res.relatorio["fontes"][0]
    assert fonte["status"] == "ERRO_CONTEUDO"
    assert "Nenhum dos 2 fatos" in fonte["motivo"]
    assert res.relatorio["trechos"][0]["status"] == "NAO_VERIFICADO"
    assert res.relatorio["excecoes_negocio"] == []


def test_extracao_pdf_via_http_mockado(tmp_path: Path) -> None:
    """8. Extrai texto de PDF valido retornado por HTTP mockado e valida os fatos."""
    pdf_bytes = _gerar_pdf_minimo(
        [
            "Manual Tecnico de Poda - SVMA",
            "Podas de conducao, adequacao e limpeza preservando colar e crista.",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pdf_bytes)

    fontes_path, base_path = _criar_cenario_mini(
        tmp_path,
        fontes_lista=[
            {
                "id": "manual-pdf",
                "nome": "Manual PDF",
                "url": "https://exemplo.sp.gov.br/manual.pdf",
                "tipo": "pdf",
                "trechos": [
                    {
                        "chunk_id": "C-PDF",
                        "fatos": ["SVMA", "conducao", "adequacao", "limpeza"],
                    }
                ],
            }
        ],
        trechos_lista=[
            {"chunk_id": "C-PDF", "titulo": "PDF", "fonte": "SVMA", "categoria": "norma", "conteudo": "Manual"}
        ],
    )

    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)
    res = executar_rpa(
        caminho_fontes=fontes_path,
        caminho_base=base_path,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=False,
        transport=httpx.MockTransport(handler),
    )

    assert res.codigo_saida == 0
    assert res.relatorio["fontes"][0]["status"] == "CONFIRMADA"
    assert res.relatorio["fontes"][0]["sha256"]


def test_codigo_saida_0_e_2_e_formato_log_auditoria(tmp_path: Path) -> None:
    """9 e 10. Valida exit code 0 e 2 e o formato estruturado das linhas de log."""
    saida_dir, estado_dir, logs_dir = _dirs(tmp_path)

    res_ok = executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
    )
    assert res_ok.codigo_saida == 0

    res_falha = executar_rpa(
        caminho_fontes=FONTES_YAML,
        caminho_base=BASE_JSONL,
        amostras_dir=AMOSTRAS_DIR,
        saida_dir=saida_dir,
        estado_dir=estado_dir,
        logs_dir=logs_dir,
        offline=True,
        simular_falha={"enel-emergencia"},
        sleep_fn=lambda _: None,
    )
    assert res_falha.codigo_saida == 2
    assert len(res_falha.relatorio["excecoes_sistema"]) == 1

    log_path = Path(res_falha.arquivo_log)
    assert log_path.name.startswith("rpa_normativo_") and log_path.name.endswith(".log")
    linhas = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(linhas) >= 10

    etapas_encontradas = set()
    for ln in linhas:
        assert "exec=" in ln
        assert "fonte=" in ln
        assert "etapa=" in ln
        assert "status=" in ln
        for token in ln.split():
            if token.startswith("etapa="):
                etapas_encontradas.add(token.split("=", 1)[1])

    assert {"inicio", "download", "extracao", "decisao", "gravacao", "fim"}.issubset(
        etapas_encontradas
    )
