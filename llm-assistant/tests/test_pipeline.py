"""Testes determinísticos do validador, ferramenta GeoSampa e esquema JSON."""

from __future__ import annotations

import json

from src.common import DEFAULT_MANIFEST_PATH, extract_json_object
from src.tools import ArborGridInspector
from src.validator import (
    apply_post_generation_filters,
    evaluate_prediction,
    validate_schema,
)


def test_extract_json_strict_and_fenced() -> None:
    payload = {
        "status": "respondido",
        "nivel_risco": "emergencia_imediata",
        "orgao_responsavel": "Enel SP Emergência",
        "telefone_ou_canal": "0800 72 72 196",
        "pode_podar_por_conta": False,
        "acao_imediata": "Acione a Enel Emergência imediatamente.",
        "fontes": ["ENEL-EMERG-01"],
    }
    parsed, strict = extract_json_object(json.dumps(payload, ensure_ascii=False))
    assert strict is True
    assert parsed == payload

    fenced = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    parsed2, strict2 = extract_json_object(fenced)
    assert strict2 is False
    ok, errors = validate_schema(parsed2)
    assert ok and errors == []


def test_evaluate_prediction_detects_phone_hallucination_and_lgpd() -> None:
    case = {
        "id": "T_DUMMY",
        "categoria": "emergencia_eletrica",
        "pergunta": "Árvore soltando faísca na fiação.",
        "esperado": {
            "status": "respondido",
            "nivel_risco": "emergencia_imediata",
            "canal_tokens": ["0800 72 72 196"],
            "pode_podar_por_conta": False,
            "fontes": ["ENEL-EMERG-01"],
        },
    }
    chunks = [
        {
            "chunk_id": "ENEL-EMERG-01",
            "fonte": "Enel SP",
            "conteudo": "Ligue para 0800 72 72 196 em emergências.",
        }
    ]
    bad_output = json.dumps(
        {
            "status": "respondido",
            "nivel_risco": "emergencia_imediata",
            "orgao_responsavel": "Enel SP",
            "telefone_ou_canal": "0800 99 99 999",
            "pode_podar_por_conta": False,
            "acao_imediata": "CPF 123.456.789-00 registrado.",
            "fontes": ["ENEL-EMERG-01"],
        },
        ensure_ascii=False,
    )
    metrics = evaluate_prediction(case, bad_output, retrieved_chunks=chunks)
    assert metrics["hallucinated_phone"] is True
    assert metrics["lgpd_clean"] is False
    assert metrics["overall_exact"] is False


def test_geosampa_inspector_real_tree_lookup() -> None:
    if not DEFAULT_MANIFEST_PATH.exists():
        return
    inspector = ArborGridInspector()
    ctx, rec = inspector.maybe_build_tool_context("Verifique a árvore GeoSampa 629339")
    assert rec is not None and ctx is not None
    assert rec["geosampa_id"] == "629339"
    assert rec["triage_v3"] == "DIVERGENCIA_PARALAXE"

    ctx2, rec2 = inspector.maybe_build_tool_context("Verifique a árvore GeoSampa 7457")
    assert rec2 is not None and ctx2 is not None
    assert rec2["triage_v3"] == "CRITICO_CONFIRMADO"


def test_post_generation_filters_sanitize_sources_and_risk_floor() -> None:
    chunks = [{"chunk_id": "NBR-16246-2"}, {"chunk_id": "ENEL-EMERG-01"}]
    raw_active = json.dumps(
        {
            "status": "respondido",
            "nivel_risco": "urgente_vistoria",
            "orgao_responsavel": "Enel SP",
            "telefone_ou_canal": "0800 72 72 196",
            "pode_podar_por_conta": False,
            "acao_imediata": "Afaste-se do fio partido.",
            "fontes": ["ENEL-EMERG-01", "NR-10"],
        },
        ensure_ascii=False,
    )
    filtered_active = json.loads(
        apply_post_generation_filters(raw_active, "Cabo arrebentou e caiu na rua.", chunks)
    )
    assert filtered_active["nivel_risco"] == "emergencia_imediata"
    assert filtered_active["fontes"] == ["ENEL-EMERG-01"]

    raw_preventive = json.dumps(
        {
            "status": "respondido",
            "nivel_risco": "preventivo_rotina",
            "orgao_responsavel": "Enel SP",
            "telefone_ou_canal": "0800 72 72 196",
            "pode_podar_por_conta": False,
            "acao_imediata": "Não se aproxime da rede elétrica.",
            "fontes": ["NBR-16246-2", "NR-10"],
        },
        ensure_ascii=False,
    )
    filtered_prev = json.loads(
        apply_post_generation_filters(raw_preventive, "Posso usar escada perto da rede?", chunks)
    )
    assert filtered_prev["nivel_risco"] == "urgente_vistoria"
    assert filtered_prev["fontes"] == ["NBR-16246-2"]

    raw_negated = json.dumps(
        {
            "status": "respondido",
            "nivel_risco": "preventivo_rotina",
            "orgao_responsavel": "SP156 (156) ou Enel Emergência (0800 72 72 196)",
            "telefone_ou_canal": "SP156 (156) ou 0800 72 72 196",
            "pode_podar_por_conta": False,
            "acao_imediata": "Abra protocolo SP156 sem acionar emergência elétrica.",
            "fontes": ["ENEL-EMERG-01"],
        },
        ensure_ascii=False,
    )
    filtered_neg = json.loads(
        apply_post_generation_filters(
            raw_negated,
            "Galho encostando sem faísca nem estalo e não há curto-circuito.",
            chunks,
        )
    )
    assert filtered_neg["nivel_risco"] == "preventivo_rotina"
