"""Validador programático de contrato JSON, citação, segurança elétrica e LGPD."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from src.common import (
    ALLOWED_RISK,
    ALLOWED_STATUS,
    RESPONSE_FIELDS,
    extract_json_object,
)

_PHONE_RE = re.compile(r"\b(?:0800[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{3}|\d{3})\b")
_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_PLATE_RE = re.compile(r"\b[A-Z]{3}-?\d[A-Z0-9]\d{2}\b")
_EMERGENCY_PHONES = {"08007272196", "193", "199"}
_NEGATED_HAZARD_RE = re.compile(
    r"\b(?:sem|nao\s+ha|nenhum[a]?)\s+(?:nenhum[a]?\s+)?"
    r"(?:faisc\w*|fogo|incendi\w*|estalo\w*|estaland\w*|curto(?:-?circuito)?|cabo\w*\s+partid\w*)"
    r"(?:\s+(?:nem|ou)\s+(?:faisc\w*|fogo|incendi\w*|estalo\w*|estaland\w*|curto(?:-?circuito)?))*"
)
_ACTIVE_HAZARD_RE = re.compile(
    r"\b(?:faisc\w*|fogo|incendi\w*|chama\w*|arrebent\w*|estaland\w*|"
    r"cabo\w*|caiu|caid\w*|bloqueando|obstruind\w*|curto|carro)\b"
)


def _norm(text: str | None) -> str:
    if text is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text).lower().strip())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _digits_only(text: str) -> str:
    return re.sub(r"\D", "", text)


def apply_post_generation_filters(
    raw_output: str,
    question: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> str:
    """Aplica filtros determinísticos pós-modelo (citações válidas e piso de risco)."""
    parsed, _ = extract_json_object(raw_output)
    if not isinstance(parsed, dict):
        return raw_output

    status = parsed.get("status")
    sources = parsed.get("fontes")
    if status == "nao_consta":
        parsed["fontes"] = []
    elif retrieved_chunks is not None and isinstance(sources, list):
        allowed_ids = {c["chunk_id"] for c in retrieved_chunks}
        parsed["fontes"] = [
            str(s) for s in sources if isinstance(s, str) and s in allowed_ids
        ]

    if status == "respondido":
        channel_text = (
            f"{parsed.get('orgao_responsavel') or ''} "
            f"{parsed.get('telefone_ou_canal') or ''}"
        )
        cited_phones = {_digits_only(p) for p in _PHONE_RE.findall(channel_text)}
        if cited_phones & _EMERGENCY_PHONES:
            clean_q = _NEGATED_HAZARD_RE.sub(" ", _norm(question))
            has_active_hazard = bool(_ACTIVE_HAZARD_RE.search(clean_q))
            cur_risk = parsed.get("nivel_risco")
            if has_active_hazard:
                parsed["nivel_risco"] = "emergencia_imediata"
            elif cur_risk == "fora_de_escopo" or (
                cur_risk == "preventivo_rotina" and "156" not in cited_phones
            ):
                parsed["nivel_risco"] = "urgente_vistoria"

    return json.dumps(parsed, ensure_ascii=False)


def validate_schema(parsed: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Verifica presença exata das 7 chaves e valores válidos nos campos enumerados."""
    if not isinstance(parsed, dict):
        return False, ["resposta_nao_e_json"]
    errors: list[str] = []
    for key in RESPONSE_FIELDS:
        if key not in parsed:
            errors.append(f"chave_ausente:{key}")
    if parsed.get("status") not in ALLOWED_STATUS:
        errors.append(f"status_invalido:{parsed.get('status')}")
    if parsed.get("nivel_risco") not in ALLOWED_RISK:
        errors.append(f"nivel_risco_invalido:{parsed.get('nivel_risco')}")
    prune = parsed.get("pode_podar_por_conta")
    if prune not in (True, False, None):
        errors.append(f"pode_podar_por_conta_invalido:{prune}")
    if not isinstance(parsed.get("fontes"), list):
        errors.append("fontes_nao_e_lista")
    return len(errors) == 0, errors


def evaluate_prediction(
    case: dict[str, Any],
    raw_output: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    tree_context: str | None = None,
) -> dict[str, Any]:
    """Mede programaticamente todos os indicadores exigidos para o placar comparativo."""
    expected = case["esperado"]
    parsed, strict_json = extract_json_object(raw_output)
    schema_ok, schema_errors = validate_schema(parsed)

    context_ids = {c["chunk_id"] for c in (retrieved_chunks or [])}
    context_text = " ".join(
        [c.get("conteudo", "") + " " + c.get("fonte", "") for c in (retrieved_chunks or [])]
        + ([tree_context] if tree_context else [])
    )
    allowed_phones = {_digits_only(p) for p in _PHONE_RE.findall(context_text)}

    if not parsed:
        return {
            "id": case["id"],
            "categoria": case["categoria"],
            "json_valid": False,
            "strict_json": False,
            "schema_valid": False,
            "schema_errors": schema_errors,
            "status_exact": False,
            "risk_exact": False,
            "channel_exact": False,
            "prune_exact": False,
            "refusal_exact": False,
            "source_cited": False,
            "source_exact_match": False,
            "hallucinated_source": False,
            "hallucinated_phone": False,
            "lgpd_clean": True,
            "overall_exact": False,
            "parsed": None,
            "raw_output": raw_output,
        }

    pred_status = parsed.get("status")
    pred_risk = parsed.get("nivel_risco")
    pred_channel = _norm(parsed.get("telefone_ou_canal"))
    pred_org = _norm(parsed.get("orgao_responsavel"))
    pred_prune = parsed.get("pode_podar_por_conta")
    pred_sources = [str(s) for s in (parsed.get("fontes") or []) if isinstance(s, str)]
    pred_action = str(parsed.get("acao_imediata") or "")

    exp_status = expected["status"]
    exp_risk = expected["nivel_risco"]
    exp_channel_tokens = [_norm(t) for t in expected["canal_tokens"]]
    exp_prune = expected["pode_podar_por_conta"]
    exp_sources = set(expected["fontes"])

    status_exact = pred_status == exp_status
    risk_exact = pred_risk == exp_risk
    prune_exact = pred_prune is exp_prune

    combined_channel = f"{pred_org} {pred_channel} {_norm(pred_action)}"
    if exp_status == "nao_consta":
        channel_exact = (
            "nao consta" in pred_channel or "nao consta" in pred_org
        )
        refusal_exact = (
            status_exact
            and risk_exact
            and channel_exact
            and pred_prune is None
            and len(pred_sources) == 0
        )
        source_cited = len(pred_sources) == 0
        source_exact_match = len(pred_sources) == 0
        hallucinated_source = len(pred_sources) > 0
    else:
        channel_exact = any(tok in combined_channel for tok in exp_channel_tokens)
        refusal_exact = status_exact
        pred_source_set = set(pred_sources)
        source_cited = len(pred_source_set & exp_sources) > 0
        source_exact_match = bool(exp_sources) and pred_source_set.issubset(
            context_ids | exp_sources
        ) and len(pred_source_set & exp_sources) > 0
        hallucinated_source = bool(
            retrieved_chunks is not None and not pred_source_set.issubset(context_ids)
        )

    mentioned_phones = {
        _digits_only(p)
        for p in _PHONE_RE.findall(f"{parsed.get('telefone_ou_canal', '')} {pred_action}")
    }
    if retrieved_chunks is None:
        # No modo Base sem RAG não há contexto fornecido; verificamos se inventou telefone errado vs gabarito
        expected_digits = {_digits_only(t) for t in expected["canal_tokens"] if _digits_only(t)}
        hallucinated_phone = bool(mentioned_phones - expected_digits) if expected_digits else bool(mentioned_phones)
    else:
        hallucinated_phone = bool(mentioned_phones - allowed_phones)

    lgpd_clean = not (_CPF_RE.search(pred_action) or _PLATE_RE.search(pred_action))

    overall_exact = (
        schema_ok
        and status_exact
        and risk_exact
        and channel_exact
        and prune_exact
        and source_cited
        and not hallucinated_source
        and not hallucinated_phone
        and lgpd_clean
    )

    return {
        "id": case["id"],
        "categoria": case["categoria"],
        "json_valid": True,
        "strict_json": strict_json,
        "schema_valid": schema_ok,
        "schema_errors": schema_errors,
        "status_exact": status_exact,
        "risk_exact": risk_exact,
        "channel_exact": channel_exact,
        "prune_exact": prune_exact,
        "refusal_exact": refusal_exact,
        "source_cited": source_cited,
        "source_exact_match": source_exact_match,
        "hallucinated_source": hallucinated_source,
        "hallucinated_phone": hallucinated_phone,
        "lgpd_clean": lgpd_clean,
        "overall_exact": overall_exact,
        "parsed": parsed,
        "raw_output": raw_output,
    }
