"""Utilitarios de privacidade (LGPD) para mascaramento de dados pessoais no assistente."""

from __future__ import annotations

import re


# Ordem intencional: e-mail -> CNPJ -> telefone -> CPF -> placa, evitando colisoes
# entre numeros de 14/11/10 digitos (CNPJ, celular sem mascara e CPF sem pontuacao).
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CNPJ_RE = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_PLACA_RE = re.compile(r"\b[A-Za-z]{3}-?\d[A-Za-z0-9]\d{2}\b", re.IGNORECASE)
_TELEFONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?55\s*)?"
    r"(?:"
    r"\(\s*0?[1-9]{2}\s*\)\s*(?:9\d{4}|\d{4})[\s.-]?\d{4}"
    r"|"
    r"\b[1-9]{2}[\s.-]+(?:9\d{4}|\d{4})[\s.-]?\d{4}\b"
    r"|"
    r"\b[1-9]{2}9\d{8}\b"
    r"|"
    r"\b[1-9]{2}[2-5]\d{7}\b"
    r")"
    r"(?!\d)"
)

_PRIVACY_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", _EMAIL_RE, "[e-mail omitido]"),
    ("cnpj", _CNPJ_RE, "[CNPJ omitido]"),
    ("telefone", _TELEFONE_RE, "[telefone omitido]"),
    ("cpf", _CPF_RE, "[CPF omitido]"),
    ("placa", _PLACA_RE, "[placa omitida]"),
)


def mask_personal_data(text: str) -> tuple[str, list[str]]:
    """Mascara dados pessoais (CPF, CNPJ, e-mail, placa e telefone) no texto informado."""
    if not text:
        return text, []

    masked = text
    found: set[str] = set()
    for category, pattern, replacement in _PRIVACY_RULES:
        masked, count = pattern.subn(replacement, masked)
        if count > 0:
            found.add(category)

    return masked, sorted(found)
