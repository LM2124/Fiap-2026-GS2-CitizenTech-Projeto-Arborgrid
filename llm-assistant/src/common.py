"""Constantes compartilhadas, contrato JSON, templates de prompt e leitura/escrita."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DIR_ROOT = Path(__file__).resolve().parent.parent
DIR_BASE = DIR_ROOT / "base"
DIR_DATA = DIR_ROOT / "data"
DIR_ADAPTERS = DIR_ROOT / "adapters"
DIR_EVAL = DIR_ROOT / "avaliacao"
DIR_ASSETS = DIR_ROOT / "assets"

REPO_ROOT = DIR_ROOT.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "arborgrid-vision" / "dataset" / "sp_geosampa_api" / "manifest.json"

BASE_MODEL = "Qwen/Qwen3-1.7B"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_RETRIEVAL_THRESHOLD = 0.30

RESPONSE_FIELDS = (
    "status",
    "nivel_risco",
    "orgao_responsavel",
    "telefone_ou_canal",
    "pode_podar_por_conta",
    "acao_imediata",
    "fontes",
)

ALLOWED_STATUS = {"respondido", "nao_consta"}
ALLOWED_RISK = {
    "emergencia_imediata",
    "urgente_vistoria",
    "preventivo_rotina",
    "fora_de_escopo",
}

SYSTEM_PROMPT = """Você é o ArborGrid Copiloto Técnico, especialista em triagem operacional de conflitos entre arborização urbana e rede elétrica em São Paulo.
Responda EXCLUSIVAMENTE com um objeto JSON válido contendo exatamente estas 7 chaves:
- "status": "respondido" ou "nao_consta"
- "nivel_risco": "emergencia_imediata", "urgente_vistoria", "preventivo_rotina" ou "fora_de_escopo"
- "orgao_responsavel": órgão ou responsável competente (ou "não consta")
- "telefone_ou_canal": telefone/canal oficial exato (ou "não consta")
- "pode_podar_por_conta": true, false ou null
- "acao_imediata": instrução objetiva e segura em português baseada somente nos trechos e dados fornecidos
- "fontes": lista com os chunk_id exatamente como aparecem no contexto utilizado (ou [] se "nao_consta")

Regras estritas:
1. Baseie-se APENAS nos trechos de [CONTEXTO] e, se houver, em [DADOS_TECNICOS_ARBORGRID]. Nunca invente leis, telefones, preços, prazos ou citações.
2. Se não houver trechos suficientes no contexto ou se a pergunta estiver fora do escopo de arborização/rede elétrica de São Paulo, responda obrigatoriamente com "status": "nao_consta", "nivel_risco": "fora_de_escopo", "orgao_responsavel": "não consta", "telefone_ou_canal": "não consta", "pode_podar_por_conta": null e "fontes": [].
3. Nunca repita nem armazene dados pessoais sensíveis (CPF, nome completo ou placa de veículo) eventualmente mencionados pelo usuário (LGPD).
4. Não escreva markdown, blocos ```json``` nem texto fora do objeto JSON."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Carrega registros de um arquivo JSONL UTF-8."""
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if stripped:
                items.append(json.loads(stripped))
    return items


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Salva uma lista de dicionários em formato JSONL UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_context_block(chunks: list[dict[str, Any]]) -> str:
    """Formata trechos recuperados com ID e título para injeção no prompt."""
    if not chunks:
        return "[CONTEXTO]\nNenhum trecho acima do limiar mínimo de confiança foi encontrado."
    lines = ["[CONTEXTO]"]
    for item in chunks:
        score_str = f" | score={item['score']:.3f}" if "score" in item else ""
        lines.append(
            f"- [{item['chunk_id']}] ({item['fonte']}{score_str}): {item['conteudo']}"
        )
    return "\n".join(lines)


def build_user_prompt(
    question: str,
    chunks: list[dict[str, Any]] | None = None,
    tree_context: str | None = None,
) -> str:
    """Monta a mensagem do usuário para zero-shot (chunks=None) ou RAG/Agente."""
    parts: list[str] = []
    if chunks is not None:
        parts.append(format_context_block(chunks))
    if tree_context:
        parts.append(f"[DADOS_TECNICOS_ARBORGRID]\n{tree_context}")
    parts.append(f"[PERGUNTA]\n{question.strip()}")
    return "\n\n".join(parts)


def format_chat_prompt(
    tokenizer: Any,
    question: str,
    chunks: list[dict[str, Any]] | None = None,
    tree_context: str | None = None,
) -> str:
    """Aplica o chat template do Qwen3 com thinking desativado para baixa latência."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(question=question, chunks=chunks, tree_context=tree_context),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_json_object(raw_text: str) -> tuple[dict[str, Any] | None, bool]:
    """Extrai objeto JSON da resposta e informa se a saída foi estritamente pura."""
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, True
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed, False
        except json.JSONDecodeError:
            pass
    return None, False
