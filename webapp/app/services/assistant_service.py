"""Serviço do Assistente Técnico Normativo ArborGrid (RAG híbrido leve + ferramenta GeoSampa + Qwen3 GGUF)."""

from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from ..metrics import ASSISTANT_RESPONSES_TOTAL, LLM_FALLBACK_TOTAL
from ..privacy import mask_personal_data
from .gold_store import GoldRiskStore
from .manifest_store import (
    ManifestStore,
    _normalize_macro_triage,
    _tree_raw_consolidated_triage,
)


logger = logging.getLogger(__name__)


_TREE_ID_RE = re.compile(
    r"(?:(?:arvore|geosampa|id)[_\s:#-]*|#)(\d{1,7})\b|\b(\d{4,7})\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_PT_STOPWORDS = frozenset(
    {
        "de",
        "da",
        "do",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "com",
        "sem",
        "por",
        "para",
        "um",
        "uma",
        "uns",
        "umas",
        "que",
        "qual",
        "quais",
        "quem",
        "como",
        "sobre",
        "esta",
        "este",
        "esse",
        "essa",
        "ao",
        "aos",
        "ou",
        "se",
        "ser",
        "tem",
        "seu",
        "sua",
        "voce",
        "eu",
        "meu",
        "minha",
        "aqui",
        "ali",
        "la",
        "fazer",
    }
)
_TOKEN_CANONICAL: dict[str, tuple[str, ...]] = {
    "podar": ("poda",),
    "podo": ("poda",),
    "podas": ("poda",),
    "podada": ("poda",),
    "podado": ("poda",),
    "podador": ("poda",),
    "cortar": ("poda", "supressao"),
    "corte": ("poda", "supressao"),
    "cortes": ("poda", "supressao"),
    "corta": ("poda", "supressao"),
    "derrubar": ("supressao", "queda"),
    "derrubada": ("supressao", "queda"),
    "remover": ("remocao", "supressao"),
    "remocao": ("remocao", "supressao"),
    "arvores": ("arvore",),
    "galhos": ("galho",),
    "ramos": ("ramo", "galho"),
    "ramo": ("ramo", "galho"),
    "fios": ("fio", "fiacao"),
    "fio": ("fio", "fiacao"),
    "cabos": ("cabo", "fiacao"),
    "cabo": ("cabo", "fiacao"),
    "rede": ("rede", "fiacao"),
    "eletricos": ("eletrica",),
    "eletrico": ("eletrica",),
    "eletricas": ("eletrica",),
    "energia": ("eletrica", "fiacao"),
    "encostando": ("contato", "tocando"),
    "encosta": ("contato", "tocando"),
    "encostar": ("contato", "tocando"),
    "tocando": ("contato", "tocando"),
    "toca": ("contato", "tocando"),
    "tocar": ("contato", "tocando"),
    "faiscando": ("faisca",),
    "faiscas": ("faisca",),
    "ventania": ("vento",),
    "ventos": ("vento",),
    "temporal": ("vento", "chuva"),
    "temporais": ("vento", "chuva"),
    "chuvas": ("chuva",),
    "caiu": ("queda", "caida"),
    "cair": ("queda",),
    "caindo": ("queda",),
    "caido": ("queda", "caida"),
    "calcadas": ("calcada",),
    "multas": ("multa",),
    "multado": ("multa",),
    "normas": ("norma",),
    "leis": ("lei",),
    "vistorias": ("vistoria",),
    "particulares": ("particular", "privado"),
    "privada": ("particular", "privado"),
    "privado": ("particular", "privado"),
    "quintal": ("particular", "privado"),
}
_EMERGENCY_TERMS = (
    "faisca",
    "fogo",
    "incendio",
    "chama",
    "cabo caido",
    "fio rompido",
    "fio partido",
    "estalo",
    "curto",
    "ferid",
    "preso",
    "desab",
    "caiu",
)
# Negacoes ("sem faisca", "nao tem faisca", "nao ha faiscas nem curto-circuito", "nao tem ninguem ferido")
# nao contam como sinal de emergencia. Ate 2 palavras entre a negacao e o termo; um termo sem negacao
# na mesma frase ("nao tem faisca, mas saiu fogo") continua valendo.
_NEGATED_EMERGENCY_RE = re.compile(
    r"\b(?:sem|nem|nenhum[a]?|nao\s+(?:ha|tem|teve|houve|existe|existem|vi|vejo|esta|estao))\s+"
    r"(?:\w+\s+){0,2}?"
    r"(?:faisc\w*|fogo|incendi\w*|chama\w*|estalo\w*|curto\w*|cabo\w*\s+(?:caid\w*|rompid\w*)"
    r"|fio\w*\s+(?:caid\w*|partid\w*|rompid\w*)|ferid\w*|pres[oa]s?)"
)

_TRIAGE_HUMAN_LABELS: dict[str, str] = {
    "CRITICO_CONFIRMADO": "Risco Crítico (confirmado nos dois ângulos)",
    "DIVERGENCIA_PARALAXE": "Risco Crítico (visível ao mudar o ângulo)",
    "CRITICO_MONO_VISADA": "Risco Crítico (foto única)",
    "CRITICO": "Risco Crítico",
    "PROXIMIDADE_MONO_VISADA": "Atenção (proximidade ao vento, foto única)",
    "PROXIMIDADE": "Atenção (proximidade ao vento)",
    "SEGURO_MONO_VISADA": "Sem conflito aparente (foto única)",
    "SEGURO": "Sem conflito aparente",
}
_PRIORITY_HUMAN_LABELS: dict[str, str] = {
    "URGENTE": "Urgente",
    "ALTA": "Alta",
    "MEDIA": "Média",
    "BAIXA": "Baixa",
}
_ALLOWED_STATUS = {"respondido", "nao_consta"}
_ALLOWED_RISK = {
    "emergencia_imediata",
    "urgente_vistoria",
    "preventivo_rotina",
    "fora_de_escopo",
}

SYSTEM_PROMPT = """Você é o ArborGrid Assistente Técnico, especialista em triagem operacional de conflitos entre arborização urbana e rede elétrica em São Paulo.
Responda EXCLUSIVAMENTE com um objeto JSON válido contendo exatamente estas 7 chaves:
- "status": "respondido" ou "nao_consta"
- "nivel_risco": "emergencia_imediata", "urgente_vistoria", "preventivo_rotina" ou "fora_de_escopo"
- "orgao_responsavel": órgão ou responsável competente em português (ou "não consta")
- "telefone_ou_canal": telefone/canal oficial exato (ou "não consta")
- "pode_podar_por_conta": true, false ou null
- "acao_imediata": instrução objetiva e segura em português com acentuação correta baseada somente nos trechos e dados fornecidos, sem códigos internos em caixa alta
- "fontes": lista com os chunk_id exatamente como aparecem no contexto utilizado (ou [] se "nao_consta")

Regras estritas:
1. Baseie-se APENAS nos trechos de [CONTEXTO] e, se houver, em [DADOS_TECNICOS_ARBORGRID]. Nunca invente leis, telefones, preços, prazos ou canais.
2. Se não houver trechos suficientes no contexto ou se a pergunta estiver fora do escopo de arborização/rede elétrica de São Paulo, responda obrigatoriamente com "status": "nao_consta", "nivel_risco": "fora_de_escopo", "orgao_responsavel": "não consta", "telefone_ou_canal": "não consta", "pode_podar_por_conta": null e "fontes": [].
3. Nunca repita nem armazene dados pessoais sensíveis (CPF, nome completo ou placa de veículo) eventualmente mencionados pelo usuário (LGPD).
4. Não escreva markdown, blocos ```json``` nem texto fora do objeto JSON."""


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    base_tokens = [
        tok
        for tok in _TOKEN_RE.findall(_normalize_text(text))
        if tok not in _PT_STOPWORDS
    ]
    expanded: list[str] = []
    for tok in base_tokens:
        canon = _TOKEN_CANONICAL.get(tok)
        if canon:
            expanded.extend(canon)
        else:
            expanded.append(tok)
    bigrams = [f"{a}_{b}" for a, b in zip(expanded, expanded[1:])]
    return expanded + bigrams


def _humanize_triage(code: str | None) -> str:
    key = (code or "").strip().upper()
    return _TRIAGE_HUMAN_LABELS.get(key, "Sem conflito aparente")


def _humanize_priority(tier: str | None) -> str:
    key = (tier or "").strip().upper()
    return _PRIORITY_HUMAN_LABELS.get(key, key.title() if key else "Média")


def _sanitize_internal_codes(text: str) -> str:
    """Substitui eventuais códigos internos em caixa alta por português legível."""
    cleaned = text
    for code, label in _TRIAGE_HUMAN_LABELS.items():
        cleaned = re.sub(rf"(?<![A-Z0-9_-]){re.escape(code)}(?![A-Z0-9_-])", label, cleaned)
    for tier, label in _PRIORITY_HUMAN_LABELS.items():
        cleaned = re.sub(rf"(?<![A-Z0-9_-]){re.escape(tier)}(?![A-Z0-9_-])", label, cleaned)
    return cleaned


class AssistantService:
    """Executa busca RAG normativa + consulta determinística GeoSampa + Qwen3 GGUF (com fallback)."""

    SUGGESTED_PROMPTS = [
        "Árvore #635210 em Pinheiros está encostando na fiação sem faíscas. O que fazer?",
        "Qual o procedimento para poda próxima à rede elétrica segundo a NBR 16246-2 e SVMA?",
        "Fio partido faiscando após ventania derrubar galho em cima de um carro!",
        "Morador pode podar galho de árvore da calçada por conta própria na Lei 17.794?",
    ]

    def __init__(
        self,
        kb_path: Path,
        manifest_store: ManifestStore,
        gold_store: GoldRiskStore,
        llm_api_url: str = "http://127.0.0.1:8080/v1/chat/completions",
        llm_model_name: str = "qwen3-1.7b-arborgrid",
        llm_timeout_seconds: float = 110.0,
    ) -> None:
        self.kb_path = Path(kb_path)
        self.manifest_store = manifest_store
        self.gold_store = gold_store
        self.llm_api_url = llm_api_url
        self.llm_model_name = llm_model_name
        self.llm_timeout_seconds = llm_timeout_seconds
        self.chunks: list[dict[str, Any]] = []
        self._idf: dict[str, float] = {}
        self._chunk_vecs: list[dict[str, float]] = []
        self._load_kb()

    def _load_kb(self) -> None:
        if not self.kb_path.exists():
            self.chunks = []
            return
        rows: list[dict[str, Any]] = []
        with self.kb_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        self.chunks = rows
        n_docs = max(len(rows), 1)
        doc_tokens: list[list[str]] = []
        df: Counter[str] = Counter()
        for row in rows:
            raw_doc = (
                f"{row.get('chunk_id', '')} {row.get('titulo', '')} "
                f"{row.get('fonte', '')} {row.get('categoria', '')} {row.get('conteudo', '')}"
            )
            toks = _tokenize(raw_doc)
            doc_tokens.append(toks)
            for term in set(toks):
                df[term] += 1

        self._idf = {
            term: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0
            for term, freq in df.items()
        }
        self._chunk_vecs = []
        for toks in doc_tokens:
            self._chunk_vecs.append(self._to_tfidf_vec(toks))

    def _to_tfidf_vec(self, toks: list[str]) -> dict[str, float]:
        counts = Counter(toks)
        total = max(len(toks), 1)
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for term, cnt in counts.items():
            idf = self._idf.get(term, 1.0)
            tf = (1.0 + math.log(cnt)) if cnt > 0 else 0.0
            val = (tf / math.log(1.0 + total)) * idf
            vec[term] = val
            norm_sq += val * val
        norm = math.sqrt(norm_sq) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        q_toks = _tokenize(query)
        q_vec = self._to_tfidf_vec(q_toks)
        q_unigrams = {t for t in q_toks if "_" not in t}
        scored: list[dict[str, Any]] = []
        for idx, chunk in enumerate(self.chunks):
            c_vec = self._chunk_vecs[idx]
            cosine = sum(val * c_vec.get(term, 0.0) for term, val in q_vec.items())
            c_unigrams = {t for t in c_vec if "_" not in t}
            overlap = len(q_unigrams & c_unigrams) / max(len(q_unigrams), 1)
            score = round(0.75 * cosine + 0.25 * overlap, 4)
            item = dict(chunk)
            item["score"] = score
            scored.append(item)
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def _lookup_tree_context(
        self, question: str, explicit_geosampa_id: str | None = None
    ) -> dict[str, Any] | None:
        candidate_id = explicit_geosampa_id.strip() if explicit_geosampa_id else None
        if candidate_id and candidate_id not in self.manifest_store.tree_by_id:
            candidate_id = None

        if not candidate_id:
            for match in _TREE_ID_RE.finditer(question):
                cand = match.group(1) or match.group(2)
                if cand and cand in self.manifest_store.tree_by_id:
                    candidate_id = cand
                    break

        if not candidate_id:
            return None

        tree = self.manifest_store.tree_by_id[candidate_id]
        gold = self.gold_store.records.get(candidate_id)
        views_summary = []
        max_contact = 0
        max_wind = 0
        max_wire = 0
        max_copa = 0
        for view in tree.get("views", []):
            v3 = view.get("v3_analysis") or {}
            c_px = int(v3.get("contact_direct_px") or 0)
            w_px = int(v3.get("wind_buffer_px") or 0)
            wr_px = int(v3.get("wire_px") or 0)
            cp_px = int(v3.get("copa_px") or 0)
            max_contact = max(max_contact, c_px)
            max_wind = max(max_wind, w_px)
            max_wire = max(max_wire, wr_px)
            max_copa = max(max_copa, cp_px)
            views_summary.append(
                {
                    "perspective": view.get("perspective"),
                    "capture_date": view.get("capture_date"),
                    "triage": v3.get("triage"),
                    "contact_direct_px": c_px,
                    "wind_buffer_px": w_px,
                    "wire_px": wr_px,
                    "copa_px": cp_px,
                }
            )

        raw_consolidated = _tree_raw_consolidated_triage(tree)
        macro_triage = _normalize_macro_triage(raw_consolidated)
        contact_copa_pct = round(min(100.0, 100.0 * max_contact / max_copa), 1) if max_copa > 0 else 0.0
        wind_copa_pct = round(min(100.0, 100.0 * max_wind / max_copa), 1) if max_copa > 0 else 0.0
        wire_frame_pct = round(min(100.0, 100.0 * max_wire / (640 * 640)), 1)
        return {
            "geosampa_id": candidate_id,
            "distrito": tree.get("distrito"),
            "subprefeitura": tree.get("subprefeitura"),
            "region_key": tree.get("region_key"),
            "region_label": tree.get("region_label"),
            "lat": round(float(tree.get("tree_lat", 0.0)), 5),
            "lon": round(float(tree.get("tree_lon", 0.0)), 5),
            "sisgau_last_inspection": tree.get("sisgau_last_inspection"),
            "triage_macro": macro_triage,
            "triage_consolidated": raw_consolidated,
            "has_parallax_divergence": bool(
                (tree.get("v3_triage") or {}).get("has_paralax_divergence", False)
            ),
            "max_contact_direct_px": max_contact,
            "max_wind_buffer_px": max_wind,
            "max_wire_px": max_wire,
            "max_copa_px": max_copa,
            "contact_copa_pct": contact_copa_pct,
            "wind_copa_pct": wind_copa_pct,
            "wire_frame_pct": wire_frame_pct,
            "views": views_summary,
            "gold_risk": gold,
        }

    def check_llm_health(self, timeout_s: float = 0.6) -> bool:
        """Verifica rapidamente se o llama-server local (Qwen3 GGUF) esta online."""
        base_url = self.llm_api_url.split("/v1/")[0].rstrip("/")
        health_url = f"{base_url}/health"
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return False

    def status(self) -> dict[str, Any]:
        return {
            "kb_loaded": len(self.chunks) > 0,
            "chunks_count": len(self.chunks),
            "llm_online": self.check_llm_health(),
            "model_name": self.llm_model_name,
            "suggested_prompts": self.SUGGESTED_PROMPTS,
        }

    def _record_llm_fallback(self, reason: str, detail: str) -> None:
        logger.warning("Fallback do LLM: reason=%s detalhe=%s", reason, detail)
        LLM_FALLBACK_TOTAL.labels(reason=reason).inc()

    def _out_of_scope_response(self) -> dict[str, Any]:
        return {
            "status": "nao_consta",
            "nivel_risco": "fora_de_escopo",
            "orgao_responsavel": "não consta",
            "telefone_ou_canal": "não consta",
            "pode_podar_por_conta": None,
            "acao_imediata": (
                "A consulta está fora do escopo normativo do ArborGrid SP (arborização urbana, "
                "manejo conforme Lei nº 17.794/2022, ABNT NBR 16246, procedimentos SP156/Enel e telemetria GeoSampa)."
            ),
            "fontes": [],
        }

    def _call_local_qwen(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        tree_ctx: dict[str, Any] | None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any] | None:
        context_lines = ["[CONTEXTO]"]
        for row in chunks:
            # Escala score lexical para a faixa de confiança >= 0.45 usada no treino SFT do Qwen3
            llm_score = min(0.95, max(float(row.get("score") or 0.0) * 2.2, 0.48))
            context_lines.append(
                f"- [{row['chunk_id']}] ({row['fonte']} | score={llm_score:.3f}): {row['conteudo']}"
            )
        parts = ["\n".join(context_lines)]
        if tree_ctx:
            human_triage = _humanize_triage(tree_ctx["triage_consolidated"])
            parts.append(
                "[DADOS_TECNICOS_ARBORGRID]\n"
                f"GeoSampa ID {tree_ctx['geosampa_id']} | Distrito {tree_ctx['distrito']} | "
                f"Subprefeitura {tree_ctx['subprefeitura']} | Triagem visual: {human_triage} | "
                f"contato_direto={tree_ctx.get('contact_copa_pct', 0.0)}% da copa | "
                f"zona_de_vento={tree_ctx.get('wind_copa_pct', 0.0)}% da copa | "
                f"fiacao_visivel={tree_ctx.get('wire_frame_pct', 0.0)}% da foto | "
                f"Última vistoria SISGAU: {tree_ctx['sisgau_last_inspection'] or 'não registrada'}."
            )
        prompt_question = question.strip()
        if history:
            recent = [
                t
                for t in history[-4:]
                if t.get("role") in ("user", "assistant") and str(t.get("content") or "").strip()
            ]
            if recent:
                hist_parts = []
                for turn in recent:
                    prefix = "Usuário" if turn["role"] == "user" else "Assistente"
                    snippet = str(turn["content"]).strip().replace("\n", " ")[:220]
                    hist_parts.append(f"{prefix}: {snippet}")
                prompt_question = (
                    f"Contexto anterior da conversa: {' | '.join(hist_parts)}\n"
                    f"Pergunta atual: {prompt_question}"
                )
        parts.append(f"[PERGUNTA]\n{prompt_question}")

        body = json.dumps(
            {
                "model": self.llm_model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n\n".join(parts)},
                ],
                "temperature": 0.0,
                "max_tokens": 280,
                "stop": ["<|im_end|>", "<|endoftext|>"],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.llm_api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.llm_timeout_seconds) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            self._record_llm_fallback("http_error", f"HTTP {exc.code}")
            return None
        except TimeoutError as exc:
            self._record_llm_fallback("timeout", str(exc) or "TimeoutError")
            return None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                self._record_llm_fallback("timeout", str(exc.reason) or "TimeoutError")
            else:
                self._record_llm_fallback("connection_error", str(exc.reason or exc))
            return None
        except OSError as exc:
            self._record_llm_fallback("connection_error", f"{type(exc).__name__}: {exc}")
            return None

        try:
            raw_resp = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._record_llm_fallback("invalid_json", f"{type(exc).__name__}: {exc}")
            return None

        try:
            content = str(raw_resp["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            self._record_llm_fallback(
                "invalid_schema", f"estrutura choices invalida: {type(exc).__name__}"
            )
            return None

        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            self._record_llm_fallback("invalid_json", "nenhum objeto JSON encontrado")
            return None

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            self._record_llm_fallback("invalid_json", f"JSONDecodeError: {exc.msg}")
            return None

        if not isinstance(parsed, dict):
            self._record_llm_fallback("invalid_schema", "payload JSON nao e objeto")
            return None

        status_val = str(parsed.get("status") or "").strip()
        acao = str(parsed.get("acao_imediata") or "").strip()
        if status_val not in _ALLOWED_STATUS or not acao:
            self._record_llm_fallback(
                "invalid_schema", "status invalido ou acao_imediata vazia"
            )
            return None
        if status_val == "nao_consta":
            return self._out_of_scope_response()
        acao_norm = _normalize_text(acao)
        if "nao consta na base" in acao_norm or "whatsapp" in acao_norm:
            self._record_llm_fallback("filtered", "termo proibido na resposta")
            return None

        allowed_ids = {c["chunk_id"] for c in chunks}
        raw_fontes = parsed.get("fontes")
        fontes = (
            [str(s) for s in raw_fontes if str(s) in allowed_ids]
            if isinstance(raw_fontes, list)
            else []
        )
        # O bloco [DADOS_TECNICOS_ARBORGRID] no prompt e derivado da especificacao ARBORGRID-TRIAGEM.
        if tree_ctx and "ARBORGRID-TRIAGEM" in allowed_ids and "ARBORGRID-TRIAGEM" not in fontes:
            fontes.append("ARBORGRID-TRIAGEM")
        if not fontes:
            self._record_llm_fallback("ungrounded", "nenhuma fonte valida citada")
            return None

        nivel_risco = str(parsed.get("nivel_risco") or "preventivo_rotina").strip()
        if nivel_risco not in _ALLOWED_RISK:
            nivel_risco = "preventivo_rotina"

        pode_podar = parsed.get("pode_podar_por_conta")
        if not isinstance(pode_podar, bool) and pode_podar is not None:
            pode_podar = None

        orgao_out = _sanitize_internal_codes(
            str(parsed.get("orgao_responsavel") or "Prefeitura de São Paulo (Subprefeituras / SVMA)")
        )
        canal_out = _sanitize_internal_codes(
            str(parsed.get("telefone_ou_canal") or "Portal SP156 (156 / sp156.prefeitura.sp.gov.br)")
        )
        acao_out = _sanitize_internal_codes(acao)

        # A classificacao de risco e a do modelo, sem regra por cima: o placar do llm-assistant mede
        # exatamente isso. Aqui so se validam schema, fontes citadas e codigos internos.

        if canal_out.count("(") > canal_out.count(")"):
            canal_out = canal_out + ")"

        return {
            "status": "respondido",
            "nivel_risco": nivel_risco,
            "orgao_responsavel": orgao_out,
            "telefone_ou_canal": canal_out,
            "pode_podar_por_conta": pode_podar,
            "acao_imediata": acao_out,
            "fontes": fontes,
        }

    def _synthesize_deterministic(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        tree_ctx: dict[str, Any] | None,
    ) -> dict[str, Any]:
        q_norm = _normalize_text(question)
        clean_q = _NEGATED_EMERGENCY_RE.sub(" ", q_norm)
        top_score = chunks[0]["score"] if chunks else 0.0

        # Out-of-scope guard when neither RAG nor GeoSampa tree matches
        if not tree_ctx and top_score < 0.12:
            return self._out_of_scope_response()

        has_emergency = any(term in clean_q for term in _EMERGENCY_TERMS)
        has_fire_or_injury = any(
            t in clean_q for t in ("fogo", "incendio", "chama", "ferid", "preso", "carro")
        )
        has_storm_fall = any(
            t in clean_q for t in ("caiu", "caida", "ventania", "tempestade", "desab")
        )
        asks_pruning = any(
            t in q_norm for t in ("podar", "poda", "cortar", "corte", "derrubar", "remover", "supress")
        )
        mentions_private = any(
            t in q_norm for t in ("particular", "quintal", "imovel interno", "meu terreno", "area privada")
        )

        if has_emergency:
            nivel_risco = "emergencia_imediata"
            if has_fire_or_injury:
                orgao = "Corpo de Bombeiros (193) + Enel Distribuição SP Emergência"
                canal = "193 (Bombeiros) / 0800 72 72 196 (Enel 24h)"
            elif has_storm_fall:
                orgao = "Defesa Civil SP + Enel Distribuição SP"
                canal = "199 (Defesa Civil) / 0800 72 72 196 (Enel 24h)"
            else:
                orgao = "Enel Distribuição São Paulo (Emergência 24h)"
                canal = "0800 72 72 196"
            pode_podar: bool | None = False
        elif tree_ctx and tree_ctx["triage_macro"] == "CRITICO":
            nivel_risco = "urgente_vistoria"
            orgao = f"Subprefeitura {tree_ctx['subprefeitura']} + Enel Distribuição SP"
            canal = "Portal/Telefone SP156 (156) + Enel (0800 72 72 196 se risco ativo)"
            pode_podar = False
        elif tree_ctx and tree_ctx["triage_macro"] == "PROXIMIDADE":
            nivel_risco = "preventivo_rotina"
            orgao = f"Subprefeitura {tree_ctx['subprefeitura']} / SVMA"
            canal = "Portal SP156 (156 / sp156.prefeitura.sp.gov.br)"
            pode_podar = False
        elif mentions_private and not ("fiacao" in q_norm or "fio" in q_norm or "rede" in q_norm):
            nivel_risco = "preventivo_rotina"
            orgao = "Proprietário do Imóvel + Subprefeitura Regional (comunicação prévia 10 dias úteis)"
            canal = "Portal SP156 / Subprefeitura (protocolo com ART/TRT)"
            pode_podar = True
        else:
            nivel_risco = "preventivo_rotina"
            orgao = "Prefeitura de São Paulo (Subprefeituras / SVMA)"
            canal = "Portal SP156 (156 / sp156.prefeitura.sp.gov.br)"
            pode_podar = False

        partes_acao: list[str] = []
        if tree_ctx:
            human_triage = _humanize_triage(tree_ctx["triage_consolidated"])
            partes_acao.append(
                f"Auditoria da árvore GeoSampa #{tree_ctx['geosampa_id']} ({tree_ctx['distrito']}): "
                f"{human_triage}, com {tree_ctx.get('contact_copa_pct', 0.0):.1f}% da copa em contato direto "
                f"com a fiação e {tree_ctx.get('wind_copa_pct', 0.0):.1f}% na margem de vento "
                f"(última vistoria SISGAU: {tree_ctx['sisgau_last_inspection'] or 'não registrada'})."
            )
            if tree_ctx.get("gold_risk"):
                g = tree_ctx["gold_risk"]
                tier_label = _humanize_priority(g.get("priority_tier"))
                partes_acao.append(
                    f"A priorização climática indica nível {tier_label} (índice {g['composite_risk_score']}/100): "
                    f"{g['recommended_action']}."
                )

        if has_emergency:
            partes_acao.append(
                "Mantenha isolamento imediato de pedestres, não se aproxime de cabos ou galhos energizados "
                "e acione os canais de emergência listados."
            )
        elif asks_pruning and not tree_ctx:
            if mentions_private:
                partes_acao.append(
                    "Em imóvel particular e sem contato com a rede elétrica, a Lei nº 17.794/2022 permite o manejo "
                    "somente mediante laudo técnico (ART/TRT) e comunicação prévia à Subprefeitura com 10 dias úteis de antecedência. "
                    "Se houver proximidade com fios elétricos, a intervenção exige profissionais habilitados sob NR-10 / ABNT NBR 16246-2 e apoio da concessionária."
                )
            else:
                partes_acao.append(
                    "Não é permitido podar ou cortar árvores em calçadas ou vias públicas por conta própria (Lei nº 17.794/2022, sujeito a multa). "
                    "Para árvores em área pública ou próximas à rede elétrica, solicite vistoria gratuita pelo Portal SP156 (telefone 156); "
                    "caso haja toque com faíscas ou fio partido, acione a Enel Emergência (0800 72 72 196). "
                    "Se a árvore estiver em terreno particular e longe dos fios, o manejo exige laudo técnico (ART) e aviso prévio de 10 dias úteis à Subprefeitura."
                )
        elif asks_pruning and tree_ctx:
            partes_acao.append(
                "Não realize poda por conta própria: pela Lei nº 17.794/2022 e pela ABNT NBR 16246-2 / NR-10, "
                "o manejo próximo à rede aérea só pode ser feito por equipe autorizada da Prefeitura (via Portal SP156) "
                "ou da concessionária Enel SP."
            )
        elif chunks:
            partes_acao.append(_sanitize_internal_codes(chunks[0]["conteudo"]))

        used_ids = [c["chunk_id"] for c in chunks[:3] if c["score"] >= 0.05]
        if tree_ctx and "ARBORGRID-TRIAGEM" not in used_ids:
            if any(c["chunk_id"] == "ARBORGRID-TRIAGEM" for c in self.chunks):
                used_ids.append("ARBORGRID-TRIAGEM")
        if asks_pruning and not tree_ctx:
            for cid in ("LEI-17794-ART05", "LEI-17794-ART18", "SP156-ROTINA"):
                if cid not in used_ids and any(c["chunk_id"] == cid for c in self.chunks):
                    used_ids.append(cid)
        if not used_ids and chunks:
            used_ids = [chunks[0]["chunk_id"]]

        return {
            "status": "respondido",
            "nivel_risco": nivel_risco,
            "orgao_responsavel": orgao,
            "telefone_ou_canal": canal,
            "pode_podar_por_conta": pode_podar,
            "acao_imediata": _sanitize_internal_codes(" ".join(partes_acao)),
            "fontes": used_ids,
        }

    def ask(
        self,
        question: str,
        geosampa_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        masked_question, question_cats = mask_personal_data(question)
        masked_cats_set = set(question_cats)
        clean_history: list[dict[str, str]] = []
        for t in history or []:
            if t.get("role") in ("user", "assistant") and str(t.get("content") or "").strip():
                masked_content, turn_cats = mask_personal_data(str(t["content"]))
                masked_cats_set.update(turn_cats)
                clean_history.append({"role": str(t["role"]), "content": masked_content})
        clean_history = clean_history[-6:]
        masked_categories = sorted(masked_cats_set)

        tree_ctx = self._lookup_tree_context(masked_question, explicit_geosampa_id=geosampa_id)
        last_user_msg = next(
            (t["content"] for t in reversed(clean_history) if t["role"] == "user"), ""
        )
        if not tree_ctx and not geosampa_id and last_user_msg:
            tree_ctx = self._lookup_tree_context(last_user_msg, explicit_geosampa_id=None)

        q_norm = _normalize_text(masked_question)
        enriched_query = masked_question
        if tree_ctx:
            enriched_query = (
                f"{masked_question} {tree_ctx['triage_consolidated']} sp156 poda rede eletrica "
                "enel nbr-16246 arborgrid"
            )
        elif last_user_msg and len(masked_question.split()) <= 6:
            enriched_query = f"{last_user_msg} {masked_question}"
        chunks = self.retrieve(enriched_query, top_k=3)

        # Se for pergunta sobre poda sem árvore vinculada, prioriza os trechos legais relevantes
        if not tree_ctx and any(t in q_norm for t in ("podar", "poda", "cortar", "corte", "supress")):
            priority_chunks: list[dict[str, Any]] = []
            for cid in ("LEI-17794-ART05", "LEI-17794-ART18", "SP156-ROTINA"):
                match = next((dict(c) for c in self.chunks if c["chunk_id"] == cid), None)
                if match:
                    match["score"] = 0.72
                    priority_chunks.append(match)
            seen_priority = {c["chunk_id"] for c in priority_chunks}
            chunks = priority_chunks + [c for c in chunks if c["chunk_id"] not in seen_priority]

        # Quando há árvore vinculada, injeta critérios de triagem e canal SP156 para citação consistente
        if tree_ctx:
            present_ids = {c["chunk_id"] for c in chunks}
            for cid in ("ARBORGRID-TRIAGEM", "SP156-ROTINA"):
                if cid not in present_ids:
                    match = next((dict(c) for c in self.chunks if c["chunk_id"] == cid), None)
                    if match:
                        match["score"] = 0.70
                        chunks.append(match)
                        present_ids.add(cid)

        top_score = chunks[0]["score"] if chunks else 0.0
        if not tree_ctx and top_score < 0.12:
            mode = "rag_deterministic"
            metric_mode = "fora_de_escopo"
            structured = self._out_of_scope_response()
        else:
            mode = "rag_deterministic"
            structured = self._call_local_qwen(
                masked_question, chunks, tree_ctx, clean_history
            )
            if structured is not None:
                mode = "qwen3_gguf"
            else:
                structured = self._synthesize_deterministic(
                    masked_question, chunks, tree_ctx
                )
            metric_mode = mode

        ASSISTANT_RESPONSES_TOTAL.labels(mode=metric_mode).inc()

        fontes_ids = set(structured.get("fontes") or [])
        citations = [
            {
                "chunk_id": c["chunk_id"],
                "titulo": c["titulo"],
                "fonte": c["fonte"],
                "categoria": c.get("categoria"),
                "score": c["score"],
                "conteudo": c["conteudo"],
                "cited": c["chunk_id"] in fontes_ids,
            }
            for c in chunks
        ]
        # Append any cited chunk not in top-k
        seen_cids = {c["chunk_id"] for c in citations}
        for cid in fontes_ids:
            if cid not in seen_cids:
                match = next((c for c in self.chunks if c["chunk_id"] == cid), None)
                if match:
                    citations.append(
                        {
                            "chunk_id": match["chunk_id"],
                            "titulo": match["titulo"],
                            "fonte": match["fonte"],
                            "categoria": match.get("categoria"),
                            "score": 0.5,
                            "conteudo": match["conteudo"],
                            "cited": True,
                        }
                    )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "question": masked_question,
            "mode": mode,
            "structured": structured,
            "answer": structured["acao_imediata"],
            "citations": citations,
            "tree_context": tree_ctx,
            "privacy": {"masked_categories": masked_categories},
            "latency_ms": latency_ms,
        }
