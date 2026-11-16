"""Módulo de busca híbrida (embeddings multilíngues + TF-IDF lexical) e varredura de limiar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common import DEFAULT_RETRIEVAL_THRESHOLD, DIR_BASE, EMBEDDING_MODEL, load_jsonl


class KnowledgeRetriever:
    """Recupera trechos normativos combinando similaridade densa e lexical (TF-IDF)."""

    def __init__(
        self,
        kb_path: Path | None = None,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
        dense_weight: float = 0.78,
    ) -> None:
        self.kb_path = kb_path or (DIR_BASE / "knowledge_base.jsonl")
        self.chunks: list[dict[str, Any]] = load_jsonl(self.kb_path)
        self.dense_weight = dense_weight
        self.model = SentenceTransformer(model_name, device=device)

        docs = [
            f"{row['chunk_id']} {row['titulo']}. {row['fonte']}. {row['conteudo']}"
            for row in self.chunks
        ]
        self.embeddings = self.model.encode(
            docs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(docs)

    def score_all(self, query: str) -> list[dict[str, Any]]:
        """Retorna todos os trechos ranqueados pelo score híbrido decrescente."""
        q_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        dense_sims = np.dot(self.embeddings, q_vec)
        q_tfidf = self.tfidf.transform([query])
        lexical_sims = np.asarray((self.tfidf_matrix @ q_tfidf.T).toarray()).ravel()

        scores = dense_sims + (0.35 * lexical_sims)
        order = np.argsort(-scores)
        ranked: list[dict[str, Any]] = []
        for idx in order:
            i = int(idx)
            item = dict(self.chunks[i])
            item["score"] = float(scores[i])
            item["dense_score"] = float(dense_sims[i])
            item["lexical_score"] = float(lexical_sims[i])
            ranked.append(item)
        return ranked

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = DEFAULT_RETRIEVAL_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Retorna os top_k trechos com score >= min_score."""
        ranked = self.score_all(query)
        filtered = [row for row in ranked if row["score"] >= min_score]
        return filtered[:top_k]


def evaluate_retrieval(
    retriever: KnowledgeRetriever,
    test_cases: list[dict[str, Any]],
    thresholds: list[float] | None = None,
    chosen_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
) -> dict[str, Any]:
    """Calcula Recall@1, Recall@3, MRR e taxa de bloqueio por limiar."""
    thresholds = thresholds or [round(0.15 + 0.05 * i, 2) for i in range(11)]

    ranked_per_case: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
        (case, retriever.score_all(case["pergunta"])) for case in test_cases
    ]

    sweep_rows: list[dict[str, Any]] = []
    for thr in thresholds:
        answerable = [
            (case, ranked)
            for case, ranked in ranked_per_case
            if case["esperado"]["status"] == "respondido"
        ]
        out_of_scope = [
            (case, ranked)
            for case, ranked in ranked_per_case
            if case["esperado"]["status"] == "nao_consta"
        ]

        r1_hits = 0
        r3_hits = 0
        rr_sum = 0.0
        false_refusals = 0

        for case, ranked in answerable:
            gold_ids = set(case["esperado"]["fontes"])
            filtered = [row for row in ranked if row["score"] >= thr]
            if not filtered:
                false_refusals += 1
                continue
            top1_ids = {filtered[0]["chunk_id"]}
            top3_ids = {row["chunk_id"] for row in filtered[:3]}
            if gold_ids & top1_ids:
                r1_hits += 1
            if gold_ids & top3_ids:
                r3_hits += 1
            for rank_idx, row in enumerate(filtered[:3], start=1):
                if row["chunk_id"] in gold_ids:
                    rr_sum += 1.0 / rank_idx
                    break

        oos_correct = sum(
            1 for _, ranked in out_of_scope if ranked[0]["score"] < thr
        )

        n_ans = max(len(answerable), 1)
        n_oos = max(len(out_of_scope), 1)
        recall_1 = r1_hits / n_ans
        recall_3 = r3_hits / n_ans
        oos_acc = oos_correct / n_oos
        balanced = 0.5 * recall_3 + 0.5 * oos_acc

        sweep_rows.append(
            {
                "threshold": thr,
                "recall_at_1": round(recall_1, 4),
                "recall_at_3": round(recall_3, 4),
                "mrr_at_3": round(rr_sum / n_ans, 4),
                "false_refusal_rate": round(false_refusals / n_ans, 4),
                "oos_refusal_accuracy": round(oos_acc, 4),
                "balanced_score": round(balanced, 4),
            }
        )

    chosen_metrics = min(
        sweep_rows,
        key=lambda row: abs(row["threshold"] - chosen_threshold),
    )

    per_case_details: list[dict[str, Any]] = []
    for case, ranked in ranked_per_case:
        top3 = [
            {
                "chunk_id": row["chunk_id"],
                "score": round(row["score"], 4),
                "dense": round(row["dense_score"], 4),
                "lexical": round(row["lexical_score"], 4),
            }
            for row in ranked[:3]
        ]
        per_case_details.append(
            {
                "id": case["id"],
                "categoria": case["categoria"],
                "expected_status": case["esperado"]["status"],
                "expected_sources": case["esperado"]["fontes"],
                "top3": top3,
                "max_score": round(ranked[0]["score"], 4),
            }
        )

    return {
        "chosen_threshold": chosen_threshold,
        "chosen_metrics": chosen_metrics,
        "sweep": sweep_rows,
        "per_case": per_case_details,
    }
