"""Avalia o módulo de busca semântica (Recall@1, Recall@3, MRR e limiar de recusa)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import DEFAULT_RETRIEVAL_THRESHOLD, DIR_ASSETS, DIR_EVAL, load_jsonl
from src.retrieval import KnowledgeRetriever, evaluate_retrieval


def main() -> None:
    test_cases = load_jsonl(DIR_EVAL / "test_set.jsonl")
    retriever = KnowledgeRetriever()
    thresholds = [round(0.20 + 0.05 * i, 2) for i in range(11)]
    report = evaluate_retrieval(
        retriever=retriever,
        test_cases=test_cases,
        thresholds=thresholds,
        chosen_threshold=DEFAULT_RETRIEVAL_THRESHOLD,
    )

    DIR_EVAL.mkdir(parents=True, exist_ok=True)
    DIR_ASSETS.mkdir(parents=True, exist_ok=True)

    with (DIR_EVAL / "retrieval_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    df = pd.DataFrame(report["sweep"])
    df.to_csv(DIR_EVAL / "retrieval_threshold_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(df["threshold"], df["recall_at_1"], marker="o", label="Recall@1 (respondíveis)")
    ax.plot(df["threshold"], df["recall_at_3"], marker="s", label="Recall@3 (respondíveis)")
    ax.plot(df["threshold"], df["oos_refusal_accuracy"], marker="^", label="Recusa Correta (não consta)")
    ax.plot(df["threshold"], df["balanced_score"], linestyle="--", marker="d", label="Score Balanceado")
    ax.axvline(report["chosen_threshold"], color="black", linestyle=":", label=f"Limiar escolhido ({report['chosen_threshold']:.2f})")
    ax.set_title("Varredura de Limiar de Similaridade Cosseno (RAG ArborGrid)")
    ax.set_xlabel("Limiar mínimo (cosine similarity)")
    ax.set_ylabel("Taxa / Score")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DIR_ASSETS / "retrieval_threshold_sweep.png", dpi=150)
    plt.close(fig)

    print("=== Varredura de Limiares ===")
    print(df.to_string(index=False))
    print("\nLimiar configurado:", report["chosen_metrics"])


if __name__ == "__main__":
    main()
