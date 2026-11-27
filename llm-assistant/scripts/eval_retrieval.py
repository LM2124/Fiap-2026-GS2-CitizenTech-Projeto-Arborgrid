"""Avalia o módulo de busca semântica (Recall@1, Recall@3, MRR e limiar de recusa).

O limiar é escolhido num conjunto de validação separado (avaliacao/validacao_limiar.jsonl), e os
30 casos de teste só medem o limiar escolhido. Regra: o maior limiar que mantém a recusa indevida
em zero sem perder Recall@3 (em triagem de risco, descartar uma pergunta legítima custa mais do que
deixar o modelo recusar uma fora do escopo).
"""

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


def escolher_limiar(sweep: list[dict]) -> float:
    """Maior limiar com recusa indevida zero entre os que mantêm o melhor Recall@3."""
    sem_recusa = [r for r in sweep if r["false_refusal_rate"] == 0] or sweep[:1]
    melhor_r3 = max(r["recall_at_3"] for r in sem_recusa)
    return max(r["threshold"] for r in sem_recusa if r["recall_at_3"] == melhor_r3)


def main() -> None:
    validation_cases = load_jsonl(DIR_EVAL / "validacao_limiar.jsonl")
    test_cases = load_jsonl(DIR_EVAL / "test_set.jsonl")
    retriever = KnowledgeRetriever(device="cpu")
    thresholds = [round(0.20 + 0.05 * i, 2) for i in range(11)]

    validation = evaluate_retrieval(retriever=retriever, test_cases=validation_cases, thresholds=thresholds)
    chosen = escolher_limiar(validation["sweep"])
    report = evaluate_retrieval(
        retriever=retriever,
        test_cases=test_cases,
        thresholds=thresholds,
        chosen_threshold=chosen,
    )
    report["limiar_escolhido_na_validacao"] = chosen
    report["limiar_em_uso"] = DEFAULT_RETRIEVAL_THRESHOLD
    report["validacao"] = {"casos": len(validation_cases), "sweep": validation["sweep"]}

    DIR_EVAL.mkdir(parents=True, exist_ok=True)
    DIR_ASSETS.mkdir(parents=True, exist_ok=True)

    with (DIR_EVAL / "retrieval_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    pd.DataFrame(validation["sweep"]).to_csv(DIR_EVAL / "retrieval_threshold_sweep_validacao.csv", index=False)
    df = pd.DataFrame(report["sweep"])
    df.to_csv(DIR_EVAL / "retrieval_threshold_sweep.csv", index=False)
    print("=== Validação (escolha do limiar) ===")
    print(pd.DataFrame(validation["sweep"]).to_string(index=False))
    print(f"Limiar escolhido na validação: {chosen:.2f} (em uso: {DEFAULT_RETRIEVAL_THRESHOLD:.2f})")

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
