"""Benchmark comparativo das 3 configurações exigidas na rubrica:

1. Config 1 — Qwen3-1.7B Base sem contexto (zero-shot)
2. Config 2 — Qwen3-1.7B Base + RAG simples (top-3 com limiar cosseno)
3. Config 3 — Qwen3-1.7B + LoRA r=16 + RAG + Ferramenta GeoSampa (Agente)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import (
    BASE_MODEL,
    DEFAULT_RETRIEVAL_THRESHOLD,
    DIR_ADAPTERS,
    DIR_ASSETS,
    DIR_EVAL,
    format_chat_prompt,
    load_jsonl,
)
from src.retrieval import KnowledgeRetriever
from src.tools import ArborGridInspector
from src.validator import apply_post_generation_filters, evaluate_prediction


def _generate_one(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 240,
) -> tuple[str, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, latency_ms, int(gen_ids.shape[0])


def _summarize(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    n = len(rows)
    oos_rows = [r for r in rows if r["categoria"] == "nao_consta"]
    ans_rows = [r for r in rows if r["categoria"] != "nao_consta"]

    def _rate(subset: list[dict[str, Any]], key: str) -> float:
        return round(sum(1 for r in subset if r[key]) / max(len(subset), 1), 4)

    latencies = [r["latency_ms"] for r in rows]
    total_tokens = sum(r["generated_tokens"] for r in rows)
    total_secs = sum(latencies) / 1000.0
    return {
        "configuracao": name,
        "total_casos": n,
        "json_valido": _rate(rows, "json_valid"),
        "schema_valido": _rate(rows, "schema_valid"),
        "acuracia_risco": _rate(rows, "risk_exact"),
        "acuracia_canal": _rate(rows, "channel_exact"),
        "seguranca_poda": _rate(rows, "prune_exact"),
        "citacao_fonte": _rate(ans_rows, "source_cited"),
        "recusa_nao_consta": _rate(oos_rows, "refusal_exact"),
        "alucinacao_fonte": _rate(rows, "hallucinated_source"),
        "alucinacao_tel": _rate(rows, "hallucinated_phone"),
        "lgpd_limpo": _rate(rows, "lgpd_clean"),
        "acuracia_geral_estrita": _rate(rows, "overall_exact"),
        "latencia_media_ms": round(float(np.mean(latencies)), 1),
        "latencia_p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "tokens_medios": round(total_tokens / max(n, 1), 1),
        "tps_medio": round(total_tokens / max(total_secs, 1e-6), 2),
    }


def _plot_placar_horizontal(
    df_summary: pd.DataFrame,
    base_model: str,
    out_path: Path,
    *,
    full: bool = False,
) -> None:
    """Renderiza gráfico horizontal (1->2->3) com quesitos principais ou completos."""
    df_plot = df_summary.copy()
    df_plot["sem_aluc_fonte"] = 1.0 - df_plot["alucinacao_fonte"]
    df_plot["sem_aluc_tel"] = 1.0 - df_plot["alucinacao_tel"]

    if full:
        metric_cols = [
            ("json_valido", "JSON Válido"),
            ("schema_valido", "Schema Válido"),
            ("acuracia_risco", "Risco"),
            ("acuracia_canal", "Órgão / Canal"),
            ("seguranca_poda", "Segurança Poda"),
            ("citacao_fonte", "Citação Fonte"),
            ("recusa_nao_consta", "Recusa (não consta)"),
            ("sem_aluc_fonte", "Sem Aluc. Fonte"),
            ("sem_aluc_tel", "Sem Aluc. Telefone"),
            ("lgpd_limpo", "LGPD Limpo"),
            ("acuracia_geral_estrita", "Acerto Geral Estrito"),
        ]
        figsize = (11.2, 9.2)
        scope_label = "Todos os 11 Quesitos"
    else:
        metric_cols = [
            ("schema_valido", "Schema Válido"),
            ("acuracia_risco", "Risco"),
            ("acuracia_canal", "Órgão / Canal"),
            ("seguranca_poda", "Segurança Poda"),
            ("citacao_fonte", "Citação Fonte"),
            ("sem_aluc_tel", "Sem Aluc. Telefone"),
            ("acuracia_geral_estrita", "Acerto Geral Estrito"),
        ]
        figsize = (10.5, 6.4)
        scope_label = "Principais Quesitos"

    plot_df = df_plot.set_index("configuracao")[[c[0] for c in metric_cols]] * 100.0
    plot_df.columns = [c[1] for c in metric_cols]
    # Coloca 3 no índice 0 (barra inferior) e 1 no último (barra superior), mantendo 1, 2, 3 de cima para baixo
    plot_df = plot_df.iloc[::-1, ::-1]

    fig, ax = plt.subplots(figsize=figsize)
    # Paleta distinta Tab10 em ordem reversa de plotagem (3=verde, 2=laranja, 1=azul)
    palette = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    plot_df.T.plot(kind="barh", ax=ax, color=palette[: len(plot_df)], width=0.76)

    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            ax.text(
                w + 1.1,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w:.1f}%",
                va="center",
                ha="left",
                fontsize=8.2,
                fontweight="bold",
                color="#1b1b1b",
            )

    model_short = base_model.split("/")[-1].replace("Qwen3-", "Qwen ")
    ax.set_title(
        f"Placar Comparativo — {model_short}: {scope_label} (0–100%)",
        pad=26,
    )
    ax.set_xlabel("Taxa de Acerto / Conformidade (%)")
    ax.set_xlim(0, 114)
    ax.grid(axis="x", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _export_scoreboards(df_summary: pd.DataFrame, base_model: str, out_prefix: str) -> None:
    """Gera as versões resumida (principal) e completa (11 quesitos) do placar."""
    _plot_placar_horizontal(
        df_summary,
        base_model,
        DIR_ASSETS / f"{out_prefix}placar_comparativo.png",
        full=False,
    )
    _plot_placar_horizontal(
        df_summary,
        base_model,
        DIR_ASSETS / f"{out_prefix}placar_comparativo_completo.png",
        full=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-name", default="lora_arborgrid_r16")
    parser.add_argument("--out-prefix", default="")
    parser.add_argument("--replot", action="store_true")
    args_cli, _ = parser.parse_known_args()

    if args_cli.replot:
        csv_path = DIR_EVAL / f"{args_cli.out_prefix}placar_comparativo.csv"
        df_summary = pd.read_csv(csv_path)
        _export_scoreboards(df_summary, args_cli.base_model, args_cli.out_prefix)
        print(f"Gráficos atualizados a partir de {csv_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_cases = load_jsonl(DIR_EVAL / "test_set.jsonl")
    retriever = KnowledgeRetriever(device="cpu")
    inspector = ArborGridInspector()

    tokenizer = AutoTokenizer.from_pretrained(args_cli.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(args_cli.base_model, dtype=dtype).to(device)

    adapter_dir = DIR_ADAPTERS / args_cli.adapter_name
    has_lora = adapter_dir.exists()
    model = (
        PeftModel.from_pretrained(base_model, str(adapter_dir), adapter_name="arborgrid")
        if has_lora
        else base_model
    )
    model.eval()

    configs = [
        ("1_base_sem_contexto", False, False, False),
        ("2_rag_simples", True, False, False),
    ]
    if has_lora:
        configs.append(("3_rag_lora_agente", True, True, True))

    all_details: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    for cfg_name, use_rag, use_agent, use_lora in configs:
        print(f"\n=== Executando {cfg_name} ({len(test_cases)} casos) ===")
        cfg_results: list[dict[str, Any]] = []
        for case in test_cases:
            chunks = (
                retriever.retrieve(
                    case["pergunta"],
                    top_k=3,
                    min_score=DEFAULT_RETRIEVAL_THRESHOLD,
                )
                if use_rag
                else None
            )
            tree_ctx, _ = (
                inspector.maybe_build_tool_context(case["pergunta"])
                if use_agent
                else (None, None)
            )
            if use_agent and tree_ctx is not None and chunks is not None:
                present_ids = {c["chunk_id"] for c in chunks}
                for required_id in ("ARBORGRID-TRIAGEM", "SP156-ROTINA"):
                    if required_id not in present_ids:
                        match = next((dict(c) for c in retriever.chunks if c["chunk_id"] == required_id), None)
                        if match is not None:
                            match["score"] = 0.70
                            chunks.append(match)
            prompt = format_chat_prompt(
                tokenizer=tokenizer,
                question=case["pergunta"],
                chunks=chunks,
                tree_context=tree_ctx,
            )

            if has_lora and not use_lora:
                with model.disable_adapter():
                    raw_out, lat_ms, n_tokens = _generate_one(model, tokenizer, prompt)
            else:
                if has_lora:
                    model.set_adapter("arborgrid")
                raw_out, lat_ms, n_tokens = _generate_one(model, tokenizer, prompt)

            if use_lora and use_rag and use_agent:
                raw_out = apply_post_generation_filters(raw_out, case["pergunta"], chunks)

            metrics = evaluate_prediction(
                case=case,
                raw_output=raw_out,
                retrieved_chunks=chunks,
                tree_context=tree_ctx,
            )
            metrics["latency_ms"] = round(lat_ms, 1)
            metrics["generated_tokens"] = n_tokens
            metrics["tps"] = round(n_tokens / max(lat_ms / 1000.0, 1e-6), 2)
            metrics["retrieved_ids"] = [c["chunk_id"] for c in (chunks or [])]
            metrics["used_tree_tool"] = tree_ctx is not None
            cfg_results.append(metrics)

        all_details[cfg_name] = cfg_results
        summary = _summarize(cfg_results, cfg_name)
        summary_rows.append(summary)
        print(summary)

    DIR_EVAL.mkdir(parents=True, exist_ok=True)
    DIR_ASSETS.mkdir(parents=True, exist_ok=True)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(DIR_EVAL / f"{args_cli.out_prefix}placar_comparativo.csv", index=False)
    with (DIR_EVAL / f"{args_cli.out_prefix}benchmark_detalhes.json").open("w", encoding="utf-8") as fp:
        json.dump(all_details, fp, indent=2, ensure_ascii=False)

    _export_scoreboards(df_summary, args_cli.base_model, args_cli.out_prefix)

    print("\n=== PLACAR FINAL ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
