"""Gera gráficos do estudo exploratório de modelos (SmolLM2-360M -> Qwen 0.6B -> Qwen 1.7B)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import DIR_ASSETS, DIR_EVAL


def _extract_metrics(row: pd.Series | dict[str, Any], vram_or_ram: float) -> dict[str, float]:
    return {
        "schema": float(row["schema_valido"]) * 100.0,
        "risco": float(row["acuracia_risco"]) * 100.0,
        "canal": float(row["acuracia_canal"]) * 100.0,
        "poda": float(row["seguranca_poda"]) * 100.0,
        "citacao": float(row["citacao_fonte"]) * 100.0,
        "recusa": float(row["recusa_nao_consta"]) * 100.0,
        "sem_tel": (1.0 - float(row["alucinacao_tel"])) * 100.0,
        "acc": float(row["acuracia_geral_estrita"]) * 100.0,
        "lat_s": float(row["latencia_media_ms"]) / 1000.0,
        "p95_s": float(row["latencia_p95_ms"]) / 1000.0,
        "tokens": float(row["tokens_medios"]),
        "mem_mib": vram_or_ram,
    }


def _load_all_data() -> dict[str, dict[str, dict[str, float]]]:
    qwen_df = pd.read_csv(DIR_EVAL / "placar_comparativo_modelos.csv")
    smol_gpu_df = pd.read_csv(DIR_EVAL / "smol360m_placar_comparativo.csv")
    smol_cpu_f16 = json.loads((DIR_EVAL / "smol360m_cpu_f16_report.json").read_text(encoding="utf-8"))["summary"]
    smol_cpu_q4 = json.loads((DIR_EVAL / "smol360m_cpu_q4_report.json").read_text(encoding="utf-8"))["summary"]

    def _qwen_row(model: str, stage: str) -> pd.Series:
        return qwen_df[(qwen_df["modelo_base"] == model) & (qwen_df["etapa_chave"] == stage)].iloc[0]

    smol_gpu = smol_gpu_df[smol_gpu_df["configuracao"] == "3_rag_lora_agente"].iloc[0]
    q06_gpu = _qwen_row("Qwen 0.6B", "3_rag_lora_agente")
    q06_cpu_f16 = _qwen_row("Qwen 0.6B", "gguf_cpu_f16")
    q06_cpu = _qwen_row("Qwen 0.6B", "4_gguf_q4km_cpu2t")
    q17_gpu = _qwen_row("Qwen 1.7B", "3_rag_lora_agente")
    q17_cpu_f16 = _qwen_row("Qwen 1.7B", "gguf_cpu_f16")
    q17_cpu = _qwen_row("Qwen 1.7B", "4_gguf_q4km_cpu2t")

    return {
        "SmolLM2-360M": {
            "gpu16": _extract_metrics(smol_gpu, 756.4),
            "cpu16": _extract_metrics(smol_cpu_f16, 829.6),
            "cpu4": _extract_metrics(smol_cpu_q4, 417.6),
        },
        "Qwen 0.6B": {
            "gpu16": _extract_metrics(q06_gpu, float(q06_gpu["memoria_execucao_mib"])),
            "cpu16": _extract_metrics(q06_cpu_f16, float(q06_cpu_f16["memoria_execucao_mib"])),
            "cpu4": _extract_metrics(q06_cpu, float(q06_cpu["memoria_execucao_mib"])),
        },
        "Qwen 1.7B": {
            "gpu16": _extract_metrics(q17_gpu, float(q17_gpu["memoria_execucao_mib"])),
            "cpu16": _extract_metrics(q17_cpu_f16, float(q17_cpu_f16["memoria_execucao_mib"])),
            "cpu4": _extract_metrics(q17_cpu, float(q17_cpu["memoria_execucao_mib"])),
        },
    }


def _plot_overview_3_models(data: dict[str, dict[str, dict[str, float]]]) -> Path:
    models = ["SmolLM2-360M", "Qwen 0.6B", "Qwen 1.7B"]
    x = np.arange(len(models))
    width = 0.35
    gap = 0.03

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.2))
    fig.suptitle(
        "Modelos e Quantização em CPU (2 threads): SmolLM2-360M vs Qwen 0.6B vs Qwen 1.7B",
        fontsize=12.8,
        fontweight="bold",
        y=0.98,
    )

    color_f16 = "#2563eb"
    color_q4 = "#f97316"
    panels = [
        ("acc", "Acerto Geral Estrito (%)", "Acurácia (%) — maior é melhor", 118, "{:.1f}%"),
        ("lat_s", "Latência Média por Resposta (s)", "Segundos — menor é melhor", 32.0, "{:.2f}s"),
        ("mem_mib", "RAM Base Alocada (MiB)", "Memória Base em CPU (MiB)", 4250, "{:.0f}"),
    ]

    for ax, (key, title, ylabel, ylim, fmt) in zip(axes, panels):
        vals_f16 = [data[m]["cpu16"][key] for m in models]
        vals_q4 = [data[m]["cpu4"][key] for m in models]

        b1 = ax.bar(
            x - (width + gap) / 2,
            vals_f16,
            width,
            label="16 bits (Original)",
            color=color_f16,
            edgecolor="#1e293b",
            linewidth=0.6,
        )
        b2 = ax.bar(
            x + (width + gap) / 2,
            vals_q4,
            width,
            label="4 bits (Quantizado)",
            color=color_q4,
            edgecolor="#1e293b",
            linewidth=0.6,
        )

        for idx, (bar, v) in enumerate(zip(b1, vals_f16)):
            extra = f"\n({data[models[idx]]['cpu16']['tokens']:.0f}t)" if key == "lat_s" else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + ylim * 0.018,
                f"{fmt.format(v)}{extra}",
                ha="center",
                va="bottom",
                fontsize=7.8 if extra else 8.3,
                fontweight="bold",
                color="#1e3a8a",
            )
        for idx, (bar, v) in enumerate(zip(b2, vals_q4)):
            extra = f"\n({data[models[idx]]['cpu4']['tokens']:.0f}t)" if key == "lat_s" else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + ylim * 0.018,
                f"{fmt.format(v)}{extra}",
                ha="center",
                va="bottom",
                fontsize=7.8 if extra else 8.3,
                fontweight="bold",
                color="#9a3412",
            )

        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, ylim)
        ax.grid(axis="y", alpha=0.28)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.916), ncol=2, frameon=True, fontsize=9.4)
    fig.text(
        0.5,
        0.015,
        "Comparação controlada de porte e quantização em CPU pura; detalhes por quesito e GPU constam nas Seções 1–3.",
        ha="center",
        fontsize=8.5,
        color="#475569",
        style="italic",
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.87])
    out_path = DIR_ASSETS / "comparativo_smol360m_vs_qwen.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_criteria_and_cost(
    model_a: str,
    model_b: str,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    color_a: str,
    color_b: str,
    title: str,
    subtitle_note: str,
    out_name: str,
    cpu_speed_extra: tuple[dict[str, float], dict[str, float]] | None = None,
) -> Path:
    """Compara dois modelos em precisão 16b GPU (quesitos à esq., latência e VRAM à dir.)."""
    fig = plt.figure(figsize=(14.4, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 0.85], wspace=0.30)
    fig.suptitle(title, fontsize=12.8, fontweight="bold", y=0.98)

    ax_q = fig.add_subplot(gs[0, 0])
    criteria = [
        ("acc", "Acerto Geral Estrito"),
        ("recusa", "Recusa 'Não Consta'"),
        ("citacao", "Citação de Fonte"),
        ("poda", "Segurança de Poda"),
        ("canal", "Órgão / Canal"),
        ("risco", "Classificação Risco"),
        ("sem_tel", "Sem Aluc. Telefone"),
        ("schema", "Schema JSON Válido"),
    ]
    y = np.arange(len(criteria))
    h = 0.36
    vals_a = [metrics_a[k] for k, _ in criteria]
    vals_b = [metrics_b[k] for k, _ in criteria]

    bars_a = ax_q.barh(y + h / 2, vals_a, h, label=model_a, color=color_a, edgecolor="#1e293b", linewidth=0.6)
    bars_b = ax_q.barh(y - h / 2, vals_b, h, label=model_b, color=color_b, edgecolor="#1e293b", linewidth=0.6)

    for bar, v in zip(bars_a, vals_a):
        ax_q.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color="#334155")
    for bar, v in zip(bars_b, vals_b):
        ax_q.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color="#0f172a")

    ax_q.set_yticks(y)
    ax_q.set_yticklabels([lbl for _, lbl in criteria], fontsize=8.8)
    ax_q.set_xlim(0, 118)
    ax_q.set_xlabel("Taxa de Acerto (%) — 30 casos independentes", fontsize=8.8)
    ax_q.set_title("Qualidade por Quesito (Original 16b, GPU)", fontsize=10.2, fontweight="bold", pad=8)
    ax_q.grid(axis="x", alpha=0.28)

    ax_l = fig.add_subplot(gs[0, 1])
    if cpu_speed_extra is not None:
        cpu_a, cpu_b = cpu_speed_extra
        groups = ["GPU 16 bits\n(PyTorch)", "CPU 2t Q4_K_M\n(checagem GGUF)"]
        gx = np.arange(len(groups))
        w = 0.34
        la = [metrics_a["lat_s"], cpu_a["lat_s"]]
        lb = [metrics_b["lat_s"], cpu_b["lat_s"]]
        ta = [metrics_a["tokens"], cpu_a["tokens"]]
        tb = [metrics_b["tokens"], cpu_b["tokens"]]
        ba = ax_l.bar(gx - w / 2 - 0.02, la, w, color=color_a, edgecolor="#1e293b", linewidth=0.6, label=model_a)
        bb = ax_l.bar(gx + w / 2 + 0.02, lb, w, color=color_b, edgecolor="#1e293b", linewidth=0.6, label=model_b)
        for bar, v, tok in zip(ba, la, ta):
            ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.2f}s\n({tok:.0f}t)", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="#334155")
        for bar, v, tok in zip(bb, lb, tb):
            ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.2f}s\n({tok:.0f}t)", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="#0f172a")
        ax_l.set_xticks(gx)
        ax_l.set_xticklabels(groups, fontsize=8.6, fontweight="bold")
        ax_l.set_ylim(0, 15.2)
        ax_l.set_title("Latência Média (s) e Tokens Gerados", fontsize=10.2, fontweight="bold", pad=8)
    else:
        groups = ["Latência Média", "Latência P95"]
        gx = np.arange(len(groups))
        w = 0.34
        la = [metrics_a["lat_s"], metrics_a["p95_s"]]
        lb = [metrics_b["lat_s"], metrics_b["p95_s"]]
        ba = ax_l.bar(gx - w / 2 - 0.02, la, w, color=color_a, edgecolor="#1e293b", linewidth=0.6)
        bb = ax_l.bar(gx + w / 2 + 0.02, lb, w, color=color_b, edgecolor="#1e293b", linewidth=0.6)
        for bar, v in zip(ba, la):
            ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.2f}s", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#334155")
        for bar, v in zip(bb, lb):
            ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.2f}s", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#0f172a")
        ax_l.set_xticks(gx)
        ax_l.set_xticklabels(groups, fontsize=8.8, fontweight="bold")
        ax_l.set_ylim(0, 15.2)
        ax_l.set_title("Latência Média e P95 (GPU 16b)", fontsize=10.2, fontweight="bold", pad=8)
    ax_l.set_ylabel("Segundos — menor é melhor", fontsize=8.8)
    ax_l.grid(axis="y", alpha=0.28)

    ax_m = fig.add_subplot(gs[0, 2])
    mx = np.arange(2)
    mvals = [metrics_a["mem_mib"], metrics_b["mem_mib"]]
    mbars = ax_m.bar(mx, mvals, width=0.52, color=[color_a, color_b], edgecolor="#1e293b", linewidth=0.6)
    for bar, v in zip(mbars, mvals):
        ax_m.text(bar.get_x() + bar.get_width() / 2, v + 110, f"{v:.0f}\nMiB", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")
    ax_m.set_xticks(mx)
    ax_m.set_xticklabels([model_a, model_b], fontsize=8.6, fontweight="bold")
    ax_m.set_ylim(0, 5100)
    ax_m.set_ylabel("VRAM Alocada em 16 bits (MiB)", fontsize=8.8)
    ax_m.set_title("Consumo de VRAM (GPU 16b)", fontsize=10.2, fontweight="bold", pad=8)
    ax_m.grid(axis="y", alpha=0.28)

    handles, labels = ax_q.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.918), ncol=2, frameon=True, fontsize=9.2)
    fig.text(0.5, 0.018, subtitle_note, ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.subplots_adjust(left=0.125, right=0.985, top=0.79, bottom=0.145)
    out_path = DIR_ASSETS / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_arch_pair(
    models_spec: list[tuple[str, str, str]],
    title: str,
    note: str,
    out_name: str,
    qwen_df: pd.DataFrame,
    smol_df: pd.DataFrame,
) -> Path:
    stages = [
        ("1_base_sem_contexto", "1. Base\nZero-Shot"),
        ("2_rag_simples", "2. RAG\nSimples"),
        ("3_rag_lora_agente", "3. LoRA + RAG\n+ Agente"),
    ]

    def _get(model: str, stage: str, col: str) -> float:
        if model == "SmolLM2-360M":
            row = smol_df[smol_df["configuracao"] == stage].iloc[0]
        else:
            row = qwen_df[(qwen_df["modelo_base"] == model) & (qwen_df["etapa_chave"] == stage)].iloc[0]
        return float(row[col]) * 100.0

    x = np.arange(len(stages))
    width = 0.34
    offsets = [-(width / 2 + 0.02), (width / 2 + 0.02)]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1))
    fig.suptitle(title, fontsize=12.6, fontweight="bold", y=0.98)

    panels = [
        ("schema_valido", "Validade de Schema JSON (%)", "Percentual (%) — maior é melhor"),
        ("acuracia_geral_estrita", "Acerto Geral Estrito (%)", "Percentual (%) — maior é melhor"),
    ]

    for ax, (col, ptitle, ylabel) in zip(axes, panels):
        for (model_name, color, text_color), offset in zip(models_spec, offsets):
            vals = [_get(model_name, st_key, col) for st_key, _ in stages]
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=model_name if col == "schema_valido" else None,
                color=color,
                edgecolor="#1e293b",
                linewidth=0.6,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 2.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    fontweight="bold",
                    color=text_color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in stages], fontsize=9.2, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.set_ylabel(ylabel, fontsize=9.0)
        ax.set_title(ptitle, fontsize=10.5, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.28)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.916), ncol=2, frameon=True, fontsize=9.4)
    fig.text(0.5, 0.02, note, ha="center", fontsize=8.5, color="#475569", style="italic")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.79, bottom=0.17, wspace=0.16)
    out_path = DIR_ASSETS / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _load_cpu_qwen_rows(qwen_df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    mapping = {
        ("Qwen 0.6B", "f16"): qwen_df[(qwen_df["modelo_base"] == "Qwen 0.6B") & (qwen_df["etapa_chave"] == "gguf_cpu_f16")].iloc[0],
        ("Qwen 0.6B", "q4"): qwen_df[(qwen_df["modelo_base"] == "Qwen 0.6B") & (qwen_df["etapa_chave"] == "4_gguf_q4km_cpu2t")].iloc[0],
        ("Qwen 1.7B", "f16"): qwen_df[(qwen_df["modelo_base"] == "Qwen 1.7B") & (qwen_df["etapa_chave"] == "gguf_cpu_f16")].iloc[0],
        ("Qwen 1.7B", "q4"): qwen_df[(qwen_df["modelo_base"] == "Qwen 1.7B") & (qwen_df["etapa_chave"] == "4_gguf_q4km_cpu2t")].iloc[0],
    }
    return mapping


def _plot_cpu_quant_quality_memory(cpu_rows: dict[tuple[str, str], pd.Series]) -> Path:
    precisions = [("f16", "16 bits (GGUF F16)"), ("q4", "4 bits (GGUF Q4_K_M)")]
    models = [("Qwen 0.6B", "#2563eb", "#1e3a8a"), ("Qwen 1.7B", "#16a34a", "#14532d")]

    x = np.arange(len(precisions))
    width = 0.34
    offsets = [-(width / 2 + 0.02), (width / 2 + 0.02)]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.suptitle(
        "Etapa 3.1 — Quantização em CPU (2 threads): Acurácia Estrita e Consumo de Memória",
        fontsize=12.6,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        ("acc", "Acerto Geral Estrito (CPU 2t)", "Acurácia (%) — maior é melhor", 118),
        ("mem", "RAM Base Alocada e Tamanho em Disco (CPU 2t)", "Memória Base Alocada (MiB)", 4250),
    ]

    for ax, (mode, ptitle, ylabel, ylim) in zip(axes, panels):
        for (model_name, color, text_color), offset in zip(models, offsets):
            vals = []
            labels_txt = []
            for pk, _ in precisions:
                r = cpu_rows[(model_name, pk)]
                if mode == "acc":
                    v = float(r["acuracia_geral_estrita"]) * 100.0
                    vals.append(v)
                    labels_txt.append(f"{v:.1f}%")
                else:
                    v = float(r["memoria_execucao_mib"])
                    disk = float(r["tamanho_gguf_mib"])
                    vals.append(v)
                    labels_txt.append(f"{v:.0f} MiB\n({disk:.0f} disco)")
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=model_name if mode == "acc" else None,
                color=color,
                edgecolor="#1e293b",
                linewidth=0.6,
            )
            for bar, v, txt in zip(bars, vals, labels_txt):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + ylim * 0.02,
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                    color=text_color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([plabel for _, plabel in precisions], fontsize=9.2, fontweight="bold")
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel, fontsize=9.0)
        ax.set_title(ptitle, fontsize=10.4, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.28)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.916), ncol=2, frameon=True, fontsize=9.4)
    fig.text(
        0.5,
        0.02,
        "Em CPU (2 threads), o Qwen 1.7B Q4_K_M mantém 93,3% intactos usando 2.015 MiB de RAM base (-44% vs F16), enquanto o Qwen 0.6B cai para 66,7%.",
        ha="center",
        fontsize=8.5,
        color="#475569",
        style="italic",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.16, wspace=0.18)
    out_path = DIR_ASSETS / "exp_3a_quantizacao_cpu_qualidade_memoria.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_cpu_quant_speed(cpu_rows: dict[tuple[str, str], pd.Series]) -> Path:
    precisions = [("f16", "16 bits (GGUF F16)"), ("q4", "4 bits (GGUF Q4_K_M)")]
    models = [("Qwen 0.6B", "#2563eb", "#1e3a8a"), ("Qwen 1.7B", "#16a34a", "#14532d")]

    x = np.arange(len(precisions))
    width = 0.34
    offsets = [-(width / 2 + 0.02), (width / 2 + 0.02)]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.0))
    fig.suptitle(
        "Etapa 3.2 — Latência e Velocidade em CPU Pura (2 threads, -t 2): 16 bits vs. 4 bits",
        fontsize=12.8,
        fontweight="bold",
        y=0.98,
    )

    specs = [
        (axes[0, 0], "latencia_media_ms", 0.001, "Latência Média por Resposta", "Segundos — menor é melhor", 31.0, "{:.2f}s"),
        (axes[0, 1], "latencia_p95_ms", 0.001, "Latência P95 (cauda 5%)", "Segundos — menor é melhor", 41.5, "{:.2f}s"),
        (axes[1, 0], "tps_medio", 1.0, "Velocidade Ponta a Ponta (Prompt + Decode)", "tok/s — maior é melhor", 38.0, "{:.1f}"),
        (axes[1, 1], "tps_decode_medio", 1.0, "Velocidade de Decode (Geração)", "tok/s — maior é melhor", 56.0, "{:.1f}"),
    ]

    for idx, (ax, col, scale, ptitle, ylabel, ylim, fmt) in enumerate(specs):
        for (model_name, color, text_color), offset in zip(models, offsets):
            vals = [float(cpu_rows[(model_name, pk)][col]) * scale for pk, _ in precisions]
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=model_name if idx == 0 else None,
                color=color,
                edgecolor="#1e293b",
                linewidth=0.6,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + ylim * 0.02,
                    fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    fontweight="bold",
                    color=text_color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([plabel for _, plabel in precisions], fontsize=9.0, fontweight="bold")
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel, fontsize=8.8)
        ax.set_title(ptitle, fontsize=10.3, fontweight="bold", pad=7)
        ax.grid(axis="y", alpha=0.28)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=True, fontsize=9.5)
    fig.text(
        0.5,
        0.015,
        "Em CPU (2 threads), a quantização Q4_K_M acelera o Qwen 1.7B em 2,7× (25,20s -> 9,30s médios; 17,3 tok/s E2E e 28,6 tok/s em decode).",
        ha="center",
        fontsize=8.5,
        color="#475569",
        style="italic",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.845, bottom=0.10, wspace=0.16, hspace=0.30)
    out_path = DIR_ASSETS / "exp_3b_quantizacao_cpu_velocidade.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    data = _load_all_data()
    qwen_df = pd.read_csv(DIR_EVAL / "placar_comparativo_modelos.csv")
    smol_df = pd.read_csv(DIR_EVAL / "smol360m_placar_comparativo.csv")
    cpu_rows = _load_cpu_qwen_rows(qwen_df)

    p_arch_1 = _plot_arch_pair(
        models_spec=[("SmolLM2-360M", "#94a3b8", "#334155"), ("Qwen 0.6B", "#2563eb", "#1e3a8a")],
        title="Etapa 1A — Seleção da Arquitetura: SmolLM2-360M vs. Qwen 0.6B (16 bits, GPU)",
        note="LoRA + RAG + Agente eleva o acerto em ambos os modelos (0% -> 20% no SmolLM2; 26,7% -> 86,7% no Qwen 0.6B).",
        out_name="exp_1a_arquitetura_smol_vs_qwen06b.png",
        qwen_df=qwen_df,
        smol_df=smol_df,
    )
    p_arch_2 = _plot_arch_pair(
        models_spec=[("Qwen 0.6B", "#2563eb", "#1e3a8a"), ("Qwen 1.7B", "#16a34a", "#14532d")],
        title="Etapa 2A — Evolução por Arquitetura: Qwen 0.6B vs. Qwen 1.7B (16 bits, GPU)",
        note="Sem LoRA + Agente ambos estagnam em 20%–33%; apenas com o sistema completo a escala 0.6B -> 1.7B atinge 93,3%.",
        out_name="exp_2a_arquitetura_qwen06b_vs_17b.png",
        qwen_df=qwen_df,
        smol_df=smol_df,
    )
    p_overview = _plot_overview_3_models(data)
    p_sec1 = _plot_criteria_and_cost(
        model_a="SmolLM2-360M",
        model_b="Qwen 0.6B",
        metrics_a=data["SmolLM2-360M"]["gpu16"],
        metrics_b=data["Qwen 0.6B"]["gpu16"],
        color_a="#94a3b8",
        color_b="#2563eb",
        title="Etapa 1B — Quesitos, Latência e VRAM: SmolLM2-360M vs. Qwen 0.6B (16 bits)",
        subtitle_note="Conclusão da Etapa 1: Qwen 0.6B é 4,3× mais preciso (86,7% vs 20,0%) e mais rápido (8,47s vs 10,44s GPU; 4,97s vs 5,16s CPU Q4_K_M), adotando-se a família Qwen.",
        out_name="exp_1_smol360m_vs_qwen06b.png",
        cpu_speed_extra=(data["SmolLM2-360M"]["cpu4"], data["Qwen 0.6B"]["cpu4"]),
    )
    p_sec2 = _plot_criteria_and_cost(
        model_a="Qwen 0.6B",
        model_b="Qwen 1.7B",
        metrics_a=data["Qwen 0.6B"]["gpu16"],
        metrics_b=data["Qwen 1.7B"]["gpu16"],
        color_a="#2563eb",
        color_b="#16a34a",
        title="Etapa 2B — Escala na Família Qwen: 0.6B vs. 1.7B (Original 16 bits, GPU)",
        subtitle_note="Conclusão da Etapa 2: O Qwen 1.7B eleva o acerto estrito para 93,3% (100% em citação, poda e canal), mas exige 2,6× mais VRAM (4.172 MiB), motivando quantização em 4 bits.",
        out_name="exp_2_qwen06b_vs_17b_escala.png",
        cpu_speed_extra=None,
    )
    p_cpu_qm = _plot_cpu_quant_quality_memory(cpu_rows)
    p_cpu_sp = _plot_cpu_quant_speed(cpu_rows)

    for p in (p_arch_1, p_arch_2, p_overview, p_sec1, p_sec2, p_cpu_qm, p_cpu_sp):
        print(f"Gráfico gerado em {p}")


if __name__ == "__main__":
    main()
