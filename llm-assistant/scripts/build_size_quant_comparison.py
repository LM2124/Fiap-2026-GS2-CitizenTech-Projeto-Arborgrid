"""Consolida o comparativo Qwen 1.7B vs Qwen 0.6B (Original e Quantizado)."""

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


STAGE_LABELS = {
    "1_base_sem_contexto": "1. Base sem contexto",
    "2_rag_simples": "2. RAG simples",
    "3_rag_lora_agente": "3. LoRA + RAG + Agente (Original 16 bits, GPU)",
    "4_gguf_q4km_cpu2t": "4. LoRA + RAG + Agente (Quantizado 4 bits, CPU 2t)",
}


def _rows_from_gpu_csv(csv_path: Path, model_name: str, mem_info: dict[str, Any]) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []
    for _, rec in df.iterrows():
        cfg = str(rec["configuracao"])
        vram = mem_info["original_bf16_vram_mib"] if cfg == "3_rag_lora_agente" else None
        rows.append(
            {
                "modelo_base": model_name,
                "formato_execucao": "Original 16 bits (GPU CUDA)",
                "etapa_chave": cfg,
                "configuracao": STAGE_LABELS.get(cfg, cfg),
                "total_casos": int(rec["total_casos"]),
                "tamanho_gguf_mib": None,
                "memoria_execucao_mib": vram,
                "json_valido": float(rec["json_valido"]),
                "schema_valido": float(rec["schema_valido"]),
                "acuracia_risco": float(rec["acuracia_risco"]),
                "acuracia_canal": float(rec["acuracia_canal"]),
                "seguranca_poda": float(rec["seguranca_poda"]),
                "citacao_fonte": float(rec["citacao_fonte"]),
                "recusa_nao_consta": float(rec["recusa_nao_consta"]),
                "alucinacao_fonte": float(rec["alucinacao_fonte"]),
                "alucinacao_tel": float(rec["alucinacao_tel"]),
                "lgpd_limpo": float(rec["lgpd_limpo"]),
                "acuracia_geral_estrita": float(rec["acuracia_geral_estrita"]),
                "latencia_media_ms": float(rec["latencia_media_ms"]),
                "latencia_p95_ms": float(rec["latencia_p95_ms"]) if "latencia_p95_ms" in rec else None,
                "tokens_medios": float(rec["tokens_medios"]) if "tokens_medios" in rec else None,
                "tps_medio": float(rec["tps_medio"]) if "tps_medio" in rec else None,
            }
        )
    return rows


def _row_from_gguf_json(
    json_path: Path,
    model_name: str,
    formato_execucao: str,
    etapa_chave: str,
    configuracao: str,
    rss_override_mib: float | None = None,
) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    s = payload["summary"]
    rss = rss_override_mib if rss_override_mib is not None else float(s["pico_ram_rss_mib"])
    return {
        "modelo_base": model_name,
        "formato_execucao": formato_execucao,
        "etapa_chave": etapa_chave,
        "configuracao": configuracao,
        "total_casos": int(s["total_casos"]),
        "tamanho_gguf_mib": float(s["tamanho_gguf_mib"]),
        "memoria_execucao_mib": float(rss),
        "json_valido": float(s["json_valido"]),
        "schema_valido": float(s["schema_valido"]),
        "acuracia_risco": float(s["acuracia_risco"]),
        "acuracia_canal": float(s["acuracia_canal"]),
        "seguranca_poda": float(s["seguranca_poda"]),
        "citacao_fonte": float(s["citacao_fonte"]),
        "recusa_nao_consta": float(s["recusa_nao_consta"]),
        "alucinacao_fonte": float(s["alucinacao_fonte"]),
        "alucinacao_tel": float(s["alucinacao_tel"]),
        "lgpd_limpo": float(s["lgpd_limpo"]),
        "acuracia_geral_estrita": float(s["acuracia_geral_estrita"]),
        "latencia_media_ms": float(s["latencia_media_ms"]),
        "latencia_p95_ms": float(s["latencia_p95_ms"]),
        "tokens_medios": float(s["tokens_medios"]) if s.get("tokens_medios") is not None else None,
        "tps_medio": float(s["tps_medio"]) if s.get("tps_medio") is not None else None,
        "tps_decode_medio": float(s["tps_decode_medio"]) if s.get("tps_decode_medio") is not None else None,
    }


def _plot_ec2_sizing_and_cost(mem_matrix: dict[str, Any], out_path: Path) -> None:
    """Compara consumo de disco/RAM em CPU contra tetos EC2 e topologia 2x small vs 1x medium."""
    plt.rcParams.update({"figure.dpi": 140})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    specs = [
        ("Qwen 0.6B\n(4 bits)", float(mem_matrix["0.6B"]["disk_mib"]), float(mem_matrix["0.6B"]["cpu_rss_mib"])),
        ("Qwen 1.7B\n(4 bits)", float(mem_matrix["1.7B"]["disk_mib"]), float(mem_matrix["1.7B"]["cpu_rss_mib"])),
        ("Qwen 0.6B\n(16 bits)", float(mem_matrix["0.6B"]["f16_disk_mib"]), float(mem_matrix["0.6B"]["f16_cpu_rss_mib"])),
        ("Qwen 1.7B\n(16 bits)", float(mem_matrix["1.7B"]["f16_disk_mib"]), float(mem_matrix["1.7B"]["f16_cpu_rss_mib"])),
    ]
    labels = [s[0] for s in specs]
    disk_vals = [s[1] for s in specs]
    rss_vals = [s[2] for s in specs]

    xm = np.arange(len(labels))
    w = 0.36
    bd = ax1.bar(xm - w / 2, disk_vals, w, label="Arquivo em disco (MiB)", color="#457b9d")
    br = ax1.bar(xm + w / 2, rss_vals, w, label="RAM Base em CPU pura (MiB)", color="#e76f51")
    ax1.axhline(2048, color="#b5179e", linestyle=":", linewidth=1.7, label="EC2 t3.small (2.048 MiB RAM)")
    ax1.axhline(4096, color="#264653", linestyle="--", linewidth=1.7, label="EC2 t3.medium (4.096 MiB RAM)")
    ax1.set_xticks(xm)
    ax1.set_xticklabels(labels, fontsize=9.2)
    ax1.set_ylim(0, 5250)
    ax1.set_ylabel("Armazenamento / RAM (MiB)")
    ax1.set_title("Envelope de Memória em CPU Pura vs. Limite EC2")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.95)

    for bars in (bd, br):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 70, f"{h:.0f}", ha="center", va="bottom", fontsize=8.2)

    topos = ["2× t3.small\n(2 GiB cada)", "1× t3.medium\n(4 GiB contíguos)"]
    yt = np.arange(len(topos))
    # Tarifa us-east-1 Linux On-Demand: 0.0208/h * 730 = 15.18/mês por small -> 30.37/mês; medium = 0.0416/h = 30.37/mês
    compute_usd = [30.37, 30.37]
    colors_topo = ["#d62728", "#2ca02c"]
    ax2.barh(yt, compute_usd, height=0.44, color=colors_topo, alpha=0.88)
    ax2.set_yticks(yt)
    ax2.set_yticklabels(topos, fontsize=9.8, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 56)
    ax2.set_xlabel("Custo Mensal de Computação EC2 (US$/mês, us-east-1)")
    ax2.set_title("Decisão AWS: 2× t3.small vs. 1× t3.medium")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    notes = [
        "US$ 30,37/mês\n(2× EBS + latência de rede\n+ 1.7B Q4 estoura 2 GiB)",
        "US$ 30,37/mês\n(1× EBS + 0 ms rede local\n+ folga para LLM + stack)",
    ]
    for idx, (cost, note) in enumerate(zip(compute_usd, notes)):
        ax2.text(cost + 0.9, idx, note, va="center", ha="left", fontsize=8.4, fontweight="bold", color="#1b1b1b")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _add_cell_borders(ax: Any) -> None:
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=3.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.2)


def _plot_matrices(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"figure.dpi": 140})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))

    row_names = ["Qwen 1.7B", "Qwen 0.6B"]
    col_names = ["Original (16 bits)", "Quantizado (4 bits)"]
    models = ["Qwen 1.7B", "Qwen 0.6B"]

    mem_mat = np.zeros((2, 2))
    acc_mat = np.zeros((2, 2))
    mem_labels = [["", ""], ["", ""]]
    acc_labels = [["", ""], ["", ""]]

    for i, m in enumerate(models):
        orig = df[(df["modelo_base"] == m) & (df["etapa_chave"] == "3_rag_lora_agente")].iloc[0]
        quant = df[(df["modelo_base"] == m) & (df["etapa_chave"] == "4_gguf_q4km_cpu2t")].iloc[0]

        mem_orig = float(orig["memoria_execucao_mib"])
        mem_quant = float(quant["memoria_execucao_mib"])

        mem_mat[i, 0] = mem_orig
        mem_mat[i, 1] = mem_quant
        mem_labels[i][0] = f"{mem_orig:,.0f} MiB\nVRAM (GPU)".replace(",", ".")
        mem_labels[i][1] = f"{mem_quant:,.0f} MiB\nRAM (CPU)".replace(",", ".")

        acc_orig = float(orig["acuracia_geral_estrita"]) * 100
        acc_quant = float(quant["acuracia_geral_estrita"]) * 100
        acc_mat[i, 0] = acc_orig
        acc_mat[i, 1] = acc_quant
        acc_labels[i][0] = f"{acc_orig:.1f}%\n({int(round(acc_orig * 0.3))}/30 casos)"
        acc_labels[i][1] = f"{acc_quant:.1f}%\n({int(round(acc_quant * 0.3))}/30 casos)"

    # Esquerda: Acurácia Geral Estrita
    im1 = ax1.imshow(acc_mat, cmap="YlGn", vmin=50, vmax=100, aspect="auto")
    ax1.set_title("Acurácia Geral Estrita (LoRA + RAG + Agente)", fontsize=11, pad=10)
    ax1.set_xticks([0, 1], labels=col_names, fontsize=9.5)
    ax1.set_yticks([0, 1], labels=row_names, fontsize=9.5)
    _add_cell_borders(ax1)
    for i in range(2):
        for j in range(2):
            color = "white" if acc_mat[i, j] > 78 else "#1b1b1b"
            ax1.text(j, i, acc_labels[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="%")

    # Direita: Uso de Memória em Execução
    im2 = ax2.imshow(mem_mat, cmap="YlOrRd", vmin=500, vmax=4600, aspect="auto")
    ax2.set_title("Uso de Memória (LoRA + RAG + Agente)", fontsize=11, pad=10)
    ax2.set_xticks([0, 1], labels=col_names, fontsize=9.5)
    ax2.set_yticks([0, 1], labels=row_names, fontsize=9.5)
    _add_cell_borders(ax2)
    for i in range(2):
        for j in range(2):
            color = "white" if mem_mat[i, j] > 2800 else "#1b1b1b"
            ax2.text(j, i, mem_labels[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="MiB")

    fig.suptitle("Matriz Tamanho × Quantização (Arquitetura Completa: LoRA + RAG + Agente)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fill_perf_panel(
    ax: Any,
    df: pd.DataFrame,
    etapas_cols: tuple[str, str],
    panel_title: str,
    metric: str,
    cmap: str,
    cbar_label: str,
) -> Any:
    """Preenche um subplot 2x2 (1.7B/0.6B x 16b/4b) para latência ou TPS."""
    row_names = ["Qwen 1.7B", "Qwen 0.6B"]
    col_names = ["Original (16 bits)", "Quantizado (4 bits)"]
    mat = np.zeros((2, 2))
    labels = [["", ""], ["", ""]]

    for i, m in enumerate(row_names):
        for j, etapa in enumerate(etapas_cols):
            rec = df[(df["modelo_base"] == m) & (df["etapa_chave"] == etapa)].iloc[0]
            if metric == "latency":
                val = float(rec["latencia_media_ms"]) / 1000.0
                p95 = float(rec["latencia_p95_ms"]) / 1000.0
                mat[i, j] = val
                labels[i][j] = f"{val:.2f} s\n(P95: {p95:.2f} s)"
            else:
                val = float(rec["tps_medio"])
                dec = float(rec["tps_decode_medio"]) if pd.notna(rec.get("tps_decode_medio")) else val
                mat[i, j] = val
                labels[i][j] = f"{val:.1f} tok/s\n({dec:.1f} decode)"

    vmin = float(mat.min()) * 0.80
    vmax = float(mat.max()) * 1.12
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(panel_title, fontsize=11, pad=10)
    ax.set_xticks([0, 1], labels=col_names, fontsize=9.5)
    ax.set_yticks([0, 1], labels=row_names, fontsize=9.5)
    _add_cell_borders(ax)
    thresh = vmin + (vmax - vmin) * 0.62
    for i in range(2):
        for j in range(2):
            color = "white" if mat[i, j] >= thresh else "#1b1b1b"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    return im


def _plot_perf_matrix_pair(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Renderiza par de matrizes 2x2 lado a lado (GPU Vulkan vs. CPU Pura 2t)."""
    plt.rcParams.update({"figure.dpi": 140})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7))

    if metric == "latency":
        cmap, cbar_label = "YlOrRd", "Latência Média (s)"
        suptitle = "Par de Matrizes 2×2 — Latência Média e P95 por Hardware (llama.cpp)"
        t_gpu = "GPU (Vulkan -ngl 99)"
        t_cpu = "CPU Pura (2 threads -ngl 0)"
    else:
        cmap, cbar_label = "YlGn", "Velocidade Média (tok/s)"
        suptitle = "Par de Matrizes 2×2 — Velocidade Ponta a Ponta (TPS) por Hardware (llama.cpp)"
        t_gpu = "GPU (Vulkan -ngl 99)"
        t_cpu = "CPU Pura (2 threads -ngl 0)"

    im1 = _fill_perf_panel(ax1, df, ("gguf_gpu_f16", "gguf_gpu_q4km"), t_gpu, metric, cmap, cbar_label)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=cbar_label)

    im2 = _fill_perf_panel(ax2, df, ("gguf_cpu_f16", "4_gguf_q4km_cpu2t"), t_cpu, metric, cmap, cbar_label)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=cbar_label)

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mem_matrix = json.loads((DIR_EVAL / "matriz_memoria_2x2.json").read_text(encoding="utf-8"))
    rows = []
    rows.extend(_rows_from_gpu_csv(DIR_EVAL / "placar_comparativo.csv", "Qwen 1.7B", mem_matrix["1.7B"]))
    rows.extend(_rows_from_gpu_csv(DIR_EVAL / "06b_placar_comparativo.csv", "Qwen 0.6B", mem_matrix["0.6B"]))

    gguf_specs = [
        ("gguf_17b_gpu_f16_report.json", "Qwen 1.7B", "GGUF 16b (GPU Vulkan)", "gguf_gpu_f16", "4g. LoRA + RAG + Agente (16b GPU)", None),
        ("gguf_17b_gpu_q4_report.json", "Qwen 1.7B", "GGUF 4b (GPU Vulkan)", "gguf_gpu_q4km", "4g. LoRA + RAG + Agente (4b GPU)", None),
        ("gguf_17b_cpu_f16_report.json", "Qwen 1.7B", "GGUF 16b (CPU 2t)", "gguf_cpu_f16", "4c. LoRA + RAG + Agente (16b CPU 2t)", mem_matrix["1.7B"].get("f16_cpu_rss_mib")),
        ("gguf_17b_cpu_np1_report.json", "Qwen 1.7B", "GGUF 4b (CPU 2t)", "4_gguf_q4km_cpu2t", STAGE_LABELS["4_gguf_q4km_cpu2t"], mem_matrix["1.7B"]["cpu_rss_mib"]),
        ("gguf_06b_gpu_f16_report.json", "Qwen 0.6B", "GGUF 16b (GPU Vulkan)", "gguf_gpu_f16", "4g. LoRA + RAG + Agente (16b GPU)", None),
        ("gguf_06b_gpu_q4_report.json", "Qwen 0.6B", "GGUF 4b (GPU Vulkan)", "gguf_gpu_q4km", "4g. LoRA + RAG + Agente (4b GPU)", None),
        ("gguf_06b_cpu_f16_report.json", "Qwen 0.6B", "GGUF 16b (CPU 2t)", "gguf_cpu_f16", "4c. LoRA + RAG + Agente (16b CPU 2t)", mem_matrix["0.6B"].get("f16_cpu_rss_mib")),
        ("gguf_06b_cpu_report.json", "Qwen 0.6B", "GGUF 4b (CPU 2t)", "4_gguf_q4km_cpu2t", STAGE_LABELS["4_gguf_q4km_cpu2t"], mem_matrix["0.6B"]["cpu_rss_mib"]),
    ]
    for fname, mname, fmt, etapa, cfg_lbl, rss_ovr in gguf_specs:
        p = DIR_EVAL / fname
        if p.exists():
            rows.append(_row_from_gguf_json(p, mname, fmt, etapa, cfg_lbl, rss_ovr))

    df = pd.DataFrame(rows)
    out_csv = DIR_EVAL / "placar_comparativo_modelos.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    out_png = DIR_ASSETS / "comparativo_17b_vs_06b_gguf.png"
    _plot_ec2_sizing_and_cost(mem_matrix, out_png)

    out_mat = DIR_ASSETS / "matriz_2x2_memoria_acuracia.png"
    _plot_matrices(df, out_mat)

    out_lat = DIR_ASSETS / "matriz_2x2_latencia_gpu_cpu.png"
    _plot_perf_matrix_pair(df, "latency", out_lat)

    out_tps = DIR_ASSETS / "matriz_2x2_tps_gpu_cpu.png"
    _plot_perf_matrix_pair(df, "tps", out_tps)

    stale_chart = DIR_ASSETS / "latencia_vs_velocidade_modelos.png"
    if stale_chart.exists():
        stale_chart.unlink()

    print(f"CSV salvo em {out_csv}")
    print(f"Graficos salvos em {out_png}, {out_mat}, {out_lat} e {out_tps}")


if __name__ == "__main__":
    main()
