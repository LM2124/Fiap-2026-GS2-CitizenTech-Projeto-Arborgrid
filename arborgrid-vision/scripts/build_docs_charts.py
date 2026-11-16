"""
Gera e consolida os gráficos em alta resolução para a documentação de `arborgrid-vision/` (`assets/`).

Todos os números são lidos diretamente dos relatórios JSON salvos em `runs/`, do pacote
`arborgrid_v3_gold_results_fixed.zip` e do manifesto operacional `dataset/sp_geosampa_api/manifest.json`.
Nenhuma fotografia do Google Street View é copiada para `assets/`.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CV_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = CV_ROOT / "assets"
RUNS_DIR = CV_ROOT / "runs"
MANIFEST_PATH = CV_ROOT / "dataset" / "sp_geosampa_api" / "manifest.json"
V3_ZIP_PATH = CV_ROOT / "arborgrid_v3_gold_results_fixed.zip"


def _load_v3_report() -> dict:
    with zipfile.ZipFile(V3_ZIP_PATH, "r") as zf:
        return json.loads(zf.read("relatorio_ab_benchmark_v3.json").decode("utf-8"))


def build_etapa1_ablation_v1() -> Path:
    """Ablação controlada V1 (68 imgs @ 1024x1024): Pré-label GPU vs Equipe Inicial vs Curadoria Ouro."""
    data = json.loads((RUNS_DIR / "pilot_ablation_1024" / "ablation_metrics_1024px.json").read_text(encoding="utf-8"))

    models = ["A", "B", "C"]
    labels = [
        "Modelo A\n(Pré-label GPU Bruto)",
        "Modelo B\n(Rotulagem Equipe Inicial)",
        "Modelo C — Ouro V1\n(Curadoria Integral)",
    ]
    x = np.arange(len(models))
    w = 0.19

    fiacao_iou = [data[m]["fiacao"]["iou"] for m in models]
    poste_iou = [data[m]["poste"]["iou"] for m in models]
    copa_iou = [data[m]["copa"]["iou"] for m in models]
    miou = [data[m]["mean_iou"] for m in models]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    b1 = ax.bar(x - 1.5 * w, fiacao_iou, w, label="IoU Fiação (%)", color="#dc2626")
    b2 = ax.bar(x - 0.5 * w, poste_iou, w, label="IoU Poste (%)", color="#0284c7")
    b3 = ax.bar(x + 0.5 * w, copa_iou, w, label="IoU Copa (%)", color="#16a34a")
    b4 = ax.bar(x + 1.5 * w, miou, w, label="mIoU Geral (%)", color="#1e293b")

    for bar in list(b1) + list(b2) + list(b3) + list(b4):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("Etapa 1 — Impacto da Curadoria Humana na Validação Isolada V1 (1024×1024, 68 imagens)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Métrica na Validação Isolada (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=4, fontsize=9)

    out = ASSETS_DIR / "etapa1_ablation_v1_qualidade_resolucao.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def build_etapa2_evolucao_v1_v2_v3() -> Path:
    """Evolução V1 -> V2 -> Matriz A/B/C V3 (Validação Isolada + Teste Cego Paulista)."""
    v2_data = json.loads((RUNS_DIR / "full_v2_132" / "metricas_v2_132imgs.json").read_text(encoding="utf-8"))
    v3_data = _load_v3_report()
    mv = v3_data["metrics_validation"]
    fp = v3_data["false_positive_suppression_paulista"]

    stages = [
        "V1 Ouro\n(68 imgs)",
        "V2 Full\n(132 imgs)",
        "V3 Mod A\n(Base 117)",
        "V3 Mod B\n(+Aug Segura)",
        "V3 Mod C Ouro\n(+HardNeg 135)",
    ]
    fiacao = [
        v2_data["metrics_v1_68imgs"]["fiacao"]["iou"],
        v2_data["metrics_v2_132imgs"]["fiacao"]["iou"],
        mv["model_A"]["fiacao_iou"],
        mv["model_B"]["fiacao_iou"],
        mv["model_C"]["fiacao_iou"],
    ]
    poste_prec = [
        v2_data["metrics_v1_68imgs"]["poste"]["precision"],
        v2_data["metrics_v2_132imgs"]["poste"]["precision"],
        mv["model_A"]["poste_precision"],
        mv["model_B"]["poste_precision"],
        mv["model_C"]["poste_precision"],
    ]
    miou = [
        v2_data["metrics_v1_68imgs"]["mean_iou"],
        v2_data["metrics_v2_132imgs"]["mean_iou"],
        mv["model_A"]["miou"],
        mv["model_B"]["miou"],
        mv["model_C"]["miou"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=160, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Etapa 2 & 3 — Evolução V1 → V2 → Matriz A/B/C V3 (Validação vs. Supressão Paulista)", fontsize=13.5, fontweight="bold", y=0.98)

    # Painel Esquerdo: Validação Isolada
    x1 = np.arange(len(stages))
    w1 = 0.25
    b1 = ax1.bar(x1 - w1, fiacao, w1, label="IoU Fiação (%)", color="#2563eb")
    b2 = ax1.bar(x1, miou, w1, label="mIoU Geral (%)", color="#10b981")
    b3 = ax1.bar(x1 + w1, poste_prec, w1, label="Precisão Poste (%)", color="#6366f1")
    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.3, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax1.set_title("Validação Isolada (8 fotos · 4 árvores congeladas)", fontweight="bold", pad=10)
    ax1.set_ylabel("Métrica (%)")
    ax1.set_ylim(0, 112)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(stages, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Painel Direito: Teste Cego Av. Paulista (20 fotos sem rede aérea)
    ab_keys = ["model_A", "model_B", "model_C"]
    ab_labels = ["V3 Mod A\n(Base V2)", "V3 Mod B\n(+Aug Segura)", "V3 Mod C Ouro\n(+Hard Negs)"]
    x2 = np.arange(len(ab_keys))
    mean_fp = [fp[k]["mean_fp_pixels"] for k in ab_keys]
    zero_fp = [fp[k]["zero_fp_rate"] for k in ab_keys]

    ax2_r = ax2.twinx()
    bars_fp = ax2.bar(x2 - 0.18, mean_fp, 0.34, color=["#64748b", "#f59e0b", "#16a34a"], label="Pixels Falsos Médios (menor = melhor)")
    bars_z = ax2_r.bar(x2 + 0.18, zero_fp, 0.34, color="#0ea5e9", alpha=0.88, label="Imagens Zero-FP (%)")

    for bar in bars_fp:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 180, f"{h:,.0f} px", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for bar in bars_z:
        h = bar.get_height()
        ax2_r.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0369a1")

    ax2.set_title("Teste Cego Av. Paulista (20 fotos · 0% rede aérea)", fontweight="bold", pad=10)
    ax2.set_ylabel("Pixels Falsos Médios por Foto (px)")
    ax2_r.set_ylabel("Taxa Zero-FP (%)")
    ax2.set_ylim(0, 10200)
    ax2_r.set_ylim(0, 100)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ab_labels, fontsize=8.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    out = ASSETS_DIR / "etapa2_evolucao_v1_v2_v3.png"
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def build_etapa3_territorios_paulista() -> Path:
    """Impacto territorial V2 -> V3 em campo (271 árvores) e distribuição final V3 por região."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    trees = manifest["trees"]

    regions = ["pinheiros", "alto_da_mooca", "vila_mariana", "avenida_paulista"]
    reg_labels = [
        "Pinheiros\n(89 árvores · Aérea)",
        "Alto da Mooca\n(54 árvores · Aérea)",
        "Vila Mariana\n(54 árvores · Aérea)",
        "Av. Paulista\n(74 árvores · Subterrânea)",
    ]

    wire_delta = []
    pct_crit = []
    pct_prox = []
    pct_par = []
    pct_safe = []

    for rk in regions:
        sub = [t for t in trees if t.get("region_key") == rk]
        v2_px = sum(int((v.get("v2_analysis") or {}).get("wire_px", 0)) for t in sub for v in t.get("views", []))
        v3_px = sum(int((v.get("v3_analysis") or {}).get("wire_px", 0)) for t in sub for v in t.get("views", []))
        wire_delta.append(100.0 * (v3_px - v2_px) / max(v2_px, 1))

        n = max(len(sub), 1)
        c_crit = c_prox = c_par = c_safe = 0
        for t in sub:
            st = (t.get("v3_triage") or {}).get("consolidated", "SEGURO")
            if st == "CRITICO_CONFIRMADO":
                c_crit += 1
            elif st == "DIVERGENCIA_PARALAXE":
                c_par += 1
            elif "PROXIMIDADE" in st or "CRITICO_MONO" in st:
                c_prox += 1
            else:
                c_safe += 1
        pct_crit.append(100.0 * c_crit / n)
        pct_par.append(100.0 * c_par / n)
        pct_prox.append(100.0 * c_prox / n)
        pct_safe.append(100.0 * c_safe / n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Etapa 3 (Campo) — Efeito dos Hard Negatives nas 271 Árvores (536 vistas GeoSampa)", fontsize=13.5, fontweight="bold", y=0.98)

    # Painel Esquerdo: Variação de pixels de fiação V2 -> V3
    x = np.arange(len(regions))
    colors_delta = ["#64748b", "#64748b", "#64748b", "#16a34a"]
    bars = ax1.bar(x, wire_delta, 0.52, color=colors_delta)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h - 3.8, f"{h:.1f}%", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax1.axhline(0, color="#334155", linewidth=0.9)
    ax1.set_title("Redução de Pixels Preditos como Fiação (V2 → V3 Ouro)", fontweight="bold", pad=10)
    ax1.set_ylabel("Variação de wire_px (%)")
    ax1.set_ylim(-68, 8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_labels, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Painel Direito: Triagem consolidada (%)
    arr_safe = np.array(pct_safe)
    arr_prox = np.array(pct_prox)
    arr_par = np.array(pct_par)
    arr_crit = np.array(pct_crit)

    ax2.bar(x, arr_crit, 0.55, label="Crítico Confirmado", color="#dc2626")
    ax2.bar(x, arr_par, 0.55, bottom=arr_crit, label="Divergência de Paralaxe", color="#7c3aed")
    ax2.bar(x, arr_prox, 0.55, bottom=arr_crit + arr_par, label="Proximidade / Mono-Visada", color="#f59e0b")
    ax2.bar(x, arr_safe, 0.55, bottom=arr_crit + arr_par + arr_prox, label="Seguro", color="#10b981")

    for i in range(len(regions)):
        ax2.text(x[i], arr_crit[i] / 2, f"{arr_crit[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax2.text(x[i], arr_crit[i] + arr_par[i] + arr_prox[i] + arr_safe[i] / 2, f"{arr_safe[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    ax2.set_title("Triagem Consolidada Multi-Perspectiva (Modelo Ouro V3)", fontweight="bold", pad=10)
    ax2.set_ylabel("Distribuição por Território (%)")
    ax2.set_ylim(0, 118)
    ax2.set_xticks(x)
    ax2.set_xticklabels(reg_labels, fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper center", ncol=2, fontsize=8.2)

    out = ASSETS_DIR / "etapa3_territorios_v2_vs_v3_paulista.png"
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def sync_etapa4_quantization_chart() -> Path:
    src = RUNS_DIR / "quantization_benchmark" / "quantizacao_segformer_b2_v3.png"
    dst = ASSETS_DIR / "etapa4_quantizacao_segformer_v3.png"
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    generated = [
        build_etapa1_ablation_v1(),
        build_etapa2_evolucao_v1_v2_v3(),
        build_etapa3_territorios_paulista(),
        sync_etapa4_quantization_chart(),
    ]
    for p in generated:
        print(f"OK -> {p}")


if __name__ == "__main__":
    main()
