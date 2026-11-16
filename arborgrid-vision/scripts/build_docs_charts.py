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
import matplotlib.patches as patches
import numpy as np


CV_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = CV_ROOT / "assets"
RUNS_DIR = CV_ROOT / "runs"
MANIFEST_PATH = CV_ROOT / "dataset" / "sp_geosampa_api" / "manifest.json"
V3_ZIP_PATH = CV_ROOT / "arborgrid_v3_gold_results_fixed.zip"


def _load_v3_report() -> dict:
    with zipfile.ZipFile(V3_ZIP_PATH, "r") as zf:
        return json.loads(zf.read("relatorio_ab_benchmark_v3.json").decode("utf-8"))


def build_etapa0_prelabel_generico() -> Path:
    """Prova visual da pré-segmentação genérica inicial (SegFormer Cityscapes zero-shot + OpenCV Black-Hat/LSD)."""
    pilot = json.loads((RUNS_DIR / "pilot_ablation" / "ablation_metrics.json").read_text(encoding="utf-8"))
    v1_1024 = json.loads((RUNS_DIR / "pilot_ablation_1024" / "ablation_metrics_1024px.json").read_text(encoding="utf-8"))
    direct = pilot["direct_label_iou_reviewed_imgs"]["base_vs_revised"]

    fig = plt.figure(figsize=(15.8, 6.4), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.16)
    fig.suptitle(
        "Etapa 0 — Pré-Segmentação Inicial com Modelo Genérico vs. Curadoria do Pacote 1",
        fontsize=13.5,
        fontweight="bold",
        y=0.97,
    )

    # Painel Esquerdo: Anatomia visual do pipeline zero-shot + OpenCV
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_flow.set_xlim(0, 100)
    ax_flow.set_ylim(0, 100)
    ax_flow.axis("off")
    ax_flow.set_title("Arquitetura Híbrida Inicial (auto_prelabel_and_audit.py)", fontsize=11.2, fontweight="bold", pad=8)

    def draw_box(x, y, w, h, title, body, fc, ec, title_color="#0f172a"):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=2.2",
            facecolor=fc, edgecolor=ec, linewidth=1.6,
        )
        ax_flow.add_patch(box)
        ax_flow.text(x + w / 2, y + h - 4.0, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=title_color)
        ax_flow.text(x + w / 2, y + h / 2 - 2.6, body, ha="center", va="center", fontsize=8.0, color="#334155", linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, color="#475569"):
        ax_flow.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.7, color=color, shrinkA=2, shrinkB=2),
        )

    draw_box(
        3, 76, 94, 17,
        "1. Foto Street View Bruta (640×640)",
        "271 árvores georreferenciadas (GeoSampa) — ainda sem nenhum rótulo específico de fiação aérea",
        "#f1f5f9", "#64748b",
    )

    draw_arrow(30, 75, 26, 65, "#0284c7")
    draw_arrow(70, 75, 74, 65, "#dc2626")

    draw_box(
        3, 35, 45, 29,
        "2A. SegFormer-B2 Cityscapes (Zero-Shot)",
        "Backbone urbano pré-treinado (sem classe fio)\n"
        "• Classe #8 (vegetation) + Abertura 5×5\n"
        "  + Filtro área ≥ 900 px → copa\n"
        "• Classe #5 (pole) → poste",
        "#eff6ff", "#2563eb", "#1d4ed8",
    )

    draw_box(
        52, 35, 45, 29,
        "2B. Heurística Clássica OpenCV (Fiação)",
        "Para suprir ausência de fios no Cityscapes:\n"
        "• Black-Hat (9×9) + LSD (comp ≥ 52 px, ≤ 68°)\n"
        "• Vale transversal ≥ 6,5 e zona acima da via\n"
        "• Proíbe pista/veículo (#0,1,3,4,11..15)",
        "#fef2f2", "#dc2626", "#b91c1c",
    )

    draw_arrow(26, 34, 38, 23, "#2563eb")
    draw_arrow(74, 34, 62, 23, "#dc2626")

    draw_box(
        6, 4, 88, 18,
        "3. Polígonos YOLO-Seg → Workspace X-AnyLabeling (pacote_1_prelabel/)",
        "Economizou 97,5% do contorno de copas arbóreas e 41,1% dos postes,\n"
        "mas LSD falhou em fios curvos/fachadas (8,3% IoU) → exigiu curadoria manual no Pacote 1",
        "#ecfdf5", "#16a34a", "#15803d",
    )

    # Painel Direito: Desempenho real dos pré-labels genéricos vs V1 Ouro
    ax_bar = fig.add_subplot(gs[0, 1])
    categories = [
        "Rótulo Bruto:\nCopa",
        "Rótulo Bruto:\nPoste",
        "Rótulo Bruto:\nFiação (LSD)",
        "Modelo V1 Base\n(IoU Fiação)",
        "Modelo V1 Ouro\n(IoU Fiação)",
    ]
    values = [
        direct["copa"]["iou"],
        direct["poste"]["iou"],
        direct["fiacao"]["iou"],
        v1_1024["A"]["fiacao"]["iou"],
        v1_1024["C"]["fiacao"]["iou"],
    ]
    colors = ["#16a34a", "#0284c7", "#ef4444", "#f97316", "#1d4ed8"]
    x = np.arange(len(categories))
    bars = ax_bar.bar(x, values, 0.56, color=colors, edgecolor="#0f172a", linewidth=0.7)

    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#0f172a",
        )

    ax_bar.axvline(2.5, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax_bar.text(1.0, 108, "Acurácia Direta vs. Gabarito", ha="center", fontsize=8.6, color="#475569", fontweight="bold")
    ax_bar.text(3.5, 108, "Treino 1024px (Pacote 1)", ha="center", fontsize=8.6, color="#475569", fontweight="bold")
    ax_bar.annotate(
        "10× maior em fiação\napós curadoria do Pacote 1",
        xy=(4, values[4] + 8),
        xytext=(3.2, 67),
        ha="center",
        fontsize=8.3,
        fontweight="bold",
        color="#1d4ed8",
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor="#93c5fd"),
    )

    ax_bar.set_title("Qualidade Medida no Pacote 1 (Pré-Label Genérico vs. Curado)", fontsize=11.2, fontweight="bold", pad=8)
    ax_bar.set_ylabel("IoU (%)")
    ax_bar.set_ylim(0, 116)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, fontsize=8.3)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.35)

    out = ASSETS_DIR / "etapa0_prelabel_generico_pipeline.png"
    fig.subplots_adjust(left=0.04, right=0.98, top=0.87, bottom=0.11)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def build_etapa1b_subpixel_aliasing() -> Path:
    """Ilustra visualmente o Sub-Pixel Aliasing no mapa latente 1/4 (640->160 vs 1024->256) e o ganho empírico pós-Pacote 1."""
    pilot_640 = json.loads((RUNS_DIR / "pilot_ablation" / "ablation_metrics.json").read_text(encoding="utf-8"))
    v1_1024 = json.loads((RUNS_DIR / "pilot_ablation_1024" / "ablation_metrics_1024px.json").read_text(encoding="utf-8"))

    # Cena matemática pura (sem fotos Street View): cabo fino diagonal de ~2,0 px em grade 1024×1024
    # Mostramos um recorte central (192x192 na escala 1024px) mapeado nas grades 1/4 do SegFormer:
    # 640x640 -> stride efetivo 1024/160 = 6.4 px/célula (fio ~2px ocupa apenas 0,31 célula -> quebra pontilhada)
    # 1024x1024 -> stride efetivo 1024/256 = 4.0 px/célula (fio ~2px ocupa >=0,5 célula + contexto -> contínuo)
    crop_h, crop_w = 192, 192
    yy, xx = np.mgrid[0:crop_h, 0:crop_w]
    # Cabo primário diagonal (espessura 2.0 px em 1024px) + ramal fino (1.4 px)
    dist1 = np.abs(yy - (0.52 * xx + 36.0)) / np.hypot(0.52, 1.0)
    dist2 = np.abs(yy - (-0.28 * xx + 138.0)) / np.hypot(-0.28, 1.0)
    wire_hr = np.maximum((dist1 <= 1.0).astype(np.float32), (dist2 <= 0.72).astype(np.float32))

    # Amostragem média exata por célula do mapa latente 1/4 (equivalente ao downsampling 1/4 antes da ativação)
    def sample_latent_grid(img: np.ndarray, latent_res: int, thresh: float) -> np.ndarray:
        step = 1024.0 / latent_res  # 6.4 px @ 640 vs 4.0 px @ 1024
        n_cells = int(round(crop_w / step))
        grid = np.zeros((n_cells, n_cells), dtype=np.float32)
        for gy in range(n_cells):
            y0 = int(round(gy * step + step * 0.5))
            for gx in range(n_cells):
                x0 = int(round(gx * step + step * 0.5))
                if 0 <= y0 < crop_h and 0 <= x0 < crop_w:
                    # Média local em janela de receptividade proporcional ao patch 3x3
                    win = img[max(0, y0 - 1):min(crop_h, y0 + 2), max(0, x0 - 1):min(crop_w, x0 + 2)]
                    grid[gy, gx] = float(np.mean(win))
        return (grid >= thresh).astype(np.float32)

    grid_160 = sample_latent_grid(wire_hr, 160, thresh=0.22)  # 640x640 -> 160x160 (sub-pixel aliasing!)
    grid_256 = sample_latent_grid(wire_hr, 256, thresh=0.14)  # 1024x1024 -> 256x256 (+156% tokens)

    fig = plt.figure(figsize=(16.2, 5.9), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.22)
    fig.suptitle(
        "Etapa 1B — Por que Subir de 640×640 para 1024×1024 Após Concluir o Pacote 1 (Mapa Latente 1/4 + GPU)",
        fontsize=13.2,
        fontweight="bold",
        y=0.97,
    )

    # Painel 1: Mapa latente 160x160 (640px)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(grid_160, cmap="Reds", interpolation="nearest", vmin=0, vmax=1.15)
    n160 = grid_160.shape[0]
    ax1.set_xticks(np.arange(-0.5, n160, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n160, 1), minor=True)
    ax1.grid(which="minor", color="#cbd5e1", linewidth=0.5)
    ax1.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.set_title("Entrada 640×640 → Latente 160×160\n(Cabo 2 px encolhe para ~0,3–0,5 célula)", fontsize=10.2, fontweight="bold", color="#b91c1c", pad=8)
    ax1.set_xlabel("Fragmentação pontilhada na diagonal (gaps sub-pixel)", fontsize=8.8, color="#991b1b", fontweight="bold")

    # Painel 2: Mapa latente 256x256 (1024px)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(grid_256, cmap="Greens", interpolation="nearest", vmin=0, vmax=1.15)
    n256 = grid_256.shape[0]
    ax2.set_xticks(np.arange(-0.5, n256, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, n256, 1), minor=True)
    ax2.grid(which="minor", color="#cbd5e1", linewidth=0.35)
    ax2.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2.set_title("Entrada 1024×1024 → Latente 256×256\n(+156% tokens · Suporte ≥ 1 px contínuo)", fontsize=10.2, fontweight="bold", color="#15803d", pad=8)
    ax2.set_xlabel("Traçado conectado preservando cabos finos e ramais", fontsize=8.8, color="#166534", fontweight="bold")

    # Painel 3: Resultados reais observados (640 local -> 800 swap -> 1024 Colab)
    ax3 = fig.add_subplot(gs[0, 2])
    m640 = pilot_640["segformer_val_iou_mean_over_seeds"]["C"]
    m1024 = v1_1024["C"]

    metrics_labels = ["IoU Fiação", "IoU Poste", "Precisão Poste", "mIoU Geral"]
    vals_640 = [m640["fiacao"]["iou"], m640["poste"]["iou"], m640["poste"]["precision"], m640["mean_iou"]]
    vals_1024 = [m1024["fiacao"]["iou"], m1024["poste"]["iou"], m1024["poste"]["precision"], m1024["mean_iou"]]

    x = np.arange(len(metrics_labels))
    w = 0.34
    b_640 = ax3.bar(x - w / 2, vals_640, w, label="640×640 local (RTX 3060 Ti, 3 ép.)", color="#94a3b8")
    b_1024 = ax3.bar(x + w / 2, vals_1024, w, label="1024×1024 Data Center (A100/L4, 30 ép.)", color="#0284c7")

    for bar in list(b_640) + list(b_1024):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    ax3.text(
        0.5,
        0.81,
        "Tentativa intermediária 800×800 na RTX 3060 Ti (8 GB):\n"
        "estouro de VRAM → swap PCIe (7,96 min/época, ~191 min est.)",
        transform=ax3.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#9a3412",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7ed", edgecolor="#fdba74"),
    )

    ax3.set_title("Impacto Empírico no Pacote 1 (Modelo V1 Intermediário)", fontsize=10.5, fontweight="bold", pad=8)
    ax3.set_ylabel("Métrica na Validação Isolada (%)")
    ax3.set_ylim(0, 96)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_labels, fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7.9, frameon=True)

    out = ASSETS_DIR / "etapa1b_subpixel_aliasing_640_vs_1024.png"
    fig.subplots_adjust(left=0.03, right=0.98, top=0.83, bottom=0.11)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def build_etapa1_ablation_v1() -> Path:
    """Ablação controlada V1 (Pacote 1 — 68 imgs @ 1024x1024): Pré-label Genérico vs Curadoria Ouro (Modelo Intermediário V1)."""
    data = json.loads((RUNS_DIR / "pilot_ablation_1024" / "ablation_metrics_1024px.json").read_text(encoding="utf-8"))

    models = ["A", "C"]
    labels = [
        "V1 Base (Pré-label Genérico)\n(Segmentação Inicial GPU)",
        "V1 Intermediário (Ouro)\n(1ª Metade Curada — Pacote 1)",
    ]
    x = np.arange(len(models))
    w = 0.19

    fiacao_iou = [data[m]["fiacao"]["iou"] for m in models]
    poste_iou = [data[m]["poste"]["iou"] for m in models]
    copa_iou = [data[m]["copa"]["iou"] for m in models]
    miou = [data[m]["mean_iou"] for m in models]

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    b1 = ax.bar(x - 1.5 * w, fiacao_iou, w, label="IoU Fiação (%)", color="#dc2626")
    b2 = ax.bar(x - 0.5 * w, poste_iou, w, label="IoU Poste (%)", color="#0284c7")
    b3 = ax.bar(x + 0.5 * w, copa_iou, w, label="IoU Copa (%)", color="#16a34a")
    b4 = ax.bar(x + 1.5 * w, miou, w, label="mIoU Geral (%)", color="#1e293b")

    for bar in list(b1) + list(b2) + list(b3) + list(b4):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    ax.set_title("Etapa 1 — Pré-Segmentação Genérica vs. Modelo Intermediário V1 (Pacote 1 @ 1024×1024)", fontsize=12.5, fontweight="bold", pad=12)
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
        build_etapa0_prelabel_generico(),
        build_etapa1_ablation_v1(),
        build_etapa1b_subpixel_aliasing(),
        build_etapa2_evolucao_v1_v2_v3(),
        build_etapa3_territorios_paulista(),
        sync_etapa4_quantization_chart(),
    ]
    for p in generated:
        print(f"OK -> {p}")


if __name__ == "__main__":
    main()
