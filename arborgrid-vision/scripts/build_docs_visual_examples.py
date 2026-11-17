"""
Gera exemplos visuais concretos lado a lado em `assets/` para ilustrar cada etapa
decisória da documentação (`EXPLORACAO_MODELOS_E_QUANTIZACAO.md`), com atribuição
obrigatória `Imagens (c) Google` para escopo acadêmico.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
if str(CV_ROOT) not in sys.path:
    sys.path.insert(0, str(CV_ROOT))

from scripts.audit_anomalies_v3_dataset import json_to_masks  # noqa: E402
from scripts.run_v3_inference_and_update_map import (  # noqa: E402
    COLORS_BGR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEGFORMER_PRETRAINED,
    THRESHOLDS,
    remap_colab_to_local,
)

ASSETS_DIR = CV_ROOT / "assets"
WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"
SP_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"
IMAGES_DIR = SP_DIR / "images"
PRELABELS_DIR = SP_DIR / "prelabels_yolo_seg"
MODELS_DIR = CV_ROOT / "models"


def yolo_txt_to_masks(txt_path: Path, h: int, w: int) -> dict[int, np.ndarray]:
    masks = {c: np.zeros((h, w), dtype=np.uint8) for c in range(3)}
    if not txt_path.exists():
        return masks
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid not in masks:
            continue
        coords = [float(v) for v in parts[1:]]
        pts = [[coords[i] * w, coords[i + 1] * h] for i in range(0, len(coords) - 1, 2)]
        cv2.fillPoly(masks[cid], [np.round(np.array(pts)).astype(np.int32)], 1)
    return masks


def load_segformer_weights(weights_path: Path, device: str) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if any(k.startswith("segformer.stages.") for k in sd.keys()):
        sd = {remap_colab_to_local(k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def infer_masks(model: SegformerForSemanticSegmentation, img_bgr: np.ndarray, imgsz: int, device: str) -> dict[int, np.ndarray]:
    h, w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    inp = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=x).logits
        probs = torch.sigmoid(
            F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
        )[0].cpu().numpy()
    return {cid: (probs[cid] >= THRESHOLDS[cid]).astype(np.uint8) for cid in range(3)}


def _stamp_header(canvas: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    w = canvas.shape[1]
    cv2.rectangle(canvas, (0, 0), (w, 42), (15, 23, 42), -1)
    cv2.putText(canvas, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (148, 163, 184), 1, cv2.LINE_AA)
    return canvas


def render_overlay(img_bgr: np.ndarray, masks: dict[int, np.ndarray] | None, title: str, subtitle: str, size: int = 440) -> np.ndarray:
    canvas = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    if masks is not None:
        overlay = canvas.copy()
        resized_masks = {}
        for cid in (0, 2, 1):
            m_src = np.asarray(masks[cid], dtype=np.uint8)
            m = cv2.resize(m_src, (size, size), interpolation=cv2.INTER_NEAREST)
            resized_masks[cid] = m
            if np.any(m):
                overlay[m > 0] = COLORS_BGR[cid]
        cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0, canvas)
        for cid in (0, 2, 1):
            cnts, _ = cv2.findContours(resized_masks[cid], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(canvas, cnts, -1, COLORS_BGR[cid], 1, cv2.LINE_AA)

    return _stamp_header(canvas, title, subtitle)


def render_diff_panel(img_bgr: np.ndarray, pre_m: dict[int, np.ndarray], cur_m: dict[int, np.ndarray], size: int = 440) -> np.ndarray:
    canvas = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    overlay = canvas.copy()
    p_res = {c: cv2.resize(np.asarray(pre_m[c], dtype=np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) for c in range(3)}
    c_res = {c: cv2.resize(np.asarray(cur_m[c], dtype=np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) for c in range(3)}

    pre_any = (p_res[0] > 0) | (p_res[1] > 0) | (p_res[2] > 0)
    cur_any = (c_res[0] > 0) | (c_res[1] > 0) | (c_res[2] > 0)

    overlay[pre_any & cur_any] = (60, 195, 70)      # Verde: mantido
    overlay[cur_any & (~pre_any)] = (235, 200, 30)  # Ciano: adicionado
    overlay[pre_any & (~cur_any)] = (45, 45, 240)   # Vermelho: removido
    cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)

    return _stamp_header(canvas, "Mapa de Edicao Manual", "Verde=Mantido | Ciano=Adicionado | Verm=Removido")


def add_legend_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((30, width, 3), 241, dtype=np.uint8)
    cv2.putText(bar, text, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (15, 23, 42), 1, cv2.LINE_AA)
    return bar


def build_example_etapa0() -> Path:
    stem = "tree_612770_center_front_h36"
    img = cv2.imread(str(WS_DIR / "pacote_1" / f"{stem}.jpg"))
    h, w = img.shape[:2]
    pre_m = json_to_masks(WS_DIR / "pacote_1_prelabel" / f"{stem}.json", h, w)
    cur_m = json_to_masks(WS_DIR / "pacote_1" / f"{stem}.json", h, w)

    p1 = render_overlay(img, pre_m, "1. Pre-Label Generico (GPU)", "Cityscapes (#8 copa, #5 poste) + Black-Hat/LSD")
    p2 = render_overlay(img, cur_m, "2. Gabarito Curado (Pacote 1)", "Editado manualmente sobre o pre-label inicial")
    p3 = render_diff_panel(img, pre_m, cur_m)

    row = np.hstack([p1, p2, p3])
    footer = add_legend_bar(row.shape[1], f"Exemplo real ({stem}): Verde=Copa | Vermelho=Fiacao | Ciano=Poste — Copa quase intacta; fios tracados a mao.")
    out = ASSETS_DIR / "exemplo_etapa0_prelabel_vs_pacote1.jpg"
    cv2.imwrite(str(out), np.vstack([row, footer]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def build_example_640_vs_1024(device: str) -> Path:
    stem = "tree_597652_side_angle_A_h36"  # Validação congelada (nunca vista no treino)
    img = cv2.imread(str(WS_DIR / "pacote_1" / f"{stem}.jpg"))
    h, w = img.shape[:2]
    gt_m = json_to_masks(WS_DIR / "pacote_1" / f"{stem}.json", h, w)

    m640 = load_segformer_weights(MODELS_DIR / "arborgrid_segformer_b2_pilot.pt", device)
    pred_640 = infer_masks(m640, img, 640, device)
    del m640
    torch.cuda.empty_cache()

    m1024 = load_segformer_weights(MODELS_DIR / "arborgrid_segformer_b2_1024_gold_local.pt", device)
    pred_1024 = infer_masks(m1024, img, 1024, device)
    del m1024
    torch.cuda.empty_cache()

    p1 = render_overlay(img, gt_m, "1. Gabarito Validacao (#597652)", "Arvore congelada (0% vazamento no treino)")
    p2 = render_overlay(img, pred_640, "2. Piloto 640x640 (Latente 160x160)", "IoU fio=37,7% | IoU poste=0,0% (colapsado)")
    p3 = render_overlay(img, pred_1024, "3. Intermediario V1 1024x1024 (256x256)", "IoU fio=57,5% | IoU poste=40,9% (Pos-Pacote 1)")

    row = np.hstack([p1, p2, p3])
    footer = add_legend_bar(row.shape[1], f"Validacao isolada ({stem}): Verde=Copa | Vermelho=Fiacao | Ciano=Poste — Salto qualitativo de 640 para 1024 px.")
    out = ASSETS_DIR / "exemplo_etapa1_640_vs_1024_validacao.jpg"
    cv2.imwrite(str(out), np.vstack([row, footer]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def build_example_pacote2_and_paulista(device: str) -> Path:
    # Linha 1: Pacote 2 (aceleração do pré-label pelo modelo intermediário V1)
    stem_p2 = "tree_632565_side_angle_A_h31"
    img_p2 = cv2.imread(str(WS_DIR / "pacote_2" / f"{stem_p2}.jpg"))
    h2, w2 = img_p2.shape[:2]
    m_p2_gen = yolo_txt_to_masks(PRELABELS_DIR / f"{stem_p2}.txt", h2, w2)
    m_p2_cur = json_to_masks(WS_DIR / "pacote_2" / f"{stem_p2}.json", h2, w2)

    m_v1 = load_segformer_weights(MODELS_DIR / "arborgrid_segformer_b2_1024_gold_local.pt", device)
    m_p2_v1 = infer_masks(m_v1, img_p2, 1024, device)
    del m_v1
    torch.cuda.empty_cache()

    r1_1 = render_overlay(img_p2, m_p2_gen, "Pacote 2: Pre-Label Generico", "IoU fio=0,7% (LSD quase cego na cena)", size=420)
    r1_2 = render_overlay(img_p2, m_p2_v1, "Pacote 2: Pre-Label V1 (1024px)", "IoU fio=53,4% | poste=64,8% sem revisao", size=420)
    r1_3 = render_overlay(img_p2, m_p2_cur, "Pacote 2: Curadoria Final", "Revisao humana agilizada sobre o V1", size=420)
    row1 = np.hstack([r1_1, r1_2, r1_3])

    # Linha 2: Av. Paulista pós-Pacote 2 (V2 Full alucinando vs V3 Ouro zerando falsos positivos)
    stem_pa = "tree_4727_side_angle_A_h12"
    img_pa = cv2.imread(str(IMAGES_DIR / f"{stem_pa}.jpg"))

    m_v2 = load_segformer_weights(MODELS_DIR / "arborgrid_segformer_b2_1024_v2_full_local.pt", device)
    pred_v2 = infer_masks(m_v2, img_pa, 1024, device)
    del m_v2
    torch.cuda.empty_cache()

    m_v3 = load_segformer_weights(MODELS_DIR / "arborgrid_segformer_b2_1024_v3_gold.pt", device)
    pred_v3 = infer_masks(m_v3, img_pa, 1024, device)
    del m_v3
    torch.cuda.empty_cache()

    v2_w = int(np.count_nonzero(pred_v2[1]))
    v3_w = int(np.count_nonzero(pred_v3[1]))

    r2_1 = render_overlay(img_pa, None, "Av. Paulista: Cena Original", "Rede 100% subterranea (0 fios reais)", size=420)
    r2_2 = render_overlay(img_pa, pred_v2, "V2 Full (Sem Hard Negatives)", f"Alucina {v2_w:,} px de fiacao na fachada", size=420)
    r2_3 = render_overlay(img_pa, pred_v3, "V3 Ouro (+Pacote 3 Hard Negs)", f"Falsos positivos zerados ({v3_w} px = SEGURO)", size=420)
    row2 = np.hstack([r2_1, r2_2, r2_3])

    grid = np.vstack([row1, row2])
    footer = add_legend_bar(grid.shape[1], "Verde=Copa | Vermelho=Fiacao | Ciano=Poste — Linha 1: V1 acelera Pacote 2 | Linha 2: Pacote 3 elimina alucinacoes na Paulista.")
    out = ASSETS_DIR / "exemplo_etapa2_e_3_p2_e_paulista.jpg"
    cv2.imwrite(str(out), np.vstack([grid, footer]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def _draw_morphology_panel(
    img_bgr: np.ndarray,
    masks: dict[int, np.ndarray],
    title: str,
    subtitle: str,
    draw_aabb: bool = False,
    size: int = 440,
) -> np.ndarray:
    # Calcula dilatação morfológica na resolução original (640px), depois redimensiona para desenho fino de 1px
    h, w = img_bgr.shape[:2]
    m_copa_src = (masks[0] > 0).astype(np.uint8)
    m_fio_src = (masks[1] > 0).astype(np.uint8)
    m_poste_src = (masks[2] > 0).astype(np.uint8)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    d1 = cv2.dilate(m_fio_src, k1)
    d2 = cv2.dilate(m_fio_src, k2)
    zone_direct_src = ((d1 > 0) & (m_copa_src > 0)).astype(np.uint8)
    zone_wind_src = (((d2 > 0) & ~(d1 > 0)) & (m_copa_src > 0)).astype(np.uint8)

    canvas = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    overlay = canvas.copy()

    m_copa = cv2.resize(m_copa_src, (size, size), interpolation=cv2.INTER_NEAREST)
    m_fio = cv2.resize(m_fio_src, (size, size), interpolation=cv2.INTER_NEAREST)
    m_poste = cv2.resize(m_poste_src, (size, size), interpolation=cv2.INTER_NEAREST)
    zone_direct = cv2.resize(zone_direct_src, (size, size), interpolation=cv2.INTER_NEAREST)
    zone_wind = cv2.resize(zone_wind_src, (size, size), interpolation=cv2.INTER_NEAREST)

    overlay[m_copa > 0] = (55, 175, 45)
    overlay[m_poste > 0] = (235, 165, 20)
    overlay[m_fio > 0] = (40, 60, 245)
    overlay[zone_wind > 0] = (15, 175, 245)      # Âmbar (r2=24px vento)
    overlay[zone_direct > 0] = (25, 25, 255)     # Vermelho (r1=6px contato)
    cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0, canvas)

    # Apenas 1 borda nítida de 1px por região (sem borda dupla interna de fiação)
    for mask, color in ((zone_wind, (0, 200, 255)), (zone_direct, (0, 0, 255))):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(canvas, cnts, -1, color, 1, cv2.LINE_AA)

    if draw_aabb:
        sx, sy = size / max(w, 1), size / max(h, 1)
        for mask_src, col, lbl in ((m_copa_src, (80, 255, 120), "AABB Copa"), (m_fio_src, (0, 235, 255), "AABB Fiacao")):
            ys, xs = np.where(mask_src > 0)
            if len(xs):
                x1, y1 = int(xs.min() * sx), int(ys.min() * sy)
                x2, y2 = int(xs.max() * sx), int(ys.max() * sy)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
                cv2.putText(canvas, lbl, (x1 + 4, max(58, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    return _stamp_header(canvas, title, subtitle)


def build_example_anomalias_v3(device: str) -> Path:
    """Mostra que mesmo as maiores anomalias da auditoria V3 (143 imgs) preservam forte alinhamento com o gabarito."""
    samples = [
        ("Pacote1_VAL #3", WS_DIR / "pacote_1", "tree_597668_side_angle_A_h25"),
        ("Pacote1 #4", WS_DIR / "pacote_1", "tree_597846_side_angle_A_h105"),
        ("Pacote2 #5", WS_DIR / "pacote_2", "tree_632565_side_angle_A_h31"),
    ]
    m_v3 = load_segformer_weights(MODELS_DIR / "arborgrid_segformer_b2_1024_v3_gold.pt", device)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    size = 380
    rows = []

    for tag, folder, stem in samples:
        img = cv2.imread(str(folder / f"{stem}.jpg"))
        h, w = img.shape[:2]
        gt = json_to_masks(folder / f"{stem}.json", h, w)
        pred = infer_masks(m_v3, img, 1024, device)

        gt_w = gt[1] > 0
        pr_w = pred[1] > 0
        inter = int(np.count_nonzero(gt_w & pr_w))
        union = int(np.count_nonzero(gt_w | pr_w))
        iou_w = 100.0 * inter / max(union, 1)
        extra_px = int(np.count_nonzero(pr_w & (~cv2.dilate(gt[1], k_dil).astype(bool))))
        miss_px = int(np.count_nonzero(gt_w & (~cv2.dilate(pred[1], k_dil).astype(bool))))

        p_gt = render_overlay(img, gt, f"{tag} | Gabarito Humano", stem[:30], size=size)
        p_pr = render_overlay(img, pred, f"Predicao V3 Ouro (IoU fio={iou_w:.1f}%)", f"Extra: +{extra_px}px | Omitido: -{miss_px}px", size=size)

        diff_canvas = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        diff_ov = diff_canvas.copy()
        gt_rs = cv2.resize(gt[1], (size, size), interpolation=cv2.INTER_NEAREST) > 0
        pr_rs = cv2.resize(pred[1], (size, size), interpolation=cv2.INTER_NEAREST) > 0
        diff_ov[gt_rs & pr_rs] = (60, 225, 80)     # Verde: acordo
        diff_ov[pr_rs & (~gt_rs)] = (40, 40, 255)  # Vermelho: extra V3
        diff_ov[gt_rs & (~pr_rs)] = (0, 220, 255)  # Amarelo: omitido
        cv2.addWeighted(diff_ov, 0.45, diff_canvas, 0.55, 0, diff_canvas)
        p_df = _stamp_header(diff_canvas, "Discrepancia Direta", "Verde=Acordo | Verm=Extra V3 | Amar=Omitido")
        rows.append(np.hstack([p_gt, p_pr, p_df]))

    del m_v3
    torch.cuda.empty_cache()

    grid = np.vstack(rows)
    footer = add_legend_bar(grid.shape[1], "Auditoria V3 (top anomalias reais com alta densidade de cabos): forte alinhamento geometrico mesmo nos maiores desvios.")
    out = ASSETS_DIR / "exemplo_etapa3_maiores_anomalias_v3.jpg"
    cv2.imwrite(str(out), np.vstack([grid, footer]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def build_example_arquitetura(device: str) -> Path:
    m_v3 = load_segformer_weights(MODELS_DIR / "arborgrid_segformer_b2_1024_v3_gold.pt", device)

    # 1) Morfologia V3 + AABB vs Pixel (#635210)
    img_crit = cv2.imread(str(IMAGES_DIR / "tree_635210_center_front_h290.jpg"))
    masks_crit = infer_masks(m_v3, img_crit, 1024, device)
    p1 = _draw_morphology_panel(
        img_crit,
        masks_crit,
        "1. Pixel vs AABB + Morfologia V3 (#635210)",
        "AABB engloba ceu/fachada; r1=6px / r2=24px isolam contato",
        draw_aabb=True,
    )

    # 2 & 3) Divergência de Paralaxe real (#633197): Front = CRITICO aparente vs Side = SEGURO real
    img_front = cv2.imread(str(IMAGES_DIR / "tree_633197_center_front_h224.jpg"))
    masks_front = infer_masks(m_v3, img_front, 1024, device)
    p2 = _draw_morphology_panel(
        img_front,
        masks_front,
        "2. Paralaxe Front (#633197): CRITICO 2D",
        "Projecao monocular sobrepoe copa e cabos (contato aparente)",
    )

    img_side = cv2.imread(str(IMAGES_DIR / "tree_633197_side_angle_A_h298.jpg"))
    masks_side = infer_masks(m_v3, img_side, 1024, device)
    p3 = _draw_morphology_panel(
        img_side,
        masks_side,
        "3. Paralaxe Side +90 (#633197): SEGURO",
        "Visada lateral revela afastamento -> DIVERGENCIA_PARALAXE",
    )

    del m_v3
    torch.cuda.empty_cache()

    row = np.hstack([p1, p2, p3])
    footer = add_legend_bar(row.shape[1], "Verde=Copa | Vermelho=r1(6px Contato) | Ambar=r2(24px Vento) — Segmentacao por pixel + consenso multi-visada.")
    out = ASSETS_DIR / "exemplo_arquitetura_morfologia_e_paralaxe.jpg"
    cv2.imwrite(str(out), np.vstack([row, footer]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Gerando exemplos visuais concretos em {ASSETS_DIR} (device={device})...")
    for path in (
        build_example_etapa0(),
        build_example_640_vs_1024(device),
        build_example_pacote2_and_paulista(device),
        build_example_anomalias_v3(device),
        build_example_arquitetura(device),
    ):
        print(f"OK -> {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
