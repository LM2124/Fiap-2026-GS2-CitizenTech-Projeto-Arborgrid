"""
ArborGrid - Auditoria Completa e Mosaico de Maiores Anomalias do Dataset (Modelo C V3)
Módulo: arborgrid-vision / scripts / audit_anomalies_v3_dataset.py

Avalia as predições do SegFormer-B2 1024px Modelo C V3 contra TODOS os 143 arquivos JSON curados
(Pacote 1 Revisado: 68 imgs, Pacote 2 Curado: 57 imgs, Pacote 3 Hard Negatives: 18 imgs).
Ranqueia as maiores discrepâncias (erros residuais do modelo, omissões sutis de rotulagem ou
casos extremos de oclusão) e gera mosaicos visuais comparativos em 4 colunas.
"""

import json
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

CV_ROOT = Path(__file__).resolve().parent.parent
if str(CV_ROOT) not in sys.path:
    sys.path.insert(0, str(CV_ROOT))

from scripts.run_v3_inference_and_update_map import (
    load_v3_model, IMAGENET_MEAN, IMAGENET_STD, THRESHOLDS, COLORS_BGR
)

WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"
OUT_DIR = CV_ROOT / "runs" / "audit_anomalies_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_TREE_IDS = {"597616", "597652", "597668", "597842"}
CLASS_ID = {"copa": 0, "fiacao": 1, "fiação": 1, "fio": 1, "poste": 2, "árvore": 0, "arvore": 0}


def json_to_masks(json_path: Path, h: int, w: int) -> dict:
    masks = {c: np.zeros((h, w), dtype=np.uint8) for c in range(3)}
    if not json_path.exists():
        return masks
    d = json.loads(json_path.read_text(encoding="utf-8"))
    orig_w = float(d.get("imageWidth") or w)
    orig_h = float(d.get("imageHeight") or h)
    sx = w / max(orig_w, 1.0)
    sy = h / max(orig_h, 1.0)
    for sh in d.get("shapes") or []:
        lbl = str(sh.get("label") or "").strip().lower()
        cid = 0 if "copa" in lbl else 1 if ("fio" in lbl or "fiacao" in lbl or "fiação" in lbl) else 2 if "poste" in lbl else None
        if cid is None:
            continue
        pts = sh.get("points") or []
        if len(pts) >= 3:
            scaled = np.round(np.array([[p[0] * sx, p[1] * sy] for p in pts])).astype(np.int32)
            cv2.fillPoly(masks[cid], [scaled], 1)
    return masks


def render_overlay_panel(img_bgr: np.ndarray, masks: dict, title: str, banner_bgr=(15, 23, 42)) -> np.ndarray:
    canvas = img_bgr.copy()
    overlay = img_bgr.copy()
    for cid in (0, 2, 1):
        m = np.asarray(masks[cid], dtype=np.uint8)
        if np.any(m):
            overlay[m > 0] = COLORS_BGR[cid]
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    for cid in (0, 2, 1):
        m = np.asarray(masks[cid], dtype=np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(canvas, cnts, -1, COLORS_BGR[cid], 2 if cid != 0 else 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), banner_bgr, -1)
    cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def render_discrepancy_panel(img_bgr: np.ndarray, gt_masks: dict, pred_masks: dict, title: str) -> np.ndarray:
    """
    Foca na discrepância de fiação e postes:
    - Verde brilhante: acordo (TP)
    - Vermelho puro: Extra previsto pelo Modelo C sem estar no JSON (FP ou fio omitido)
    - Amarelo ciano/ouro: Presente no JSON mas não previsto pelo Modelo C (FN)
    """
    canvas = (img_bgr.astype(np.float32) * 0.55).astype(np.uint8)
    gt_w = gt_masks[1] > 0
    pr_w = pred_masks[1] > 0

    tp = gt_w & pr_w
    fp_model = pr_w & (~gt_w)
    fn_model = gt_w & (~pr_w)

    canvas[tp] = (60, 230, 80)        # Verde: acordo
    canvas[fp_model] = (40, 40, 255)  # Vermelho: Extra do modelo
    canvas[fn_model] = (0, 220, 255)  # Amarelo: Omitido pelo modelo

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (30, 27, 75), -1)
    cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print("AUDITORIA COMPLETA DE ANOMALIAS NO DATASET V3 (143 IMAGENS vs MODELO C V3)")
    print(f"Dispositivo: {device}")
    print("=" * 85)

    model = load_v3_model(device)

    packs = [
        ("Pacote1", WS_DIR / "pacote_colega_1_revised"),
        ("Pacote2", WS_DIR / "pacote_colega_2"),
        ("Pacote3_HardNeg", WS_DIR / "pacote_3_hard_negatives"),
    ]

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    records = []

    for pack_name, pack_dir in packs:
        jpgs = sorted(pack_dir.glob("*.jpg"))
        print(f"Auditando {pack_name}: {len(jpgs)} imagens...")
        for img_p in jpgs:
            stem = img_p.stem
            json_p = pack_dir / f"{stem}.json"
            if not json_p.exists():
                continue

            is_val = (pack_name == "Pacote1" and any(vid in stem for vid in VAL_TREE_IDS))
            split_tag = f"{pack_name}_VAL" if is_val else pack_name

            orig_bgr = cv2.imread(str(img_p))
            h, w = orig_bgr.shape[:2]
            img_1024 = cv2.resize(orig_bgr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_1024, cv2.COLOR_BGR2RGB)
            img_f = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device != "cpu")):
                    logits = model(pixel_values=x).logits
                probs = torch.sigmoid(
                    F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
                )[0].cpu().numpy()

            pred_masks = {cid: (probs[cid] >= THRESHOLDS[cid]).astype(np.uint8) for cid in range(3)}
            gt_masks = json_to_masks(json_p, h, w)

            # Métricas de Fiação
            pw, gw = pred_masks[1], gt_masks[1]
            inter_w = int(np.count_nonzero(pw & gw))
            union_w = int(np.count_nonzero(pw | gw))
            iou_w = (inter_w / union_w * 100.0) if union_w > 0 else 100.0

            gt_w_dil = cv2.dilate(gw, k_dilate)
            pred_w_dil = cv2.dilate(pw, k_dilate)
            extra_wire_px = int(np.count_nonzero(pw & (gt_w_dil == 0)))
            missed_wire_px = int(np.count_nonzero(gw & (pred_w_dil == 0)))

            # Métricas de Copa
            pc, gc = pred_masks[0], gt_masks[0]
            inter_c = int(np.count_nonzero(pc & gc))
            union_c = int(np.count_nonzero(pc | gc))
            iou_c = (inter_c / union_c * 100.0) if union_c > 0 else 100.0
            copa_diff_px = int(np.count_nonzero(pc ^ gc))

            # Métricas de Poste
            pp, gp = pred_masks[2], gt_masks[2]
            inter_p = int(np.count_nonzero(pp & gp))
            union_p = int(np.count_nonzero(pp | gp))
            iou_p = (inter_p / union_p * 100.0) if union_p > 0 else 100.0
            poste_diff_px = int(np.count_nonzero(pp ^ gp))

            # Score unificado de anomalia
            # Dá peso alto a fios fantasmas / omitidos distantes (>7px do rótulo) e discrepância de IoU
            score = (
                extra_wire_px * 1.6
                + missed_wire_px * 1.4
                + poste_diff_px * 0.4
                + (100.0 - iou_w) * 18.0
                + (100.0 - iou_c) * 12.0
            )

            records.append({
                "pack": split_tag,
                "stem": stem,
                "img_path": str(img_p),
                "orig_bgr": orig_bgr,
                "gt_masks": gt_masks,
                "pred_masks": pred_masks,
                "iou_wire": round(iou_w, 2),
                "iou_copa": round(iou_c, 2),
                "iou_poste": round(iou_p, 2),
                "extra_wire_px": extra_wire_px,
                "missed_wire_px": missed_wire_px,
                "copa_diff_px": copa_diff_px,
                "poste_diff_px": poste_diff_px,
                "score": round(score, 1),
            })

    records.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 85)
    print("TOP 12 MAIORES ANOMALIAS DO DATASET V3 COMPLETO (143 IMAGENS):")
    print("=" * 85)
    for rank, r in enumerate(records[:12], start=1):
        print(
            f"#{rank:02d} [{r['pack']}] {r['stem']} | Score: {r['score']:.0f} | "
            f"IoU Fio: {r['iou_wire']:.1f}% | Extra Fio: {r['extra_wire_px']} px | "
            f"Omitido Fio: {r['missed_wire_px']} px | IoU Copa: {r['iou_copa']:.1f}%"
        )

    # Salvar relatório JSON (sem arrays de imagem)
    report = [
        {k: v for k, v in r.items() if k not in ("orig_bgr", "gt_masks", "pred_masks")}
        for r in records
    ]
    report_path = OUT_DIR / "audit_anomalies_v3_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Montar Mosaico Top 8 Anomalias (4 colunas por linha: Orig | Gabarito | Modelo C | Discrepância)
    top_n = 8
    tile_sz = 420
    rows = []
    for rank, r in enumerate(records[:top_n], start=1):
        orig = r["orig_bgr"]
        gt = r["gt_masks"]
        pr = r["pred_masks"]

        p_orig = cv2.resize(orig, (tile_sz, tile_sz))
        cv2.rectangle(p_orig, (0, 0), (tile_sz, 28), (15, 23, 42), -1)
        cv2.putText(p_orig, f"#{rank} [{r['pack']}] {r['stem'][:20]}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 250, 252), 1, cv2.LINE_AA)

        p_gt = render_overlay_panel(cv2.resize(orig, (tile_sz, tile_sz)),
                                    {c: cv2.resize(gt[c], (tile_sz, tile_sz), interpolation=cv2.INTER_NEAREST) for c in range(3)},
                                    "Gabarito JSON (Humano)", banner_bgr=(30, 58, 138))

        p_pr = render_overlay_panel(cv2.resize(orig, (tile_sz, tile_sz)),
                                    {c: cv2.resize(pr[c], (tile_sz, tile_sz), interpolation=cv2.INTER_NEAREST) for c in range(3)},
                                    f"Pred Modelo C | Fio IoU: {r['iou_wire']:.1f}%", banner_bgr=(20, 83, 45))

        p_disc = render_discrepancy_panel(cv2.resize(orig, (tile_sz, tile_sz)),
                                          {c: cv2.resize(gt[c], (tile_sz, tile_sz), interpolation=cv2.INTER_NEAREST) for c in range(3)},
                                          {c: cv2.resize(pr[c], (tile_sz, tile_sz), interpolation=cv2.INTER_NEAREST) for c in range(3)},
                                          f"Discrep: +{r['extra_wire_px']}px / -{r['missed_wire_px']}px")

        rows.append(np.hstack([p_orig, p_gt, p_pr, p_disc]))

    mosaic = np.vstack(rows)
    out_mosaic = OUT_DIR / "mosaico_maiores_anomalias_v3.jpg"
    cv2.imwrite(str(out_mosaic), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"\nRelatório salvo em: {report_path}")
    print(f"Mosaico salvo em:   {out_mosaic}")


if __name__ == "__main__":
    main()
