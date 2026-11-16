"""
ArborGrid - Auditoria e Mosaico de Anomalias no Pacote 2 (Modelo V2 Full)
Módulo: arborgrid-vision / scripts / audit_anomalies_pacote_2_v2.py

Avalia as predições do SegFormer-B2 1024px V2 Full contra os rótulos do pacote_colega_2 (64 imagens).
Ranqueia e gera mosaico visual com as maiores anomalias e discrepâncias (especialmente entre
as imagens ainda não refinadas manualmente pelo usuário).
"""

import json
import re
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = CV_ROOT / "models"
WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"
PACK2_DIR = WS_DIR / "pacote_colega_2"
RUNS_DIR = CV_ROOT / "runs" / "audit_anomalies_pack2"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

V2_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_v2_full_local.pt"
SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_ID = {"copa": 0, "fiacao": 1, "poste": 2, "fiação": 1, "árvore": 0, "arvore": 0}
COLORS_BGR = {0: (60, 195, 45), 1: (35, 45, 245), 2: (235, 165, 20)}


def json_to_masks(json_path: Path, h: int = 640, w: int = 640) -> dict:
    masks = {c: np.zeros((h, w), dtype=np.uint8) for c in range(3)}
    if not json_path.exists():
        return masks
    d = json.loads(json_path.read_text(encoding="utf-8"))
    for sh in d.get("shapes") or []:
        lbl = str(sh.get("label") or "").strip().lower()
        if lbl not in CLASS_ID:
            continue
        cid = CLASS_ID[lbl]
        pts = sh.get("points") or []
        if len(pts) >= 3:
            cv2.fillPoly(masks[cid], [np.round(np.array(pts)).astype(np.int32)], 1)
    return masks


def render_panel(img_bgr: np.ndarray, masks: dict, title: str) -> np.ndarray:
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
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (15, 23, 42), -1)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("AUDITORIA DE ANOMALIAS NO PACOTE 2 (MODELO V2 FULL 1024PX)")
    print(f"Dispositivo: {device}")
    print(f"Diretório:   {PACK2_DIR}")
    print("=" * 80)

    # 1. Carregar modelo
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(V2_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # 2. Carregar imagens do pacote 2
    jpg_paths = sorted(PACK2_DIR.glob("*.jpg"))
    print(f"Total de imagens encontradas: {len(jpg_paths)}")

    user_session_cutoff = datetime(2026, 11, 12, 1, 0)
    anomalies = []

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for idx, img_p in enumerate(jpg_paths, start=1):
        stem = img_p.stem
        json_p = PACK2_DIR / f"{stem}.json"
        is_user_refined = False
        if json_p.exists():
            mtime = datetime.fromtimestamp(json_p.stat().st_mtime)
            is_user_refined = (mtime >= user_session_cutoff)

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

        pred_masks = {
            0: (probs[0] >= 0.50).astype(np.uint8),
            1: (probs[1] >= 0.38).astype(np.uint8),
            2: (probs[2] >= 0.48).astype(np.uint8),
        }

        gt_masks = json_to_masks(json_p, h, w)

        # Fiação métricas
        pred_wire = pred_masks[1]
        gt_wire = gt_masks[1]
        inter_wire = int(np.count_nonzero(pred_wire & gt_wire))
        union_wire = int(np.count_nonzero(pred_wire | gt_wire))
        iou_wire = (inter_wire / union_wire * 100) if union_wire > 0 else 100.0

        # Fios que o modelo previu mas o JSON não tem
        gt_dilated = cv2.dilate(gt_wire, k_dilate)
        extra_wire_px = int(np.count_nonzero(pred_wire & (gt_dilated == 0)))

        # Fios que o JSON tem mas o modelo não previu
        pred_dilated = cv2.dilate(pred_wire, k_dilate)
        missed_wire_px = int(np.count_nonzero(gt_wire & (pred_dilated == 0)))

        # Copa métricas
        pred_copa = pred_masks[0]
        gt_copa = gt_masks[0]
        inter_copa = int(np.count_nonzero(pred_copa & gt_copa))
        union_copa = int(np.count_nonzero(pred_copa | gt_copa))
        iou_copa = (inter_copa / union_copa * 100) if union_copa > 0 else 100.0

        # Anomaly score: combina discrepância de fiação e copa
        # Imagens não refinadas têm peso natural nas discrepâncias
        score = extra_wire_px * 1.5 + missed_wire_px * 1.2 + (100.0 - iou_wire) * 20.0 + (100.0 - iou_copa) * 5.0

        anomalies.append({
            "stem": stem,
            "img_path": img_p,
            "orig_bgr": orig_bgr,
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "is_user_refined": is_user_refined,
            "iou_wire": iou_wire,
            "iou_copa": iou_copa,
            "extra_wire_px": extra_wire_px,
            "missed_wire_px": missed_wire_px,
            "score": score,
        })

    # Ordenar por score de anomalia decrescente
    anomalies.sort(key=lambda a: a["score"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 MAIORES ANOMALIAS IDENTIFICADAS NO PACOTE 2:")
    print("=" * 80)
    for rank, an in enumerate(anomalies[:10], start=1):
        status_ref = "REVISADA_PELO_USUARIO" if an["is_user_refined"] else "PRE_LABEL_NAO_REFINADO"
        print(f"#{rank:02d} [{status_ref}] {an['stem']}")
        print(f"     IoU Fiação: {an['iou_wire']:.1f}% | Extra Fio: {an['extra_wire_px']} px | Fio Omitido: {an['missed_wire_px']} px | IoU Copa: {an['iou_copa']:.1f}% | Score: {an['score']:.0f}")

    # Montar Mosaico Visual das Top 6 Anomalias
    vis_rows = []
    for an in anomalies[:6]:
        status_ref = "Revisada" if an["is_user_refined"] else "PreLabel Nao-Refinado"
        title_orig = f"Original ({status_ref}): {an['stem'][:22]}"
        title_gt = f"JSON Atual: IoU Fio={an['iou_wire']:.1f}%"
        title_pred = f"V2 Full: +{an['extra_wire_px']}px extra | -{an['missed_wire_px']}px omissao"

        p_orig = render_panel(an["orig_bgr"], {0: np.zeros_like(an["orig_bgr"][:,:,0]), 1: np.zeros_like(an["orig_bgr"][:,:,0]), 2: np.zeros_like(an["orig_bgr"][:,:,0])}, title_orig)
        p_gt = render_panel(an["orig_bgr"], an["gt_masks"], title_gt)
        p_pred = render_panel(an["orig_bgr"], an["pred_masks"], title_pred)

        strip = np.hstack([cv2.resize(p, (480, 480)) for p in [p_orig, p_gt, p_pred]])
        vis_rows.append(strip)

    mosaic = np.vstack(vis_rows)
    mosaic_out = RUNS_DIR / "mosaico_top_anomalias_pacote2.jpg"
    cv2.imwrite(str(mosaic_out), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"\nMosaico de anomalias salvo em: {mosaic_out}")

    # Salva relatório JSON
    report = [
        {
            "rank": r,
            "stem": an["stem"],
            "is_user_refined": an["is_user_refined"],
            "iou_wire": round(an["iou_wire"], 2),
            "iou_copa": round(an["iou_copa"], 2),
            "extra_wire_px": an["extra_wire_px"],
            "missed_wire_px": an["missed_wire_px"],
            "score": round(an["score"], 1),
        }
        for r, an in enumerate(anomalies[:15], start=1)
    ]
    (RUNS_DIR / "relatorio_top_anomalias_pacote2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Relatório JSON salvo com sucesso!")


if __name__ == "__main__":
    main()

