"""
ArborGrid - Scanner de Anomalias e Discrepâncias no Pacote 1 Revisado
Módulo: arborgrid-vision / scripts / audit_anomalies_pacote_1_gold.py

Compara as predições do SegFormer-B2 1024px Ouro contra os rótulos 100% revisados
do pacote_1 (68 imagens). Identifica e ranqueia as maiores discrepâncias:
1. Fios de alta confiança previstos pela IA e não anotados pelo usuário.
2. Fios anotados pelo usuário que a IA não recuperou (baixa recall / atípicos).
3. Divergências na infraestrutura de postes.
Gera relatório estruturado e mosaico visual das maiores anomalias.
"""

import json
import re
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = CV_ROOT / "models"
WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"
PACK1_REVISED = WS_DIR / "pacote_1"
OUT_DIR = CV_ROOT / "runs" / "audit_anomalies_pack1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_gold_local.pt"
if not GOLD_WEIGHTS.exists():
    GOLD_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_gold.pt"

SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_ID = {"copa": 0, "fiacao": 1, "poste": 2, "fiação": 1, "árvore": 0, "arvore": 0}
COLORS_BGR = {0: (60, 195, 45), 1: (35, 45, 245), 2: (235, 165, 20)}


def remap_colab_to_local(k: str) -> str:
    k = k.replace("decode_head.linear_projections.", "decode_head.linear_c.")
    m = re.match(r"segformer\.stages\.(\d+)\.patch_embeddings\.(.*)", k)
    if m:
        return f"segformer.encoder.patch_embeddings.{m.group(1)}.{m.group(2)}"
    m = re.match(r"segformer\.stages\.(\d+)\.layer_norm\.(.*)", k)
    if m:
        return f"segformer.encoder.layer_norm.{m.group(1)}.{m.group(2)}"
    m = re.match(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.(.*)", k)
    if m:
        stage, block, rest = m.group(1), m.group(2), m.group(3)
        rest = rest.replace("layernorm_before.", "layer_norm_1.")
        rest = rest.replace("layernorm_after.", "layer_norm_2.")
        rest = rest.replace("attention.q_proj.", "attention.self.query.")
        rest = rest.replace("attention.k_proj.", "attention.self.key.")
        rest = rest.replace("attention.v_proj.", "attention.self.value.")
        rest = rest.replace("attention.o_proj.", "attention.output.dense.")
        rest = rest.replace("attention.sequence_reduction.sequence_reduction.", "attention.self.sr.")
        rest = rest.replace("attention.sequence_reduction.layer_norm.", "attention.self.layer_norm.")
        rest = rest.replace("mlp.fc1.", "mlp.dense1.")
        rest = rest.replace("mlp.fc2.", "mlp.dense2.")
        return f"segformer.encoder.block.{stage}.{block}.{rest}"
    return k


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
        if np.any(masks[cid]):
            overlay[masks[cid] > 0] = COLORS_BGR[cid]
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    for cid in (0, 2, 1):
        cnts, _ = cv2.findContours(masks[cid], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(canvas, cnts, -1, COLORS_BGR[cid], 2 if cid != 0 else 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (15, 23, 42), -1)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("AUDITORIA DE ANOMALIAS NO PACOTE 1 REVISADO (SEGFORMER 1024PX OURO)")
    print(f"Dispositivo: {device}")
    print(f"Diretório:   {PACK1_REVISED}")
    print("=" * 80)

    # 1. Carregar modelo
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(GOLD_WEIGHTS, map_location="cpu", weights_only=True)
    if any(k.startswith("segformer.stages.") for k in sd.keys()):
        sd = {remap_colab_to_local(k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # 2. Iterar sobre as 68 fotos
    jpg_paths = sorted(PACK1_REVISED.glob("*.jpg"))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_dilate_tight = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    anomalies = []

    for idx, img_p in enumerate(jpg_paths, start=1):
        stem = img_p.stem
        orig_bgr = cv2.imread(str(img_p))
        h, w = orig_bgr.shape[:2]

        # Inferência em 1024x1024
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
            1: (probs[1] >= 0.40).astype(np.uint8),
            2: (probs[2] >= 0.48).astype(np.uint8),
        }

        gt_masks = json_to_masks(PACK1_REVISED / f"{stem}.json", h, w)

        # Diagnóstico de discrepâncias na fiação
        gt_wire = gt_masks[1]
        pr_wire = pred_masks[1]
        gt_wire_dil = cv2.dilate(gt_wire, k_dilate)
        pr_wire_dil = cv2.dilate(pr_wire, k_dilate_tight)

        # Fios previstos com alta confiança que estão LONGE dos polígonos do usuário
        unlabeled_pred = (gt_wire_dil == 0) & (probs[1] >= 0.60)
        unlabeled_px = int(np.count_nonzero(unlabeled_pred))

        # Fios rotulados pelo usuário que a IA não detectou
        gt_wire_px = int(np.count_nonzero(gt_wire))
        recovered_gt_px = int(np.count_nonzero((gt_wire > 0) & (pr_wire_dil > 0)))
        recall_pct = (recovered_gt_px / max(gt_wire_px, 1)) * 100.0

        # Divergência de postes
        gt_pole_px = int(np.count_nonzero(gt_masks[2]))
        pr_pole_px = int(np.count_nonzero(pred_masks[2]))
        pole_diff_px = abs(gt_pole_px - pr_pole_px)

        reasons = []
        score = 0.0

        # Critério 1: Condutores prováveis não anotados
        if unlabeled_px >= 300:
            reasons.append(f"Fio previsto não rotulado ({unlabeled_px} px com conf >= 0.60 longe dos polígonos)")
            score += unlabeled_px / 120.0

        # Critério 2: Fio anotado com baixo recall na IA
        if gt_wire_px >= 800 and recall_pct < 35.0:
            reasons.append(f"Fio anotado com baixo recall na IA ({recall_pct:.1f}% recuperado de {gt_wire_px} px)")
            score += (100.0 - recall_pct) * 0.15

        # Critério 3: Grande divergência de poste
        if (gt_pole_px == 0 and pr_pole_px >= 1200) or (gt_pole_px >= 1500 and pr_pole_px == 0):
            reasons.append(f"Divergência substancial de poste (GT={gt_pole_px} px vs IA={pr_pole_px} px)")
            score += pole_diff_px / 400.0

        if score > 0.0:
            # Salva imagem do caso
            p1 = orig_bgr
            p2 = render_panel(orig_bgr, gt_masks, f"SUA REVISAO: {stem[:22]}")
            p3 = render_panel(orig_bgr, pred_masks, "PREDICAO IA OURO 1024PX")
            trio = np.hstack([p1, p2, p3])
            trio_path = OUT_DIR / f"anom_{stem}.jpg"
            cv2.imwrite(str(trio_path), trio, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            anomalies.append({
                "stem": stem,
                "score": round(score, 2),
                "reasons": reasons,
                "unlabeled_wire_px": unlabeled_px,
                "gt_wire_px": gt_wire_px,
                "wire_recall_pct": round(recall_pct, 1),
                "gt_pole_px": gt_pole_px,
                "pr_pole_px": pr_pole_px,
                "preview": trio_path.as_posix(),
            })

    anomalies.sort(key=lambda a: a["score"], reverse=True)

    print(f"\nTotal de imagens com discrepâncias identificadas: {len(anomalies)} de {len(jpg_paths)}")
    print("=" * 80)
    print("TOP 10 MAIORES DISCREPÂNCIAS NO PACOTE 1:")
    print("=" * 80)
    for i, a in enumerate(anomalies[:10], start=1):
        print(f"{i:2d}. {a['stem']} (Score: {a['score']:.1f})")
        for r in a["reasons"]:
            print(f"    - {r}")

    # Salva JSON consolidado
    (OUT_DIR / "anomalies_pack1_gold.json").write_text(
        json.dumps(anomalies, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Gera mosaico das Top 6 discrepâncias
    if anomalies:
        top_panels = []
        for a in anomalies[:6]:
            tpath = Path(a["preview"])
            if tpath.exists():
                timg = cv2.imread(str(tpath))
                # Redimensiona mantendo proporção para caber em mosaico elegante (largura 960)
                th, tw = timg.shape[:2]
                scale = 960.0 / tw
                top_panels.append(cv2.resize(timg, (960, int(th * scale))))
        if top_panels:
            mosaic = np.vstack(top_panels)
            cv2.imwrite(str(OUT_DIR / "mosaico_top_anomalias_pacote1.jpg"), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(f"\nMosaico das top anomalias salvo em: {OUT_DIR / 'mosaico_top_anomalias_pacote1.jpg'}")


if __name__ == "__main__":
    main()

