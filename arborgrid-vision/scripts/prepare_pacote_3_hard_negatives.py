"""
ArborGrid - Preparador do Pacote 3 (Hard Negative Mining da Av. Paulista)
Módulo: arborgrid-vision / scripts / prepare_pacote_3_hard_negatives.py

Empacota os 18 casos de falsos positivos da Avenida Paulista (#39, #43, #36,
#7204, #5692, #5695, #651875, #4728 e #4565) com pré-anotação poligonal via
SegFormer V2 Full para curadoria manual rápida no X-AnyLabeling.
"""

import json
import shutil
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
SP_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"
IMAGES_DIR = SP_DIR / "images"
WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"
PAC3_DIR = WS_DIR / "pacote_3_hard_negatives"
MODELS_DIR = CV_ROOT / "models"
V2_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_v2_full_local.pt"

SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_MAP = {0: "copa", 1: "fiacao", 2: "poste"}
THRESHOLDS = {0: 0.50, 1: 0.38, 2: 0.48}
MIN_AREAS = {0: 180, 1: 25, 2: 120}
EPS_FACTORS = {0: 0.005, 1: 0.003, 2: 0.004}

IDS_TARGET = ["39", "43", "36", "7204", "5692", "5695", "651875", "4728", "4565"]


def mask_to_polygons(binary_mask: np.ndarray, label: str, eps_factor: float, min_area: int) -> list:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(eps_factor * peri, 1.0)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
        shapes.append({
            "label": label,
            "score": None,
            "points": pts,
            "group_id": None,
            "description": "prelabel_v2_hard_negative",
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })
    return shapes


def main():
    PAC3_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("ARBORGRID — PREPARAÇÃO DO PACOTE 3 (HARD NEGATIVE MINING DA PAULISTA)")
    print(f"Dispositivo: {device}")
    print(f"Diretório de saída: {PAC3_DIR}")
    print("=" * 80)

    # 1. Carregar manifesto e filtrar árvores alvo
    manifest = json.loads((SP_DIR / "manifest.json").read_text(encoding="utf-8"))
    target_trees = [t for t in manifest["trees"] if str(t.get("geosampa_id")) in IDS_TARGET]

    files_to_process = []
    for t in target_trees:
        for v in t.get("views", []):
            fn = v.get("file_name")
            if fn and (IMAGES_DIR / fn).exists():
                files_to_process.append((fn, str(t["geosampa_id"]), v["perspective"]))

    print(f"Total de fotografias selecionadas: {len(files_to_process)}")

    # 2. Carregar modelo SegFormer V2 Full
    print("Carregando SegFormer-B2 1024px V2 Full...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(V2_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # 3. Processar cada imagem
    for idx, (fn, cid, persp) in enumerate(files_to_process, start=1):
        src_img = IMAGES_DIR / fn
        dst_img = PAC3_DIR / fn
        dst_json = PAC3_DIR / f"{src_img.stem}.json"

        # Copiar imagem JPG
        shutil.copy2(src_img, dst_img)

        # Ler e inferir
        orig_bgr = cv2.imread(str(src_img))
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

        all_shapes = []
        for class_id in (0, 1, 2):
            lbl = CLASS_MAP[class_id]
            bin_m = (probs[class_id] >= THRESHOLDS[class_id]).astype(np.uint8)
            shapes = mask_to_polygons(bin_m, lbl, EPS_FACTORS[class_id], MIN_AREAS[class_id])
            all_shapes.extend(shapes)

        xany_doc = {
            "version": "2.4.0",
            "flags": {},
            "checked": False,
            "shapes": all_shapes,
            "imagePath": fn,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
            "description": f"ArborGrid Hard Negative - Tree #{cid} ({persp})",
        }

        dst_json.write_text(json.dumps(xany_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        counts = {
            "copa": sum(1 for s in all_shapes if s["label"] == "copa"),
            "fiacao": sum(1 for s in all_shapes if s["label"] == "fiacao"),
            "poste": sum(1 for s in all_shapes if s["label"] == "poste"),
        }
        print(f"  [{idx:02d}/{len(files_to_process)}] #{cid} ({persp}) -> {counts['copa']} copas, {counts['fiacao']} fios, {counts['poste']} postes anotados.")

    print(f"\nPacote 3 preparado com sucesso em: {PAC3_DIR}")
    print("Agora basta abrir esta pasta no X-AnyLabeling e deletar os polígonos de 'fiacao' incorretos!")


if __name__ == "__main__":
    main()

