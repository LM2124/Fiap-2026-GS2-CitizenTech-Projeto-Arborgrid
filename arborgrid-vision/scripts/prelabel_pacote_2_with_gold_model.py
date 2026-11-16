"""
ArborGrid - Pré-rotulador Automatizado de Alta Resolução (SegFormer-B2 1024px Ouro)
Aplica o modelo treinado em 1024x1024 sobre o Pacote 2 do X-AnyLabeling.
"""

import json
import shutil
import time
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
PACK2_DIR = WS_DIR / "pacote_colega_2"
BACKUP_DIR = WS_DIR / "pacote_colega_2_old_prelabels_backup"

GOLD_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_gold_local.pt"
if not GOLD_WEIGHTS.exists():
    GOLD_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_gold.pt"

SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_MAP = {0: "copa", 1: "fiacao", 2: "poste"}
COLORS_BGR = {0: (60, 195, 45), 1: (35, 45, 245), 2: (235, 165, 20)}

THRESHOLDS = {
    0: 0.50,  # copa
    1: 0.38,  # fiacao (calibrado para alta sensibilidade de cabos finos)
    2: 0.48,  # poste
}

MIN_AREAS = {
    0: 180,  # copa
    1: 25,   # fiacao
    2: 120,  # poste
}

EPS_FACTORS = {
    0: 0.005,  # copa (simplificação suave)
    1: 0.003,  # fiacao (preserva retas de cabos)
    2: 0.004,  # poste (preserva verticalidade)
}


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


def load_gold_model(device: str):
    print(f"Carregando pesos ouro do modelo: {GOLD_WEIGHTS.name}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(GOLD_WEIGHTS, map_location="cpu", weights_only=True)
    if any(k.startswith("segformer.stages.") for k in sd.keys()):
        sd = {remap_colab_to_local(k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def mask_to_polygons(binary_mask: np.ndarray, cid: int, min_area: int, eps_factor: float) -> list[list[list[float]]]:
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps_factor * peri, True)
        if len(approx) >= 3:
            pts = [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in approx]
            polys.append(pts)
    return polys


def render_preview(img_bgr: np.ndarray, shapes: list[dict], title: str) -> np.ndarray:
    canvas = img_bgr.copy()
    overlay = img_bgr.copy()
    c_inv = {"copa": 0, "fiacao": 1, "poste": 2}

    # Preenchimento transparente
    for sh in shapes:
        cid = c_inv.get(sh["label"], 0)
        pts = np.array(sh["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], COLORS_BGR[cid])

    cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)

    # Bordas sólidas
    for sh in shapes:
        cid = c_inv.get(sh["label"], 0)
        pts = np.array(sh["points"], dtype=np.int32)
        thick = 2 if cid != 0 else 1
        cv2.polylines(canvas, [pts], True, COLORS_BGR[cid], thick, cv2.LINE_AA)

    # Faixa superior de título
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (15, 23, 42), -1)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("PRÉ-ROTULAGEM AUTOMATIZADA EM ALTA RESOLUÇÃO — PACOTE COLEGA 2")
    print(f"Dispositivo de Execução: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Diretório de Destino:   {PACK2_DIR}")
    print("=" * 80)

    # 1. Backup de segurança
    if not BACKUP_DIR.exists():
        print(f"Criando backup do estado inicial de {PACK2_DIR.name} em {BACKUP_DIR.name}...")
        shutil.copytree(PACK2_DIR, BACKUP_DIR)
        print("Backup concluído com sucesso.")
    else:
        print(f"Backup existente encontrado em {BACKUP_DIR.name}. Prosseguindo.")

    # 2. Carregar modelo
    model = load_gold_model(device)

    # 3. Listar imagens
    jpg_paths = sorted(PACK2_DIR.glob("*.jpg"))
    print(f"Total de imagens a processar: {len(jpg_paths)}")

    stats = {"copa": 0, "fiacao": 0, "poste": 0}
    t_start = time.time()
    preview_panels = []

    for idx, img_p in enumerate(jpg_paths, start=1):
        stem = img_p.stem
        orig_bgr = cv2.imread(str(img_p))
        orig_h, orig_w = orig_bgr.shape[:2]

        # Entrada em 1024x1024
        img_1024 = cv2.resize(orig_bgr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_1024, cv2.COLOR_BGR2RGB)
        img_f = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device != "cpu")):
                logits = model(pixel_values=x).logits
            probs = torch.sigmoid(
                F.interpolate(logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            )[0].cpu().numpy()

        shapes = []
        for cid in [0, 2, 1]:  # Ordem: copa primeiro, poste, fiação no topo
            label = CLASS_MAP[cid]
            bmask = (probs[cid] >= THRESHOLDS[cid]).astype(np.uint8)
            polys = mask_to_polygons(bmask, cid, MIN_AREAS[cid], EPS_FACTORS[cid])
            for pts in polys:
                shapes.append(
                    {
                        "label": label,
                        "score": round(float(np.mean(probs[cid][bmask > 0])), 3) if np.any(bmask) else None,
                        "points": pts,
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "polygon",
                        "flags": {},
                        "attributes": {},
                    }
                )
                stats[label] += 1

        # Salva o arquivo JSON
        json_data = {
            "version": "2.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_p.name,
            "imageData": None,
            "imageHeight": orig_h,
            "imageWidth": orig_w,
        }
        json_path = PACK2_DIR / f"{stem}.json"
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Seleciona amostras para o mosaico de auditoria visual (as primeiras 8 imagens)
        if idx <= 8:
            title = f"{stem[:25]} | {len([s for s in shapes if s['label']=='fiacao'])} fios | {len([s for s in shapes if s['label']=='poste'])} post"
            preview_panels.append(render_preview(orig_bgr, shapes, title))

        if idx % 10 == 0 or idx == len(jpg_paths):
            elapsed = time.time() - t_start
            pace = elapsed / idx
            print(f"[{idx:02d}/{len(jpg_paths)}] Processado em {pace:.2f}s/img | Total acumulado: {stats['fiacao']} fios, {stats['poste']} postes, {stats['copa']} copas")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("CONCLUÍDO COM SUCESSO!")
    print(f"Tempo total de inferência: {total_time:.1f} segundos ({total_time / len(jpg_paths):.2f}s por imagem)")
    print(f"Estatísticas dos novos pré-labels gerados em {PACK2_DIR.name}:")
    print(f"  - Copas anotadas:   {stats['copa']}")
    print(f"  - Fiações anotadas: {stats['fiacao']} (média de {stats['fiacao']/len(jpg_paths):.1f} feixes por imagem)")
    print(f"  - Postes anotados:  {stats['poste']} (média de {stats['poste']/len(jpg_paths):.1f} postes por imagem)")
    print("=" * 80)

    # 4. Gerar Mosaico de Auditoria Visual
    if preview_panels:
        row1 = np.hstack(preview_panels[:4])
        row2 = np.hstack(preview_panels[4:8])
        mosaic = np.vstack([row1, row2])
        out_mosaic_path = WS_DIR / "mosaico_prelabel_pacote_2_ouro.jpg"
        cv2.imwrite(str(out_mosaic_path), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"Mosaico visual de verificação salvo em: {out_mosaic_path}")


if __name__ == "__main__":
    main()

