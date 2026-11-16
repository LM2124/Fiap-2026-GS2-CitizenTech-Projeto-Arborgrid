"""
ArborGrid - Inferência do Modelo Ouro V3 (Modelo C — Hard Negative Mining) 1024px e Atualização do Mapa
Módulo: arborgrid-vision / scripts / run_v3_inference_and_update_map.py

Executa inferência do SegFormer-B2 1024px Modelo C V3 (treinado em 135 imagens com Pacotes 1+2+3 Hard Negatives
e augmentations calibradas) sobre TODAS as 271 árvores georreferenciadas de São Paulo (Pinheiros, Alto da Mooca,
Vila Mariana e Avenida Paulista).
"""

import json
import re
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
SP_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"
IMAGES_DIR = SP_DIR / "images"
V3_OVERLAYS_DIR = SP_DIR / "v3_overlays"
V3_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = SP_DIR / "manifest.json"

MODELS_DIR = CV_ROOT / "models"
V3_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_v3_gold.pt"

SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

THRESHOLDS = {0: 0.50, 1: 0.40, 2: 0.50}

COLORS_BGR = {
    0: (60, 195, 45),    # copa: verde
    1: (35, 45, 245),    # fiacao: vermelho
    2: (235, 165, 20),   # poste: ciano
}

STATUS_COLORS_HEX = {
    "CRITICO": "#ef4444",
    "PROXIMIDADE": "#f59e0b",
    "SEGURO": "#10b981",
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


def load_v3_model(device: str):
    if not V3_WEIGHTS.exists():
        raise FileNotFoundError(f"Pesos do Modelo C V3 não encontrados em: {V3_WEIGHTS}")
    print(f"Carregando pesos do SegFormer V3 Ouro (Modelo C): {V3_WEIGHTS.name}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(V3_WEIGHTS, map_location="cpu", weights_only=True)
    if any(k.startswith("segformer.stages.") for k in sd.keys()):
        sd = {remap_colab_to_local(k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def analyze_conflict_morphology_calibrated(mask_copa: np.ndarray, mask_wire: np.ndarray):
    """
    Motor morfológico calibrado com filtros determinísticos anti-ruído:
    1. Descarte de fragmentos de fiação < 80 px.
    2. Foco na massa arbórea principal (descarte < 15% da copa dominante ou < 800 px).
    3. Dilatação elíptica: contato direto (r1=6px) e oscilação de vento (r2=24px).
    """
    num_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(mask_wire, connectivity=8)
    clean_wire = np.zeros_like(mask_wire)
    for i in range(1, num_w):
        if stats_w[i, cv2.CC_STAT_AREA] >= 80:
            clean_wire[labels_w == i] = 1

    num_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(mask_copa, connectivity=8)
    clean_copa = np.zeros_like(mask_copa)
    if num_c > 1:
        areas = stats_c[1:, cv2.CC_STAT_AREA]
        max_area = np.max(areas)
        for i in range(1, num_c):
            if stats_c[i, cv2.CC_STAT_AREA] >= max(800, max_area * 0.15):
                clean_copa[labels_c == i] = 1
    else:
        clean_copa = mask_copa

    k_direct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k_wind = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))

    wire_direct = cv2.dilate(clean_wire, k_direct)
    wire_wind = cv2.dilate(clean_wire, k_wind)

    inter_direct = (wire_direct > 0) & (clean_copa > 0)
    inter_wind = (wire_wind > 0) & (clean_copa > 0) & (~inter_direct)

    px_direct = int(np.count_nonzero(inter_direct))
    px_wind = int(np.count_nonzero(inter_wind))
    wire_px = int(np.count_nonzero(clean_wire))
    copa_px = int(np.count_nonzero(clean_copa))

    if px_direct >= 3000 and wire_px >= 800:
        triage = "CRITICO"
    elif (px_direct >= 250 or px_wind >= 1200) and wire_px >= 200:
        triage = "PROXIMIDADE"
    else:
        triage = "SEGURO"

    return triage, px_direct, px_wind, wire_px, copa_px, inter_direct, inter_wind, clean_wire, clean_copa


def render_v3_overlay(orig_bgr: np.ndarray, masks: dict, inter_direct: np.ndarray, triage: str, title: str) -> np.ndarray:
    canvas = orig_bgr.copy()
    overlay = orig_bgr.copy()

    for cid in (0, 2, 1):
        m = np.asarray(masks[cid], dtype=np.uint8)
        if np.any(m):
            overlay[m > 0] = COLORS_BGR[cid]

    if np.any(inter_direct):
        overlay[inter_direct] = (255, 0, 220)

    cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)

    for cid in (0, 2, 1):
        m = np.asarray(masks[cid], dtype=np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(canvas, cnts, -1, COLORS_BGR[cid], 2 if cid != 0 else 1, cv2.LINE_AA)

    if np.any(inter_direct):
        cnts_i, _ = cv2.findContours(inter_direct.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_i:
            cv2.drawContours(canvas, cnts_i, -1, (255, 255, 255), 2, cv2.LINE_AA)

    banner_color = {"CRITICO": (30, 30, 220), "PROXIMIDADE": (20, 140, 240), "SEGURO": (40, 160, 40)}.get(triage, (15, 23, 42))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), banner_color, -1)
    cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print("ARBORGRID — INFERÊNCIA MODELO C (V3 OURO) 1024PX EM TODAS AS 271 ÁRVORES DE SP")
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    trees = manifest.get("trees", [])
    model = load_v3_model(device)

    print(f"Total de árvores no manifesto: {len(trees)}")

    t0_global = time.time()
    processed_views = 0
    region_stats = {}

    for t_idx, tree in enumerate(trees, start=1):
        reg = tree.get("region_key", "pinheiros")
        if reg not in region_stats:
            region_stats[reg] = {
                "trees": 0,
                "v2_crit": 0, "v2_paralax": 0, "v2_prox": 0, "v2_safe": 0,
                "v3_crit": 0, "v3_paralax": 0, "v3_prox": 0, "v3_safe": 0,
                "v2_wire_px_sum": 0, "v3_wire_px_sum": 0
            }
        region_stats[reg]["trees"] += 1

        views_results = []
        for v in tree.get("views", []):
            img_file = v.get("file_name")
            img_path = IMAGES_DIR / img_file
            if not img_path.exists():
                continue

            orig_bgr = cv2.imread(str(img_path))
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

            raw_masks = {
                0: (probs[0] >= THRESHOLDS[0]).astype(np.uint8),
                1: (probs[1] >= THRESHOLDS[1]).astype(np.uint8),
                2: (probs[2] >= THRESHOLDS[2]).astype(np.uint8),
            }

            triage, px_direct, px_wind, wire_px, copa_px, inter_direct, inter_wind, clean_wire, clean_copa = analyze_conflict_morphology_calibrated(
                raw_masks[0], raw_masks[1]
            )

            disp_masks = {0: clean_copa, 1: clean_wire, 2: raw_masks[2]}
            ov_name = f"ov_v3_{img_file}"
            ov_path = V3_OVERLAYS_DIR / ov_name
            title = f"{triage} (V3-C) | Copa: {copa_px} px | Fio: {wire_px} px | Contato: {px_direct} px"
            canvas = render_v3_overlay(orig_bgr, disp_masks, inter_direct, triage, title)
            cv2.imwrite(str(ov_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            v["v3_analysis"] = {
                "triage": triage,
                "color_hex": STATUS_COLORS_HEX[triage],
                "copa_px": copa_px,
                "wire_px": wire_px,
                "pole_px": int(np.count_nonzero(raw_masks[2])),
                "contact_direct_px": px_direct,
                "wind_buffer_px": px_wind,
                "v3_overlay_file": ov_name,
            }
            region_stats[reg]["v3_wire_px_sum"] += wire_px
            region_stats[reg]["v2_wire_px_sum"] += int(v.get("v2_analysis", {}).get("wire_px", 0))
            views_results.append((v.get("perspective"), triage, STATUS_COLORS_HEX[triage]))
            processed_views += 1

        # Consolidação da Árvore (V3)
        views_dict = {r[0]: (r[1], r[2]) for r in views_results}
        has_side = ("side_angle_A" in views_dict) or ("side_angle_B" in views_dict)
        st_front = views_dict.get("center_front", ("SEGURO", STATUS_COLORS_HEX["SEGURO"]))[0]
        col_front = STATUS_COLORS_HEX[st_front]

        if has_side:
            side_key = "side_angle_A" if "side_angle_A" in views_dict else "side_angle_B"
            st_side = views_dict[side_key][0]
            col_side = STATUS_COLORS_HEX[st_side]
            has_single_view = False
        else:
            st_side = "INDISPONIVEL"
            col_side = "#64748b"
            has_single_view = True

        if has_single_view:
            has_paralax_divergence = False
            if st_front == "CRITICO":
                cons_triage = "CRITICO_MONO_VISADA"
            elif st_front == "PROXIMIDADE":
                cons_triage = "PROXIMIDADE_MONO_VISADA"
            else:
                cons_triage = "SEGURO_MONO_VISADA"
        else:
            if st_front == "CRITICO" and st_side == "CRITICO":
                cons_triage = "CRITICO_CONFIRMADO"
            elif (st_front == "CRITICO" and st_side == "SEGURO") or (st_front == "SEGURO" and st_side == "CRITICO"):
                cons_triage = "DIVERGENCIA_PARALAXE"
            elif st_front == "CRITICO" or st_side == "CRITICO":
                cons_triage = "DIVERGENCIA_PARALAXE"
            elif st_front == "PROXIMIDADE" or st_side == "PROXIMIDADE":
                cons_triage = "PROXIMIDADE"
            else:
                cons_triage = "SEGURO"
            has_paralax_divergence = (cons_triage == "DIVERGENCIA_PARALAXE")

        tree["v3_triage"] = {
            "front_triage": st_front,
            "front_color": col_front,
            "side_triage": st_side,
            "side_color": col_side,
            "consolidated": cons_triage,
            "has_paralax_divergence": has_paralax_divergence,
            "has_single_view": has_single_view,
        }

        # Contagem comparativa por região
        v2_c = tree.get("v2_triage", {}).get("consolidated", "SEGURO")
        for prefix, c in [("v2", v2_c), ("v3", cons_triage)]:
            if c == "CRITICO_CONFIRMADO":
                region_stats[reg][f"{prefix}_crit"] += 1
            elif c == "DIVERGENCIA_PARALAXE":
                region_stats[reg][f"{prefix}_paralax"] += 1
            elif c in ("PROXIMIDADE", "PROXIMIDADE_MONO_VISADA", "CRITICO_MONITORADO"):
                region_stats[reg][f"{prefix}_prox"] += 1
            else:
                region_stats[reg][f"{prefix}_safe"] += 1

        if t_idx % 50 == 0 or t_idx == len(trees):
            el = time.time() - t0_global
            print(f"[{t_idx:03d}/{len(trees)}] Árvores processadas ({processed_views} fotos) em {el:.1f}s")

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifesto atualizado com v3_analysis / v3_triage: {MANIFEST_PATH}")

    print("\n" + "=" * 85)
    print("COMPARATIVO POR TERRITÓRIO (V2 FULL vs. V3 MODELO C — HARD NEGATIVES)")
    print("=" * 85)
    for reg, s in region_stats.items():
        print(f"• {reg} ({s['trees']} árvores):")
        print(f"   V2 Full -> Crítico: {s['v2_crit']} | Paralaxe: {s['v2_paralax']} | Prox: {s['v2_prox']} | Seguro: {s['v2_safe']} | Fio Total: {s['v2_wire_px_sum']:,} px")
        print(f"   V3-C    -> Crítico: {s['v3_crit']} | Paralaxe: {s['v3_paralax']} | Prox: {s['v3_prox']} | Seguro: {s['v3_safe']} | Fio Total: {s['v3_wire_px_sum']:,} px")

    import sys
    if str(CV_ROOT) not in sys.path:
        sys.path.insert(0, str(CV_ROOT))
    from scripts.generate_interactive_map import generate_html_map
    generate_html_map(manifest)


if __name__ == "__main__":
    main()
