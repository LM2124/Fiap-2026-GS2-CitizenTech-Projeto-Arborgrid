"""
ArborGrid - Amostragem e Inferência de Controle Negativo na Avenida Paulista
Módulo: arborgrid-vision / scripts / fetch_and_infer_paulista_control.py

Coleta e analisa uma amostra de árvores ao longo do corredor da Avenida Paulista
(região com fiação elétrica 100% subterrânea). Serve como Grupo de Controle Negativo
para comprovar perante a banca acadêmica que o modelo SegFormer V2 Full não alucina
fios e que o sistema discrimina com precisão vias com rede aérea vs rede subterrânea.
"""

import os
import sys
import json
import time
import math
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from transformers import SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
SP_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"
IMAGES_DIR = SP_DIR / "images"
V2_OVERLAYS_DIR = SP_DIR / "v2_overlays"
MANIFEST_PATH = SP_DIR / "manifest.json"
MODELS_DIR = CV_ROOT / "models"
V2_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_v2_full_local.pt"

SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

sys.path.append(str(CV_ROOT))
from scripts.fetch_geosampa_streetview import (
    load_api_key,
    fetch_geosampa_trees,
    collect_tree_perspectives,
    download_sv_image,
)
from scripts.run_v2_inference_and_update_map import (
    analyze_conflict_morphology_calibrated,
    render_v2_overlay,
    update_html_map,
    STATUS_COLORS_HEX,
)

# Eixo viário da Avenida Paulista
LAT1, LON1 = -23.5552, -46.6625  # Praça do Ciclista / Consolação
LAT2, LON2 = -23.5714, -46.6437  # Praça Oswaldo Cruz / Paraíso


def dist_to_paulista_m(lat: float, lon: float) -> float:
    x = (lon - LON1) * 101750.0
    y = (lat - LAT1) * 111000.0
    dx = (LON2 - LON1) * 101750.0
    dy = (LAT2 - LAT1) * 111000.0
    seg_len_sq = dx * dx + dy * dy
    t = max(0.0, min(1.0, (x * dx + y * dy) / seg_len_sq))
    return math.hypot(x - t * dx, y - t * dy)


def pos_along_paulista_m(lat: float, lon: float) -> float:
    x = (lon - LON1) * 101750.0
    y = (lat - LAT1) * 111000.0
    dx = (LON2 - LON1) * 101750.0
    dy = (LAT2 - LAT1) * 111000.0
    seg_len = math.hypot(dx, dy)
    t = max(0.0, min(1.0, (x * dx + y * dy) / (seg_len * seg_len)))
    return t * seg_len


def main():
    api_key = load_api_key()
    if not api_key:
        print("Erro: GOOGLE_MAPS_API_KEY não encontrada.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("ARBORGRID — CONTROLE NEGATIVO: AVENIDA PAULISTA (FIAÇÃO SUBTERRÂNEA)")
    print(f"Dispositivo: {device}")
    print("=" * 80)

    # 1. Carregar manifesto existente
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing_ids = {str(t.get("geosampa_id")) for t in manifest.get("trees", [])}

    # 2. Consultar árvores no GeoSampa no recorte da Paulista
    bbox_paulista = (-46.6660, -23.5730, -46.6410, -23.5530)
    print("Consultando GeoSampa (Arborização Viária) no corredor da Paulista...")
    data = fetch_geosampa_trees(bbox_paulista, max_features=5000)
    features = data.get("features", [])
    print(f"Total de registros na área expandida: {len(features)}")

    # 3. Filtrar árvores situadas no corredor da avenida (raio <= 45m da linha central)
    dx = (LON2 - LON1) * 101750.0
    dy = (LAT2 - LAT1) * 111000.0

    candidates_north = []
    candidates_south = []

    for f in features:
        cid = str(f["properties"].get("cd_identificador", ""))
        lon, lat = f["geometry"]["coordinates"][:2]
        d_center = dist_to_paulista_m(lat, lon)
        if d_center <= 45.0:
            s_pos = pos_along_paulista_m(lat, lon)
            px = (lon - LON1) * 101750.0
            py = (lat - LAT1) * 111000.0
            cross = px * dy - py * dx
            if cross >= 0:
                candidates_north.append((f, cid, d_center, s_pos))
            else:
                candidates_south.append((f, cid, d_center, s_pos))

    candidates_north.sort(key=lambda x: x[3])
    candidates_south.sort(key=lambda x: x[3])
    print(f"Candidatas na calçada Norte (Bela Vista): {len(candidates_north)}")
    print(f"Candidatas na calçada Sul (Jardins): {len(candidates_south)}")

    # 4. Amostragem integral em cada calçada (sem exclusão por filtro de espaçamento linear)
    selected_trees = []

    for sidewalk_candidates in (candidates_north, candidates_south):
        for feat, cid, d_center, s_pos in sidewalk_candidates:
            if cid in existing_ids:
                continue
            selected_trees.append(feat)

    print(f"Novas árvores selecionadas para cobrir 100% da calha da Paulista: {len(selected_trees)}")

    # 5. Coleta de Street View
    print("\n[1/2] Coletando Street View para o grupo de controle...")
    new_trees_records = []
    for idx, feat in enumerate(selected_trees, start=1):
        props = feat.get("properties") or {}
        tree_id = str(props.get("cd_identificador", f"paulista_{idx}"))
        lon, lat = feat["geometry"]["coordinates"][:2]

        views = collect_tree_perspectives(lat, lon, api_key=api_key, num_views=2)
        if not views:
            print(f"  [{idx}/{len(selected_trees)}] ID #{tree_id}: sem Street View próximo.")
            continue

        saved_views = []
        for v in views:
            fname = f"tree_{tree_id}_{v['perspective']}_h{int(v['heading'])}.jpg"
            img_path = IMAGES_DIR / fname
            if not img_path.exists():
                download_sv_image(v, api_key=api_key, out_path=img_path)
                time.sleep(0.08)
            v_rec = dict(v)
            v_rec["file_name"] = fname
            saved_views.append(v_rec)

        new_trees_records.append({
            "geosampa_id": tree_id,
            "region_key": "avenida_paulista",
            "region_label": "Avenida Paulista (Controle Negativo - Fiação Subterrânea)",
            "distrito": props.get("nm_distrito_municipal", "BELA VISTA"),
            "subprefeitura": props.get("nm_subprefeitura", "SE"),
            "tree_lat": lat,
            "tree_lon": lon,
            "sisgau_last_inspection": props.get("dt_inspecao"),
            "views": saved_views,
            "dataset_role": "negative_control",
        })
        print(f"  [{idx}/{len(selected_trees)}] ID #{tree_id}: {len(saved_views)} visadas obtidas.")

    print(f"\nTotal de árvores coletadas com sucesso: {len(new_trees_records)}")

    # 6. Carregar Modelo SegFormer V2 Full para inferência
    print("\n[2/2] Executando inferência SegFormer-B2 1024px V2 Full nas fotos da Paulista...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(V2_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    paulista_stats = {"CRITICO": 0, "PROXIMIDADE": 0, "SEGURO": 0}
    consensus_stats = {"AMBOS_CRITICO": 0, "DIVERGENCIA_PARALAXE": 0, "PROXIMIDADE": 0, "AMBOS_SEGURO": 0, "MONO_VISADA": 0}

    for t_idx, tree in enumerate(new_trees_records, start=1):
        views_res = []
        for v in tree.get("views", []):
            img_path = IMAGES_DIR / v["file_name"]
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
                0: (probs[0] >= 0.50).astype(np.uint8),
                1: (probs[1] >= 0.38).astype(np.uint8),
                2: (probs[2] >= 0.48).astype(np.uint8),
            }

            triage, px_direct, px_wind, wire_px, copa_px, inter_direct, inter_wind, clean_wire, clean_copa = analyze_conflict_morphology_calibrated(
                raw_masks[0], raw_masks[1]
            )
            paulista_stats[triage] += 1

            disp_masks = {0: clean_copa, 1: clean_wire, 2: raw_masks[2]}
            ov_name = f"ov_v2_{v['file_name']}"
            ov_path = V2_OVERLAYS_DIR / ov_name
            title = f"{triage} (V2) | Copa: {copa_px} px | Fio: {wire_px} px | Contato: {px_direct} px"
            canvas = render_v2_overlay(orig_bgr, disp_masks, inter_direct, triage, title)
            cv2.imwrite(str(ov_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            v["v2_analysis"] = {
                "triage": triage,
                "color_hex": STATUS_COLORS_HEX[triage],
                "copa_px": copa_px,
                "wire_px": wire_px,
                "pole_px": int(np.count_nonzero(raw_masks[2])),
                "contact_direct_px": px_direct,
                "wind_buffer_px": px_wind,
                "v2_overlay_file": ov_name,
            }
            views_res.append((v["perspective"], triage, STATUS_COLORS_HEX[triage]))

        # Consolidação
        views_dict = {r[0]: (r[1], r[2]) for r in views_res}
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
            consensus_stats["MONO_VISADA"] += 1
            if st_front == "CRITICO":
                cons_triage = "CRITICO_MONO_VISADA"
            elif st_front == "PROXIMIDADE":
                cons_triage = "PROXIMIDADE_MONO_VISADA"
            else:
                cons_triage = "SEGURO_MONO_VISADA"
        else:
            if st_front == "CRITICO" and st_side == "CRITICO":
                cons_triage = "CRITICO_CONFIRMADO"
                consensus_stats["AMBOS_CRITICO"] += 1
            elif (st_front == "CRITICO" and st_side == "SEGURO") or (st_front == "SEGURO" and st_side == "CRITICO"):
                cons_triage = "DIVERGENCIA_PARALAXE"
                consensus_stats["DIVERGENCIA_PARALAXE"] += 1
            elif st_front == "CRITICO" or st_side == "CRITICO":
                cons_triage = "DIVERGENCIA_PARALAXE"
                consensus_stats["DIVERGENCIA_PARALAXE"] += 1
            elif st_front == "PROXIMIDADE" or st_side == "PROXIMIDADE":
                cons_triage = "PROXIMIDADE"
                consensus_stats["PROXIMIDADE"] += 1
            else:
                cons_triage = "SEGURO"
                consensus_stats["AMBOS_SEGURO"] += 1
            has_paralax_divergence = (cons_triage == "DIVERGENCIA_PARALAXE")

        tree["v2_triage"] = {
            "front_triage": st_front,
            "front_color": col_front,
            "side_triage": st_side,
            "side_color": col_side,
            "consolidated": cons_triage,
            "has_paralax_divergence": has_paralax_divergence,
            "has_single_view": has_single_view,
        }

    # 7. Atualizar manifesto
    manifest["trees"].extend(new_trees_records)

    regions_catalog = manifest.get("regions_catalog", {})
    if "avenida_paulista" in regions_catalog:
        total_p = len([t for t in manifest["trees"] if t.get("region_key") == "avenida_paulista"])
        regions_catalog["avenida_paulista"]["total_geosampa_trees"] = total_p
    manifest["regions_catalog"] = regions_catalog

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifesto atualizado com +{len(new_trees_records)} árvores da Paulista!")
    print(f"Estatísticas de Visadas na Paulista: {paulista_stats}")
    print(f"Estatísticas de Consenso na Paulista: {consensus_stats}")

    # 8. Atualizar Mapa HTML
    update_html_map(manifest)
    print("Mapa HTML atualizado com sucesso!")


if __name__ == "__main__":
    main()

