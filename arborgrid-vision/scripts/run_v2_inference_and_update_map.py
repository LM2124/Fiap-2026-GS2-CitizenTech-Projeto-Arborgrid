"""
ArborGrid - Inferência do Modelo V2 Full 1024px e Atualização do Mapa com Morfologia Calibrada
Módulo: arborgrid-vision / scripts / run_v2_inference_and_update_map.py

Executa inferência do SegFormer-B2 1024px V2 Full (treinado no dataset de 132 imagens)
sobre todas as 197 árvores de SP (Pinheiros, Alto da Mooca e Vila Mariana - 389 fotos).
Aplica a morfologia calibrada com filtro de ruído fino (<80px) e eliminação de mudas isoladas (<15%),
gera os overlays de produção em 'v2_overlays' e atualiza o mapa interativo HTML.
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
V2_OVERLAYS_DIR = SP_DIR / "v2_overlays"
V2_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = SP_DIR / "manifest.json"

MODELS_DIR = CV_ROOT / "models"
V2_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_v2_full_local.pt"
if not V2_WEIGHTS.exists():
    V2_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_v2_full.pt"

SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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


def load_model(device: str):
    print(f"Carregando pesos do SegFormer V2 Full: {V2_WEIGHTS.name}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    )
    sd = torch.load(V2_WEIGHTS, map_location="cpu", weights_only=True)
    if any(k.startswith("segformer.stages.") for k in sd.keys()):
        sd = {remap_colab_to_local(k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def analyze_conflict_morphology_calibrated(mask_copa: np.ndarray, mask_wire: np.ndarray):
    """
    Motor morfológico calibrado com filtros determinísticos anti-ruído:
    1. Descarte de fragmentos de fiação < 80 px (evita falsos alarmes em reflexos pontuais).
    2. Foco na massa arbórea principal (descarte de pequenos trechos < 15% da copa dominante ou < 800px).
    3. Dilatação elíptica: contato direto (r1=6px) e oscilação de vento (r2=24px).
    4. Limiares físicos calibrados para ambiente urbano de São Paulo.
    """
    # 1. Filtro de ruído em fiação: remove ilhas < 80 px
    num_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(mask_wire, connectivity=8)
    clean_wire = np.zeros_like(mask_wire)
    for i in range(1, num_w):
        if stats_w[i, cv2.CC_STAT_AREA] >= 80:
            clean_wire[labels_w == i] = 1

    # 2. Filtro de copa: foca na copa principal da árvore
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

    # 3. Dilatações morfológicas elípticas
    k_direct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))   # raio ~6px
    k_wind = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))     # raio ~24px

    wire_direct = cv2.dilate(clean_wire, k_direct)
    wire_wind = cv2.dilate(clean_wire, k_wind)

    inter_direct = (wire_direct > 0) & (clean_copa > 0)
    inter_wind = (wire_wind > 0) & (clean_copa > 0) & (~inter_direct)

    px_direct = int(np.count_nonzero(inter_direct))
    px_wind = int(np.count_nonzero(inter_wind))
    wire_px = int(np.count_nonzero(clean_wire))
    copa_px = int(np.count_nonzero(clean_copa))

    # 4. Classificação calibrada (limiares realistas para infraestrutura pesada)
    if px_direct >= 3000 and wire_px >= 800:
        triage = "CRITICO"
    elif (px_direct >= 250 or px_wind >= 1200) and wire_px >= 200:
        triage = "PROXIMIDADE"
    else:
        triage = "SEGURO"

    return triage, px_direct, px_wind, wire_px, copa_px, inter_direct, inter_wind, clean_wire, clean_copa


def render_v2_overlay(orig_bgr: np.ndarray, masks: dict, inter_direct: np.ndarray, triage: str, title: str) -> np.ndarray:
    canvas = orig_bgr.copy()
    overlay = orig_bgr.copy()

    # Preenchimento base das classes
    for cid in (0, 2, 1):
        m = np.asarray(masks[cid], dtype=np.uint8)
        if np.any(m):
            overlay[m > 0] = COLORS_BGR[cid]

    # Destaque de contato direto em Magenta/Vermelho brilhante
    if np.any(inter_direct):
        overlay[inter_direct] = (255, 0, 220)

    cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)

    # Contornos nítidos
    for cid in (0, 2, 1):
        m = np.asarray(masks[cid], dtype=np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(canvas, cnts, -1, COLORS_BGR[cid], 2 if cid != 0 else 1, cv2.LINE_AA)

    if np.any(inter_direct):
        cnts_i, _ = cv2.findContours(inter_direct.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_i:
            cv2.drawContours(canvas, cnts_i, -1, (255, 255, 255), 2, cv2.LINE_AA)

    # Faixa superior de telemetria
    banner_color = {"CRITICO": (30, 30, 220), "PROXIMIDADE": (20, 140, 240), "SEGURO": (40, 160, 40)}.get(triage, (15, 23, 42))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), banner_color, -1)
    cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("EXECUÇÃO DO MODELO V2 FULL 1024PX — INFERÊNCIA E MAPEAMENTO COMPLETO DE SP")
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    trees = manifest.get("trees", [])

    model = load_model(device)

    target_trees = [t for t in trees if t.get("region_key") in ("alto_da_mooca", "vila_mariana", "pinheiros")]
    print(f"Total de árvores selecionadas para processamento: {len(target_trees)} ({len(target_trees)*2} imagens)")

    t0_global = time.time()
    processed_views = 0

    stats_triage = {"CRITICO": 0, "PROXIMIDADE": 0, "SEGURO": 0}
    stats_consensus = {"AMBOS_CRITICO": 0, "DIVERGENCIA_PARALAXE": 0, "PROXIMIDADE": 0, "AMBOS_SEGURO": 0}

    for t_idx, tree in enumerate(target_trees, start=1):
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
                0: (probs[0] >= 0.50).astype(np.uint8),
                1: (probs[1] >= 0.38).astype(np.uint8),
                2: (probs[2] >= 0.48).astype(np.uint8),
            }

            triage, px_direct, px_wind, wire_px, copa_px, inter_direct, inter_wind, clean_wire, clean_copa = analyze_conflict_morphology_calibrated(
                raw_masks[0], raw_masks[1]
            )
            stats_triage[triage] += 1

            disp_masks = {0: clean_copa, 1: clean_wire, 2: raw_masks[2]}
            ov_name = f"ov_v2_{img_file}"
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
            views_results.append((v.get("perspective"), triage, STATUS_COLORS_HEX[triage]))
            processed_views += 1

        # Consolidação da Árvore
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

        # Consenso Multi-Perspectiva
        if has_single_view:
            has_paralax_divergence = False
            stats_consensus["MONO_VISADA"] = stats_consensus.get("MONO_VISADA", 0) + 1
            if st_front == "CRITICO":
                cons_triage = "CRITICO_MONO_VISADA"
            elif st_front == "PROXIMIDADE":
                cons_triage = "PROXIMIDADE_MONO_VISADA"
            else:
                cons_triage = "SEGURO_MONO_VISADA"
        else:
            if st_front == "CRITICO" and st_side == "CRITICO":
                cons_triage = "CRITICO_CONFIRMADO"
                stats_consensus["AMBOS_CRITICO"] += 1
            elif (st_front == "CRITICO" and st_side == "SEGURO") or (st_front == "SEGURO" and st_side == "CRITICO"):
                cons_triage = "DIVERGENCIA_PARALAXE"
                stats_consensus["DIVERGENCIA_PARALAXE"] += 1
            elif st_front == "CRITICO" or st_side == "CRITICO":
                cons_triage = "DIVERGENCIA_PARALAXE"
                stats_consensus["DIVERGENCIA_PARALAXE"] += 1
            elif st_front == "PROXIMIDADE" or st_side == "PROXIMIDADE":
                cons_triage = "PROXIMIDADE"
                stats_consensus["PROXIMIDADE"] += 1
            else:
                cons_triage = "SEGURO"
                stats_consensus["AMBOS_SEGURO"] += 1
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

        if t_idx % 25 == 0 or t_idx == len(target_trees):
            el = time.time() - t0_global
            print(f"[{t_idx:03d}/{len(target_trees)}] Árvores processadas ({processed_views} fotos) em {el:.1f}s ({el/max(processed_views,1):.2f}s/foto)")

    # Salva manifest atualizado
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest atualizado com sucesso em: {MANIFEST_PATH}")
    print(f"Estatísticas de Visadas Individuais (V2 Calibrado): {stats_triage}")
    print(f"Estatísticas de Consenso por Árvore (V2 Calibrado): {stats_consensus}")

    # Atualiza o Mapa HTML
    update_html_map(manifest)


def update_html_map(manifest: dict):
    """
    Gera o novo mapa HTML delegando para generate_interactive_map.
    """
    from scripts.generate_interactive_map import generate_html_map
    return generate_html_map(manifest)


def _legacy_update_html_map(manifest: dict):
    map_path = SP_DIR / "mapa_arborgrid_sp.html"
    trees_js = json.dumps(manifest.get("trees", []), ensure_ascii=False)
    regions_js = json.dumps(manifest.get("regions_catalog", {}), ensure_ascii=False)

    html_code = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>ArborGrid SP — Mapa Multi-Perspectiva com Marcadores Bicolores (SegFormer 1024px V2 Full)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; }}
    #topbar {{
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
      padding: 9px 16px; background: #1e293b; border-bottom: 1px solid #334155; gap: 10px;
    }}
    .brand {{ font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 8px; }}
    .region-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }}
    .reg-btn {{
      padding: 5px 11px; border-radius: 6px; border: 1px solid #475569; background: #0f172a;
      color: #e2e8f0; font-size: 12px; font-weight: 600; cursor: pointer; transition: 0.15s;
    }}
    .reg-btn:hover {{ background: #334155; }}
    .reg-btn.active {{ background: #2563eb; border-color: #3b82f6; color: #ffffff; }}
    .controls-right {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; }}
    .preset-select {{
      background: #0f172a; color: #f8fafc; border: 1px solid #10b981; border-radius: 6px;
      padding: 5px 9px; font-size: 12px; font-weight: 600; cursor: pointer;
    }}
    .legend {{ display: flex; gap: 12px; align-items: center; background: #0f172a; padding: 4px 10px; border-radius: 6px; border: 1px solid #334155; }}
    .split-legend-icon {{ width: 14px; height: 14px; vertical-align: middle; margin-right: 4px; }}
    #statsbar {{
      display: flex; gap: 18px; padding: 6px 16px; background: #0f172a;
      border-bottom: 1px solid #1e293b; font-size: 12px; color: #cbd5e1; flex-wrap: wrap;
    }}
    #statsbar b {{ color: #f8fafc; }}
    #map {{ height: calc(100vh - 84px); width: 100%; }}
    .popup-card {{ color: #0f172a; min-width: 520px; max-width: 640px; }}
    .views-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }}
    .view-box {{ background: #f1f5f9; padding: 7px; border-radius: 6px; font-size: 12px; border: 1px solid #cbd5e1; }}
    .view-box img {{ width: 100%; border-radius: 4px; display: block; margin: 5px 0; cursor: pointer; }}
    .gmaps-btn {{
      display: inline-block; margin-top: 5px; padding: 5px 9px; background: #2563eb;
      color: #fff !important; text-decoration: none; border-radius: 4px; font-weight: 600; font-size: 11px;
    }}
    .gmaps-btn:hover {{ background: #1d4ed8; }}
    .badge {{ padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; color: #fff; }}
    .leaflet-div-icon {{ background: transparent; border: none; }}
    .bbox-toggle-label {{
      display: inline-flex; align-items: center; gap: 5px; cursor: pointer;
      background: #0f172a; padding: 5px 9px; border-radius: 6px; border: 1px solid #334155;
      font-size: 12px; font-weight: 600; color: #cbd5e1; user-select: none;
    }}
    .bbox-toggle-label:hover {{ border-color: #64748b; color: #f8fafc; }}
    .bbox-toggle-label input {{ cursor: pointer; accent-color: #2563eb; }}
    .region-bbox-popup {{
      background: #0f172a !important; color: #f8fafc !important; border: 1px solid #334155 !important;
      border-radius: 6px !important; box-shadow: 0 4px 14px rgba(0,0,0,0.5) !important; padding: 8px 11px !important;
      pointer-events: none !important; user-select: none !important;
    }}
    .region-bbox-popup .leaflet-tooltip-pane {{ z-index: 650; pointer-events: none !important; }}
  </style>
</head>
<body>
  <div id="topbar">
    <div class="brand">ArborGrid SP — Auditoria Georreferenciada Multi-Perspectiva</div>
    <div class="region-tabs" id="regionTabs"></div>
    <div class="controls-right">
      <select id="presetSelect" class="preset-select" title="Modo de visualização dos modelos">
        <option value="v2_full_bicolor" selected>★ Modo V2 Full (Dataset 132 Imgs + Morfologia Calibrada Anti-Ruído)</option>
        <option value="gold_segformer_bicolor">Modo V1 Ouro (Dataset 68 Imgs)</option>
        <option value="all">Preset Clássico (YOLO/GPU Inicial)</option>
      </select>
      <label class="bbox-toggle-label" title="Exibir ou ocultar as delimitações das áreas de amostragem">
        <input type="checkbox" id="toggleBBoxes" checked />
        <span>Delimitações</span>
      </label>
      <div class="legend">
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="#ef4444" stroke="#fff"/></svg> Crítico (Contato)</span>
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="#f59e0b" stroke="#fff"/></svg> Proximidade (Vento)</span>
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="#10b981" stroke="#fff"/></svg> Seguro</span>
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><path d="M 10 2 A 8 8 0 0 0 10 18 Z" fill="#ef4444"/><path d="M 10 2 A 8 8 0 0 1 10 18 Z" fill="#10b981"/><circle cx="10" cy="10" r="8" fill="none" stroke="#fff"/></svg> Bicolor (Paralaxe)</span>
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><path d="M 10 2 A 8 8 0 0 0 10 18 Z" fill="#ef4444"/><path d="M 10 2 A 8 8 0 0 1 10 18 Z" fill="#64748b"/><circle cx="10" cy="10" r="8" fill="none" stroke="#fff"/></svg> Mono-Visada (Sem 2º Ângulo)</span>
      </div>
    </div>
  </div>

  <div id="statsbar">
    <div>Região: <b id="statRegion">Todas</b></div>
    <div>Total Árvores: <b id="statTotal">0</b></div>
    <div>Crítico Confirmado (Ambos): <b id="statCritConfirm" style="color:#ef4444;">0</b></div>
    <div>Divergência de Paralaxe: <b id="statParalax" style="color:#f97316;">0</b></div>
    <div>Mono-Visada: <b id="statMono" style="color:#94a3b8;">0</b></div>
    <div>Proximidade Monitorada: <b id="statProx" style="color:#f59e0b;">0</b></div>
    <div>Seguro: <b id="statSafe" style="color:#10b981;">0</b></div>
  </div>

  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const trees = {trees_js};
    const regionsCatalog = {regions_js};

    const map = L.map('map', {{ zoomControl: true }}).setView([-23.5650, -46.6450], 13);

    const esriTopo = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Tiles &copy; Esri | GeoSampa/PMSP | Google Street View', maxZoom: 19 }}
    );
    const esriStreet = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Tiles &copy; Esri | GeoSampa/PMSP | Google Street View', maxZoom: 19 }}
    );
    const esriSat = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Satélite &copy; Esri, Maxar | GeoSampa/PMSP', maxZoom: 19 }}
    );
    const cartoVoyager = L.tileLayer(
      'https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png',
      {{ attribution: '&copy; OpenStreetMap contributors &copy; CARTO', subdomains: 'abcd', maxZoom: 20 }}
    );
    const cartoPositron = L.tileLayer(
      'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
      {{ attribution: '&copy; OpenStreetMap contributors &copy; CARTO', subdomains: 'abcd', maxZoom: 20 }}
    );

    esriTopo.addTo(map);

    const regionsLayer = L.layerGroup().addTo(map);
    const auditedLayer = L.layerGroup().addTo(map);

    L.control.layers(
      {{
        'Mapa Topo (Esri)': esriTopo,
        'Mapa de Ruas (Esri Street)': esriStreet,
        'Satélite HD (Esri Imagery)': esriSat,
        'OpenStreetMap Voyager (Carto)': cartoVoyager,
        'OpenStreetMap Claro (Positron)': cartoPositron
      }},
      {{
        'Delimitações das Regiões': regionsLayer,
        'Árvores Auditadas': auditedLayer
      }},
      {{ position: 'topright' }}
    ).addTo(map);

    let currentRegion = 'ALL';
    let currentPreset = 'v2_full_bicolor';

    function createSplitMarkerIcon(colorFront, colorSide, isParalax) {{
      const size = isParalax ? 24 : 20;
      const r = size / 2;
      const strokeCol = isParalax ? '#ffffff' : '#0f172a';
      const strokeW = isParalax ? 2.5 : 1.5;

      const svg = `
        <svg width="${{size}}" height="${{size}}" viewBox="0 0 ${{size}} ${{size}}" style="filter: drop-shadow(0 2px 4px rgba(0,0,0,0.45));">
          <!-- Metade Esquerda: Visada Frontal (center_front) -->
          <path d="M ${{r}} 1 A ${{r-1}} ${{r-1}} 0 0 0 ${{r}} ${{size-1}} Z" fill="${{colorFront}}" stroke="${{strokeCol}}" stroke-width="${{strokeW}}" />
          <!-- Metade Direita: Visada Lateral (side_angle_A) -->
          <path d="M ${{r}} 1 A ${{r-1}} ${{r-1}} 0 0 1 ${{r}} ${{size-1}} Z" fill="${{colorSide}}" stroke="${{strokeCol}}" stroke-width="${{strokeW}}" />
          <!-- Linha divisória vertical -->
          <line x1="${{r}}" y1="1" x2="${{r}}" y2="${{size-1}}" stroke="${{strokeCol}}" stroke-width="${{strokeW}}" />
        </svg>`;

      return L.divIcon({{
        html: svg,
        className: 'split-marker',
        iconSize: [size, size],
        iconAnchor: [r, r]
      }});
    }}

    function renderMarkers() {{
      auditedLayer.clearLayers();
      const filtered = currentRegion === 'ALL'
        ? trees
        : trees.filter(t => (t.region_key || 'pinheiros') === currentRegion);

      filtered.forEach(t => {{
        let colFront = '#10b981', colSide = '#10b981', isParalax = false, consTriage = 'SEGURO';

        if (currentPreset === 'v2_full_bicolor') {{
          const vt = t.v2_triage || {{}};
          colFront = vt.front_color || '#10b981';
          colSide  = vt.side_color || '#10b981';
          isParalax = Boolean(vt.has_paralax_divergence);
          consTriage = vt.consolidated || 'SEGURO';
        }} else if (currentPreset === 'gold_segformer_bicolor') {{
          const gt = t.gold_triage || {{}};
          colFront = gt.front_color || '#10b981';
          colSide  = gt.side_color || '#10b981';
          isParalax = Boolean(gt.has_paralax_divergence);
          consTriage = gt.consolidated || 'SEGURO';
        }} else {{
          // Classic preset
          const c = t.consolidated_risk || 'LOW';
          const col = (c === 'HIGH') ? '#ef4444' : (c === 'MEDIUM') ? '#f59e0b' : '#10b981';
          colFront = col; colSide = col;
          consTriage = (c === 'HIGH') ? 'CRITICO_CONFIRMADO' : (c === 'MEDIUM') ? 'PROXIMIDADE' : 'SEGURO';
        }}

        const icon = createSplitMarkerIcon(colFront, colSide, isParalax);
        const marker = L.marker([t.tree_lat, t.tree_lon], {{ icon: icon }});

        let viewsHtml = '';
        (t.views || []).forEach(v => {{
          let an = {{}}, ovPath = '', triageCol = '#475569';
          if (currentPreset === 'v2_full_bicolor') {{
            an = v.v2_analysis || {{}};
            ovPath = an.v2_overlay_file ? ('v2_overlays/' + an.v2_overlay_file) : ('images/' + v.file_name);
            triageCol = an.color_hex || '#475569';
          }} else if (currentPreset === 'gold_segformer_bicolor') {{
            an = v.gold_analysis || {{}};
            ovPath = an.gold_overlay_file ? ('gold_overlays/' + an.gold_overlay_file) : ('images/' + v.file_name);
            triageCol = an.color_hex || '#475569';
          }} else {{
            an = v.gpu_analysis || {{}};
            ovPath = an.overlay_file ? ('gpu_overlays/' + an.overlay_file) : ('images/' + v.file_name);
            triageCol = (an.risk_level === 'HIGH') ? '#ef4444' : (an.risk_level === 'MEDIUM') ? '#f59e0b' : '#10b981';
          }}

          const rawPath = 'images/' + v.file_name;
          viewsHtml += `
            <div class="view-box">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <b>${{v.perspective}}</b>
                <span class="badge" style="background:${{triageCol}}">${{an.triage || an.risk_level || 'SEGURO'}}</span>
              </div>
              <div style="font-size:11px;color:#64748b;">${{v.distance_m}}m da câmera · fov=${{v.fov}}°</div>
              <img src="${{ovPath}}"
                   onmouseover="this.src='${{rawPath}}'" onmouseout="this.src='${{ovPath}}'"
                   title="Passe o mouse para comparar com a foto original" />
              <div style="color:#334155;font-size:11px;">
                Copa: ${{an.copa_px || an.copa_pct || 0}} px | Fio: ${{an.wire_px || 0}} px | <b>Contato: ${{an.contact_direct_px || an.conflict_overlap_px || 0}} px</b>
              </div>
            </div>`;
        }});

        if (t.views && t.views.length === 1) {{
          viewsHtml += `
            <div class="view-box" style="display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;background:#f8fafc;border:2px dashed #94a3b8;color:#64748b;padding:12px 10px;">
              <div style="font-size:20px;margin-bottom:4px;">📷❌</div>
              <b style="color:#334155;font-size:12px;">Visada Lateral Indisponível</b>
              <div style="font-size:11px;color:#64748b;margin-top:4px;line-height:1.35;">
                Sem 2º panorama do Street View a &le;28m na malha viária.<br>
                <b>Análise mono-perspectiva</b> (sem confirmação de paralaxe).
              </div>
            </div>`;
        }}

        const bottomNote = isParalax
          ? '⚠️ Divergência angular detectada — contato não confirmado em ambas as visadas.'
          : (t.views && t.views.length === 1)
            ? 'ℹ️ Visada única disponível no Street View — sem verificação de paralaxe angular.'
            : 'Análise consistente.';

        const gmapsUrl = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${{t.tree_lat}},${{t.tree_lon}}`;
        const popupContent = `
          <div class="popup-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <h3 style="margin:0;font-size:14px;">Árvore GeoSampa #${{t.geosampa_id}} (${{t.distrito || ''}})</h3>
              <span class="badge" style="background:${{colFront}}">${{consTriage}}</span>
            </div>
            <div style="font-size:11px;color:#64748b;margin:3px 0 8px 0;">
              Local: ${{t.tree_lat.toFixed(5)}}, ${{t.tree_lon.toFixed(5)}} | Espécie: ${{t.nome_cientifico || 'Indeterminada'}}
            </div>
            <div class="views-grid">${{viewsHtml}}</div>
            <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:11px;color:#475569;">${{bottomNote}}</span>
              <a href="${{gmapsUrl}}" target="_blank" class="gmaps-btn">Abrir Google Street View ↗</a>
            </div>
          </div>`;

        marker.bindPopup(popupContent, {{ maxWidth: 660 }});
        auditedLayer.addLayer(marker);
      }});

      updateStats(filtered);
    }}

    function updateStats(items) {{
      let critConfirm = 0, paralax = 0, mono = 0, prox = 0, safe = 0;
      items.forEach(t => {{
        let cons = '';
        if (currentPreset === 'v2_full_bicolor') {{
          cons = (t.v2_triage || {{}}).consolidated || 'SEGURO';
        }} else if (currentPreset === 'gold_segformer_bicolor') {{
          cons = (t.gold_triage || {{}}).consolidated || 'SEGURO';
        }} else {{
          const r = t.consolidated_risk || 'LOW';
          cons = (r === 'HIGH') ? 'CRITICO_CONFIRMADO' : (r === 'MEDIUM') ? 'PROXIMIDADE' : 'SEGURO';
        }}

        if (cons === 'CRITICO_CONFIRMADO') critConfirm++;
        else if (cons === 'DIVERGENCIA_PARALAXE') paralax++;
        else if (cons.indexOf('MONO_VISADA') !== -1) mono++;
        else if (cons === 'PROXIMIDADE' || cons === 'CRITICO_MONITORADO') prox++;
        else safe++;
      }});

      document.getElementById('statTotal').innerText = items.length;
      document.getElementById('statCritConfirm').innerText = critConfirm;
      document.getElementById('statParalax').innerText = paralax;
      document.getElementById('statMono').innerText = mono;
      document.getElementById('statProx').innerText = prox;
      document.getElementById('statSafe').innerText = safe;
    }}

    const regionStyles = {{
      pinheiros: {{ color: '#3b82f6', fill: '#2563eb' }},
      alto_da_mooca: {{ color: '#f59e0b', fill: '#d97706' }},
      vila_mariana: {{ color: '#a855f7', fill: '#7c3aed' }},
      avenida_paulista: {{ color: '#10b981', fill: '#059669' }}
    }};

    function renderRegionBoxes() {{
      regionsLayer.clearLayers();
      for (const [key, reg] of Object.entries(regionsCatalog)) {{
        if (!reg.bbox && !reg.corridor_polygon) continue;
        const st = regionStyles[key] || {{ color: '#64748b', fill: '#475569' }};

        const isSelected = (currentRegion === key);
        const isDimmed = (currentRegion !== 'ALL' && !isSelected);

        const shapeOptions = {{
          color: st.color,
          weight: isSelected ? 3.5 : 2,
          dashArray: isSelected ? null : '6, 6',
          fill: false,
          opacity: isSelected ? 1.0 : (isDimmed ? 0.25 : 0.85),
          interactive: true
        }};

        let shape;
        if (reg.corridor_polygon) {{
          shape = L.polygon(reg.corridor_polygon, shapeOptions);
        }} else {{
          const [minLon, minLat, maxLon, maxLat] = reg.bbox;
          shape = L.rectangle([[minLat, minLon], [maxLat, maxLon]], shapeOptions);
        }}

        const count = trees.filter(t => (t.region_key || 'pinheiros') === key).length;
        const gridBadge = reg.is_negative_control
          ? '<span style="color:#34d399;font-weight:700;">● Rede 100% Subterrânea (Controle Negativo)</span>'
          : '<span style="color:#fbbf24;font-weight:700;">▲ Rede Elétrica Aérea Convencional</span>';

        const geoTotal = reg.total_geosampa_trees ? `${{reg.total_geosampa_trees}} árvores` : 'Cadastradas no SISGAU/GeoSampa';
        const areaDesc = reg.corridor_polygon ? 'Polígono do Corredor Viário' : 'Retângulo Amostral (BBox)';
        const tooltipHtml = `
          <div style="font-family:system-ui; font-size:12px; line-height:1.45;">
            <div style="font-size:13px; font-weight:700; color:${{st.color}}; margin-bottom:3px;">${{reg.label}}</div>
            <div>${{gridBadge}}</div>
            <div style="margin-top:4px;">Árvores no GeoSampa: <b>${{geoTotal}}</b></div>
            <div>Amostra Auditada no Projeto: <b>${{count}} árvores</b></div>
            <div style="font-size:11px; color:#94a3b8; margin-top:3px;">${{areaDesc}}</div>
          </div>`;

        let hoverTimer = null;
        shape.on('mouseover', (e) => {{
          clearTimeout(hoverTimer);
          hoverTimer = setTimeout(() => {{
            shape.bindTooltip(tooltipHtml, {{
              sticky: false,
              className: 'region-bbox-popup',
              direction: 'top',
              offset: [0, -10]
            }}).openTooltip(e.latlng);
          }}, 500);
        }});

        shape.on('mouseout', () => {{
          clearTimeout(hoverTimer);
          if (shape.getTooltip()) {{
            shape.closeTooltip();
            shape.unbindTooltip();
          }}
        }});

        regionsLayer.addLayer(shape);
      }}
    }}

    function buildRegionTabs() {{
      const container = document.getElementById('regionTabs');
      container.innerHTML = '';

      const btnAll = document.createElement('button');
      btnAll.className = 'reg-btn active';
      btnAll.innerText = 'Todos os Bairros (' + trees.length + ')';
      btnAll.onclick = () => selectRegion('ALL', btnAll);
      container.appendChild(btnAll);

      for (const [key, reg] of Object.entries(regionsCatalog)) {{
        const count = trees.filter(t => (t.region_key || 'pinheiros') === key).length;
        if (count === 0) continue;
        const btn = document.createElement('button');
        btn.className = 'reg-btn';
        btn.setAttribute('data-key', key);
        btn.innerText = `${{reg.label}} (${{count}})`;
        btn.onclick = () => selectRegion(key, btn);
        container.appendChild(btn);
      }}
    }}

    function selectRegion(key, btn) {{
      currentRegion = key;
      document.querySelectorAll('.reg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (key !== 'ALL' && regionsCatalog[key]) {{
        const reg = regionsCatalog[key];
        if (reg.corridor_polygon) {{
          const poly = L.polygon(reg.corridor_polygon);
          map.fitBounds(poly.getBounds(), {{ padding: [50, 50], maxZoom: 16, duration: 1.0 }});
        }} else if (reg.bbox) {{
          const [minLon, minLat, maxLon, maxLat] = reg.bbox;
          map.fitBounds([[minLat, minLon], [maxLat, maxLon]], {{ padding: [50, 50], maxZoom: 16, duration: 1.0 }});
        }}
        document.getElementById('statRegion').innerText = reg.label;
      }} else {{
        map.flyTo([-23.5650, -46.6450], 13, {{ duration: 1.0 }});
        document.getElementById('statRegion').innerText = 'Todos os Bairros';
      }}
      renderRegionBoxes();
      renderMarkers();
    }}

    document.getElementById('presetSelect').onchange = (e) => {{
      currentPreset = e.target.value;
      renderMarkers();
    }};

    document.getElementById('toggleBBoxes').addEventListener('change', (e) => {{
      if (e.target.checked) {{
        map.addLayer(regionsLayer);
      }} else {{
        map.removeLayer(regionsLayer);
      }}
    }});

    buildRegionTabs();
    renderRegionBoxes();
    renderMarkers();
  </script>
</body>
</html>
"""
    map_path.write_text(html_code, encoding="utf-8")
    print(f"Mapa HTML atualizado com sucesso em: {map_path}")


if __name__ == "__main__":
    main()

