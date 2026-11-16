"""
ArborGrid - Inferência do Modelo Ouro 1024px e Atualização do Mapa Interativo com Marcador Bicolor
Módulo: arborgrid-vision / scripts / run_gold_inference_and_update_map.py

Executa inferência do SegFormer-B2 1024px Ouro nas árvores de outros bairros (Alto da Mooca e Vila Mariana),
calcula o conflito morfológico sem premissas frágeis de altura de cabos, gera overlays de alta definição
e atualiza o mapa HTML com marcadores bipartidos (metade esquerda = frontal, metade direita = lateral).
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
GOLD_OVERLAYS_DIR = SP_DIR / "gold_overlays"
GOLD_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = SP_DIR / "manifest.json"

MODELS_DIR = CV_ROOT / "models"
GOLD_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_gold_local.pt"
if not GOLD_WEIGHTS.exists():
    GOLD_WEIGHTS = MODELS_DIR / "arborgrid_segformer_b2_1024_gold.pt"

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
    print(f"Carregando pesos ouro do SegFormer: {GOLD_WEIGHTS.name}...")
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


def analyze_conflict_morphology(mask_copa: np.ndarray, mask_wire: np.ndarray):
    """
    Calcula o contato direto (r1=6px) e a zona preventiva de vento (r2=24px).
    Sem hipóteses frágeis de altura de cabos em pixels.
    """
    k_direct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))   # raio ~6px
    k_wind = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))     # raio ~24px

    wire_direct = cv2.dilate(mask_wire, k_direct)
    wire_wind = cv2.dilate(mask_wire, k_wind)

    inter_direct = (wire_direct > 0) & (mask_copa > 0)
    inter_wind = (wire_wind > 0) & (mask_copa > 0) & (~inter_direct)

    px_direct = int(np.count_nonzero(inter_direct))
    px_wind = int(np.count_nonzero(inter_wind))

    if px_direct >= 45:
        triage = "CRITICO"
    elif px_wind >= 110:
        triage = "PROXIMIDADE"
    else:
        triage = "SEGURO"

    return triage, px_direct, px_wind, inter_direct, inter_wind


def render_gold_overlay(orig_bgr: np.ndarray, masks: dict, inter_direct: np.ndarray, triage: str, title: str) -> np.ndarray:
    canvas = orig_bgr.copy()
    overlay = orig_bgr.copy()

    # Preenchimento base das classes
    for cid in (0, 2, 1):
        if np.any(masks[cid]):
            overlay[masks[cid] > 0] = COLORS_BGR[cid]

    # Destaque de interseção crítica em Magenta/Vermelho brilhante
    if np.any(inter_direct):
        overlay[inter_direct] = (255, 0, 200)

    cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)

    # Contornos nítidos
    for cid in (0, 2, 1):
        cnts, _ = cv2.findContours(masks[cid], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    print("EXECUÇÃO DO MODELO OURO 1024PX — ANÁLISE MULTI-PERSPECTIVA")
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    trees = manifest.get("trees", [])

    model = load_model(device)

    # Filtrar árvores para processar: Alto da Mooca e Vila Mariana (108 árvores)
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

            masks = {
                0: (probs[0] >= 0.50).astype(np.uint8),
                1: (probs[1] >= 0.38).astype(np.uint8),
                2: (probs[2] >= 0.48).astype(np.uint8),
            }

            triage, px_direct, px_wind, inter_direct, inter_wind = analyze_conflict_morphology(masks[0], masks[1])
            stats_triage[triage] += 1

            ov_name = f"ov_gold_{img_file}"
            ov_path = GOLD_OVERLAYS_DIR / ov_name
            title = f"{triage} | Copa: {int(np.count_nonzero(masks[0]))} px | Fio: {int(np.count_nonzero(masks[1]))} px | Contato: {px_direct} px"
            canvas = render_gold_overlay(orig_bgr, masks, inter_direct, triage, title)
            cv2.imwrite(str(ov_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            v["gold_analysis"] = {
                "triage": triage,
                "color_hex": STATUS_COLORS_HEX[triage],
                "copa_px": int(np.count_nonzero(masks[0])),
                "wire_px": int(np.count_nonzero(masks[1])),
                "pole_px": int(np.count_nonzero(masks[2])),
                "contact_direct_px": px_direct,
                "wind_buffer_px": px_wind,
                "gold_overlay_file": ov_name,
            }
            views_results.append((v.get("perspective"), triage, STATUS_COLORS_HEX[triage]))
            processed_views += 1

        # Consolidação da Árvore
        st_front = next((r[1] for r in views_results if r[0] == "center_front"), "SEGURO")
        st_side = next((r[1] for r in views_results if r[0] == "side_angle_A"), "SEGURO")
        col_front = STATUS_COLORS_HEX[st_front]
        col_side = STATUS_COLORS_HEX[st_side]

        # Consenso
        if st_front == "CRITICO" and st_side == "CRITICO":
            cons_triage = "CRITICO_CONFIRMADO"
            stats_consensus["AMBOS_CRITICO"] += 1
        elif (st_front == "CRITICO" and st_side == "SEGURO") or (st_front == "SEGURO" and st_side == "CRITICO"):
            cons_triage = "DIVERGENCIA_PARALAXE"
            stats_consensus["DIVERGENCIA_PARALAXE"] += 1
        elif st_front == "CRITICO" or st_side == "CRITICO":
            cons_triage = "CRITICO_MONITORADO"
            stats_consensus["AMBOS_CRITICO"] += 1
        elif st_front == "PROXIMIDADE" or st_side == "PROXIMIDADE":
            cons_triage = "PROXIMIDADE"
            stats_consensus["PROXIMIDADE"] += 1
        else:
            cons_triage = "SEGURO"
            stats_consensus["AMBOS_SEGURO"] += 1

        tree["gold_triage"] = {
            "front_triage": st_front,
            "front_color": col_front,
            "side_triage": st_side,
            "side_color": col_side,
            "consolidated": cons_triage,
            "has_paralax_divergence": (cons_triage == "DIVERGENCIA_PARALAXE"),
        }

        if t_idx % 25 == 0 or t_idx == len(target_trees):
            el = time.time() - t0_global
            print(f"[{t_idx:03d}/{len(target_trees)}] Árvores processadas ({processed_views} fotos) em {el:.1f}s ({el/max(processed_views,1):.2f}s/foto)")

    # Salva manifest atualizado
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest atualizado com sucesso em: {MANIFEST_PATH}")
    print(f"Estatísticas de Visadas Individuais: {stats_triage}")
    print(f"Estatísticas de Consenso por Árvore: {stats_consensus}")

    # Atualiza o Mapa HTML
    update_html_map(manifest)


def update_html_map(manifest: dict):
    """
    Gera o novo mapa HTML com suporte a marcadores bicolores (duas metades) via SVG.
    """
    map_path = SP_DIR / "mapa_arborgrid_sp.html"
    trees_js = json.dumps(manifest.get("trees", []), ensure_ascii=False)
    regions_js = json.dumps(manifest.get("regions_catalog", {}), ensure_ascii=False)

    html_code = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>ArborGrid SP — Mapa Multi-Perspectiva com Marcadores Bicolores (SegFormer 1024px)</title>
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
  </style>
</head>
<body>
  <div id="topbar">
    <div class="brand">ArborGrid SP — Auditoria Georreferenciada Multi-Perspectiva</div>
    <div class="region-tabs" id="regionTabs"></div>
    <div class="controls-right">
      <select id="presetSelect" class="preset-select" title="Modo de visualização dos modelos">
        <option value="gold_segformer_bicolor">★ Novo Modo: SegFormer 1024px Ouro (Marcador Bicolor: Frontal | Lateral)</option>
        <option value="all">Preset Clássico: Modelo Unificado</option>
      </select>
      <div class="legend">
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="#ef4444" stroke="#fff"/></svg> Crítico (Contato)</span>
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="#f59e0b" stroke="#fff"/></svg> Proximidade (Vento)</span>
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="#10b981" stroke="#fff"/></svg> Seguro</span>
        <span><svg class="split-legend-icon" viewBox="0 0 20 20"><path d="M 10 2 A 8 8 0 0 0 10 18 Z" fill="#ef4444"/><path d="M 10 2 A 8 8 0 0 1 10 18 Z" fill="#10b981"/><circle cx="10" cy="10" r="8" fill="none" stroke="#fff"/></svg> Bicolor (Paralaxe / Divergência)</span>
      </div>
    </div>
  </div>

  <div id="statsbar">
    <div>Região: <b id="statRegion">Todas</b></div>
    <div>Total Árvores: <b id="statTotal">0</b></div>
    <div>Crítico Confirmado (Ambos): <b id="statCritConfirm" style="color:#ef4444;">0</b></div>
    <div>Divergência de Paralaxe (1 Visada): <b id="statParalax" style="color:#f97316;">0</b></div>
    <div>Proximidade Monitorada: <b id="statProx" style="color:#f59e0b;">0</b></div>
    <div>Seguro: <b id="statSafe" style="color:#10b981;">0</b></div>
  </div>

  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const trees = {trees_js};
    const regionsCatalog = {regions_js};

    const map = L.map('map', {{ zoomControl: true }}).setView([-23.5615, -46.6870], 14);

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
    const osm = L.tileLayer(
      'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
      {{ attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }}
    );

    esriTopo.addTo(map);

    L.control.layers(
      {{
        'Mapa Topo (Esri)': esriTopo,
        'Mapa de Ruas (Esri Street)': esriStreet,
        'Satélite HD (Esri Imagery)': esriSat,
        'OpenStreetMap': osm
      }},
      null,
      {{ position: 'topright' }}
    ).addTo(map);

    const auditedLayer = L.layerGroup().addTo(map);

    let currentRegion = 'ALL';
    let currentPreset = 'gold_segformer_bicolor';

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
        const gt = t.gold_triage || {{}};
        const colFront = gt.front_color || '#10b981';
        const colSide  = gt.side_color || '#10b981';
        const isParalax = Boolean(gt.has_paralax_divergence);

        const icon = createSplitMarkerIcon(colFront, colSide, isParalax);
        const marker = L.marker([t.tree_lat, t.tree_lon], {{ icon: icon }});

        let viewsHtml = '';
        (t.views || []).forEach(v => {{
          const ga = v.gold_analysis || v.gpu_analysis || {{}};
          const ovPath = ga.gold_overlay_file ? ('gold_overlays/' + ga.gold_overlay_file) : ('gpu_overlays/' + ga.overlay_file);
          const rawPath = 'images/' + v.file_name;
          const triageCol = ga.color_hex || '#475569';
          viewsHtml += `
            <div class="view-box">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <b>${{v.perspective}}</b>
                <span class="badge" style="background:${{triageCol}}">${{ga.triage || 'SEGURO'}}</span>
              </div>
              <div style="font-size:11px;color:#64748b;">${{v.distance_m}}m da câmera · fov=${{v.fov}}°</div>
              <img src="${{ovPath}}"
                   onmouseover="this.src='${{rawPath}}'" onmouseout="this.src='${{ovPath}}'"
                   title="Passe o mouse para comparar com a foto original" />
              <div style="color:#334155;font-size:11px;">
                Copa: ${{ga.copa_px || ga.copa_pct || 0}} px | Fio: ${{ga.wire_px || 0}} px | <b>Contato: ${{ga.contact_direct_px || ga.conflict_overlap_px || 0}} px</b>
              </div>
              <a class="gmaps-btn" href="${{v.google_maps_url}}" target="_blank" rel="noopener">
                Ver no Google Maps 360° HD
              </a>
            </div>`;
        }});

        const consLabel = gt.consolidated || t.consolidated_triage || 'SEGURO';
        let badgeBg = '#10b981';
        if (consLabel.includes('CRITICO')) badgeBg = '#ef4444';
        else if (consLabel === 'DIVERGENCIA_PARALAXE') badgeBg = '#f97316';
        else if (consLabel === 'PROXIMIDADE') badgeBg = '#f59e0b';

        const popupHtml = `
          <div class="popup-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
              <b>Árvore GeoSampa ID #${{t.geosampa_id}} (${{t.region_label || t.distrito || 'SP'}})</b>
              <span class="badge" style="background:${{badgeBg}}">${{consLabel}}</span>
            </div>
            <div style="font-size:12px;color:#475569;margin-top:3px;">
              Coord: ${{t.tree_lat.toFixed(6)}}, ${{t.tree_lon.toFixed(6)}} · Última Vistoria SISGAU: <b>${{t.sisgau_last_inspection || 'Nenhuma (NULL)'}}</b>
            </div>
            <div style="font-size:11px;color:#0284c7;margin-top:2px;">
              Marcador Bicolor: Metade Esquerda = Frontal (${{gt.front_triage || 'SEGURO'}}) | Metade Direita = Lateral (${{gt.side_triage || 'SEGURO'}})
            </div>
            <div class="views-grid">${{viewsHtml}}</div>
          </div>`;

        marker.bindPopup(popupHtml, {{ maxWidth: 660 }});
        auditedLayer.addLayer(marker);
      }});

      updateStatsBar(filtered);
    }}

    function updateStatsBar(filtered) {{
      document.getElementById('statRegion').innerText = currentRegion === 'ALL' ? 'Todas as Regiões' : (regionsCatalog[currentRegion]?.label || currentRegion);
      document.getElementById('statTotal').innerText = filtered.length;

      let crit = 0, paralax = 0, prox = 0, safe = 0;
      filtered.forEach(t => {{
        const c = t.gold_triage?.consolidated || 'SEGURO';
        if (c === 'CRITICO_CONFIRMADO' || c === 'CRITICO_MONITORADO') crit++;
        else if (c === 'DIVERGENCIA_PARALAXE') paralax++;
        else if (c === 'PROXIMIDADE') prox++;
        else safe++;
      }});

      document.getElementById('statCritConfirm').innerText = crit;
      document.getElementById('statParalax').innerText = paralax;
      document.getElementById('statProx').innerText = prox;
      document.getElementById('statSafe').innerText = safe;
    }}

    // Construção das Tabs de Região
    const tabsContainer = document.getElementById('regionTabs');
    const allBtn = document.createElement('button');
    allBtn.className = 'reg-btn active';
    allBtn.innerText = 'São Paulo (Todas)';
    allBtn.onclick = () => selectRegion('ALL', allBtn);
    tabsContainer.appendChild(allBtn);

    Object.keys(regionsCatalog).forEach(k => {{
      const btn = document.createElement('button');
      btn.className = 'reg-btn';
      btn.innerText = regionsCatalog[k].label || k;
      btn.onclick = () => selectRegion(k, btn);
      tabsContainer.appendChild(btn);
    }});

    function selectRegion(regKey, btnEl) {{
      currentRegion = regKey;
      document.querySelectorAll('.reg-btn').forEach(b => b.classList.remove('active'));
      btnEl.classList.add('active');

      if (regKey !== 'ALL' && regionsCatalog[regKey]?.bbox) {{
        const b = regionsCatalog[regKey].bbox;
        map.fitBounds([[b[1], b[0]], [b[3], b[2]]]);
      }} else {{
        map.setView([-23.5615, -46.6870], 13);
      }}
      renderMarkers();
    }}

    renderMarkers();
  </script>
</body>
</html>
"""
    map_path.write_text(html_code, encoding="utf-8")
    print(f"Mapa interativo HTML atualizado com sucesso em: {map_path}")


if __name__ == "__main__":
    main()

