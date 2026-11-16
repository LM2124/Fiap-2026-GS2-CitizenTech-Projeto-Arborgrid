"""
ArborGrid - Pré-Labeling GPU Incremental (SegFormer-1024) + Morfologia OpenCV + Mapa Multi-Região
Módulo: arborgrid-vision / scripts / auto_prelabel_and_audit.py

Destaques desta versão:
  - Processamento incremental em GPU (só processa imagens novas adicionadas ao manifesto, salvo `--recompute`).
  - Mapa Interativo Multi-Região (Pinheiros, Alto da Mooca e futuras regiões) com botões de navegação rápida (`flyToBounds`).
  - Modo de Visualização do Cadastro Governamental Completo (`GeoSampa`): renderiza milhares de árvores não auditadas
    em alta performance via HTML5 Canvas (`L.canvas()`) como círculos vazados brancos/cinza clicáveis.
  - Seletor de mapas-base Esri sem necessidade de chave de API (Viário Limpo, Ruas e Satélite Aéreo HD).
"""

import json
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"
IMAGES_DIR = DATA_DIR / "images"
PRELABELS_DIR = DATA_DIR / "prelabels_yolo_seg"
OVERLAYS_DIR = DATA_DIR / "gpu_overlays"
CADASTRAL_DIR = DATA_DIR / "geosampa_cadastral"
LABELING_WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"

CANONICAL_CLASSES = ["copa", "fiacao", "poste"]
SEGFORMER_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


def binary_mask_to_yolo_polygons(
    mask: np.ndarray,
    cls_id: int,
    min_area_px: float = 400.0,
    approx_eps_ratio: float = 0.004,
) -> list[str]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_eps_ratio * peri, True)
        if len(approx) < 3:
            continue
        coords = []
        for pt in approx[:, 0, :]:
            nx = min(1.0, max(0.0, float(pt[0]) / w))
            ny = min(1.0, max(0.0, float(pt[1]) / h))
            coords.append(f"{nx:.6f} {ny:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))
    return lines


def detect_overhead_wires_contextual(img_bgr: np.ndarray, seg_map: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)

    forbidden = np.isin(seg_map, [0, 1, 3, 4, 11, 12, 13, 14, 15]).astype(np.uint8) * 255
    forbidden = cv2.dilate(forbidden, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    sky_or_pole = np.isin(seg_map, [5, 10]).astype(np.uint8) * 255
    sky_or_pole_near = cv2.dilate(sky_or_pole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    wire_mask = np.zeros((h, w), dtype=np.uint8)
    lsd = cv2.createLineSegmentDetector(0)
    det_res = lsd.detect(gray)
    lines = det_res[0] if det_res is not None else None
    if lines is None:
        return wire_mask

    for ln in lines:
        x1, y1, x2, y2 = ln.reshape(-1)[:4]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 52.0:
            continue

        angle_deg = float(np.degrees(np.arctan2(dy, max(dx, 1e-5))))
        if angle_deg > 68.0:
            continue
        if max(y1, y2) > h * 0.78:
            continue

        num_pts = max(10, int(length // 5))
        xs = np.clip(np.linspace(x1, x2, num_pts).astype(int), 0, w - 1)
        ys = np.clip(np.linspace(y1, y2, num_pts).astype(int), 0, h - 1)

        if np.mean(forbidden[ys, xs] > 0) > 0.12:
            continue

        inside_veg_ratio = float(np.mean(seg_map[ys, xs] == 8))
        if inside_veg_ratio > 0.45:
            continue

        nx_vec = -(y2 - y1) / max(length, 1e-5)
        ny_vec = (x2 - x1) / max(length, 1e-5)
        offset_px = 5.0
        xs_pos = np.clip(np.round(xs + nx_vec * offset_px).astype(int), 0, w - 1)
        ys_pos = np.clip(np.round(ys + ny_vec * offset_px).astype(int), 0, h - 1)
        xs_neg = np.clip(np.round(xs - nx_vec * offset_px).astype(int), 0, w - 1)
        ys_neg = np.clip(np.round(ys - ny_vec * offset_px).astype(int), 0, h - 1)

        i_center = gray[ys, xs].astype(np.float32)
        i_pos = gray[ys_pos, xs_pos].astype(np.float32)
        i_neg = gray[ys_neg, xs_neg].astype(np.float32)
        valley_contrast = np.minimum(i_pos, i_neg) - i_center

        if float(np.median(valley_contrast)) < 6.5:
            continue

        on_building_ratio = float(np.mean(seg_map[ys, xs] == 2))
        near_sky_pole_ratio = float(np.mean(sky_or_pole_near[ys, xs] > 0))
        bh_strength = float(np.mean(blackhat[ys, xs]))

        if on_building_ratio > 0.50 and near_sky_pole_ratio < 0.30:
            continue
        if bh_strength < 9.0 and near_sky_pole_ratio < 0.35:
            continue

        cv2.line(wire_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=6)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, close_k)
    return wire_mask


def extract_clean_canopy_mask(seg_map: np.ndarray) -> np.ndarray:
    h, w = seg_map.shape[:2]
    veg = (seg_map == 8).astype(np.uint8) * 255
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    clean = np.zeros_like(veg)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(veg)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        top_y = stats[i, cv2.CC_STAT_TOP]
        cy = centroids[i][1]
        if area < 900:
            continue
        if top_y > h * 0.68 and cy > h * 0.78:
            continue
        clean[labels == i] = 255
    return clean


def process_single_view(
    img_path: Path,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: str,
    out_overlay_path: Path,
    out_label_path: Path,
) -> dict:
    img_pil = Image.open(img_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    seg_map = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    mask_copa = extract_clean_canopy_mask(seg_map)
    mask_poste = ((seg_map == 5).astype(np.uint8)) * 255
    mask_fiacao = detect_overhead_wires_contextual(img_bgr, seg_map)

    yolo_lines = []
    yolo_lines.extend(binary_mask_to_yolo_polygons(mask_copa, cls_id=0, min_area_px=1200.0, approx_eps_ratio=0.004))
    yolo_lines.extend(binary_mask_to_yolo_polygons(mask_fiacao, cls_id=1, min_area_px=220.0, approx_eps_ratio=0.006))
    yolo_lines.extend(binary_mask_to_yolo_polygons(mask_poste, cls_id=2, min_area_px=500.0, approx_eps_ratio=0.005))

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    k_crit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    k_warn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39))

    wires_crit = cv2.dilate(mask_fiacao, k_crit, iterations=1)
    wires_warn = cv2.dilate(mask_fiacao, k_warn, iterations=1)

    inter_crit = cv2.bitwise_and(wires_crit, mask_copa)
    inter_warn = cv2.bitwise_and(wires_warn, mask_copa)

    crit_px = int(np.count_nonzero(inter_crit))
    warn_px = int(np.count_nonzero(inter_warn))
    copa_px = int(np.count_nonzero(mask_copa))
    fio_px = int(np.count_nonzero(mask_fiacao))
    poste_px = int(np.count_nonzero(mask_poste))

    if crit_px >= 260 and fio_px >= 280:
        triage = "CANDIDATO_CONFLITO"
    elif warn_px >= 500 and fio_px >= 250:
        triage = "CANDIDATO_PROXIMO"
    else:
        triage = "PROVAVEL_SEGURO"

    vis = img_bgr.copy()
    overlay = np.zeros_like(vis)
    overlay[mask_copa > 0] = (55, 195, 55)
    overlay[mask_poste > 0] = (235, 150, 35)
    overlay[mask_fiacao > 0] = (255, 255, 0)
    overlay[inter_crit > 0] = (30, 30, 245)

    active = (mask_copa > 0) | (mask_poste > 0) | (mask_fiacao > 0) | (inter_crit > 0)
    vis[active] = cv2.addWeighted(vis[active], 0.56, overlay[active], 0.44, 0)

    contours, _ = cv2.findContours(inter_crit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 230, 255), 2)

    out_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_overlay_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 91])

    return {
        "triage": triage,
        "copa_pct": round(copa_px / (h * w) * 100, 1),
        "wire_px": fio_px,
        "pole_px": poste_px,
        "conflict_overlap_px": crit_px,
        "yolo_polygons_count": len(yolo_lines),
        "overlay_file": out_overlay_path.name,
    }


def load_unaudited_cadastral_points(manifest: dict) -> list[list]:
    """
    Lê todos os arquivos `geosampa_<region>.geojson` baixados e retorna lista enxuta
    `[lat, lon, id, region_key, has_sisgau_inspection]` das árvores governamentais NÃO auditadas,
    permitindo desenhar milhares de pontos vazados brancos/cinza via HTML5 Canvas sem travar o navegador.
    """
    audited_ids = {str(t["geosampa_id"]) for t in manifest.get("trees", [])}
    points = []
    if not CADASTRAL_DIR.exists():
        return points

    for gj_path in sorted(CADASTRAL_DIR.glob("geosampa_*.geojson")):
        reg_key = gj_path.stem.replace("geosampa_", "")
        data = json.loads(gj_path.read_text(encoding="utf-8"))
        for feat in data.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            props = feat.get("properties") or {}
            tid = str(props.get("cd_identificador", ""))
            if tid in audited_ids:
                continue
            lon, lat = geom["coordinates"][:2]
            insp = 1 if props.get("dt_inspecao") else 0
            points.append([round(lat, 6), round(lon, 6), tid, reg_key, insp])
    return points


def build_interactive_leaflet_map(manifest: dict, out_html_paths: list[Path]) -> None:
    """Gera Mapa Multi-Região com Seletor de Bairros, Cadastro Completo GeoSampa (Canvas) e Street View 360 HD."""
    unaudited_pts = load_unaudited_cadastral_points(manifest)
    regions_cat = manifest.get("regions_catalog") or {
        "pinheiros": {
            "label": "Pinheiros (Zona Oeste)",
            "bbox": [-46.6960, -23.5680, -46.6790, -23.5550],
        }
    }

    trees_js = json.dumps(manifest.get("trees", []), ensure_ascii=False)
    regions_js = json.dumps(regions_cat, ensure_ascii=False)
    cadastral_js = json.dumps(unaudited_pts, ensure_ascii=False, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>ArborGrid — Mapa Multi-Região de Auditoria (São Paulo · GeoSampa + Street View)</title>
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
      background: #0f172a; color: #f8fafc; border: 1px solid #3b82f6; border-radius: 6px;
      padding: 5px 9px; font-size: 12px; font-weight: 600; cursor: pointer;
    }}
    .toggle-pill {{
      display: inline-flex; align-items: center; gap: 6px; background: #0f172a;
      border: 1px solid #475569; padding: 4px 10px; border-radius: 6px; cursor: pointer; user-select: none;
    }}
    .legend {{ display: flex; gap: 11px; align-items: center; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
    .dot-hollow {{
      display: inline-block; width: 10px; height: 10px; border-radius: 50%;
      background: #ffffff; border: 2px solid #64748b; margin-right: 4px;
    }}
    #statsbar {{
      display: flex; gap: 18px; padding: 6px 16px; background: #0f172a;
      border-bottom: 1px solid #1e293b; font-size: 12px; color: #cbd5e1; flex-wrap: wrap;
    }}
    #statsbar b {{ color: #f8fafc; }}
    #map {{ height: calc(100vh - 84px); width: 100%; }}
    .popup-card {{ color: #0f172a; min-width: 520px; max-width: 630px; }}
    .views-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }}
    .view-box {{ background: #f1f5f9; padding: 7px; border-radius: 6px; font-size: 12px; border: 1px solid #cbd5e1; }}
    .view-box img {{ width: 100%; border-radius: 4px; display: block; margin: 5px 0; cursor: pointer; }}
    .gmaps-btn {{
      display: inline-block; margin-top: 5px; padding: 5px 9px; background: #2563eb;
      color: #fff !important; text-decoration: none; border-radius: 4px; font-weight: 600; font-size: 11px;
    }}
    .gmaps-btn:hover {{ background: #1d4ed8; }}
    .badge {{ padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; color: #fff; }}
  </style>
</head>
<body>
  <div id="topbar">
    <div class="brand">🌳⚡ ArborGrid SP — Auditoria Georreferenciada</div>
    <div class="region-tabs" id="regionTabs"></div>
    <div class="controls-right">
      <select id="presetSelect" class="preset-select" title="Alterna entre o lote de treino curado, os bairros somente de inferência e o preset pós-treino de cobertura total">
        <option value="all">🔬 Preset: Todas Auditadas (Treino + Inferência)</option>
        <option value="train_curated">🏷️ Preset: Apenas Lote de Labeling/Treino (Pinheiros)</option>
        <option value="inference_only">🚀 Preset: Apenas Inferência (Sem Labeling Manual)</option>
        <option value="trained_yolo_full">🤖 Preset: Modelo YOLOv8-Seg Treinado (Cobertura Total)</option>
      </select>
      <label class="toggle-pill" title="Desenha todas as árvores registradas pela Prefeitura no GeoSampa como círculos brancos/cinza">
        <input type="checkbox" id="chkCadastral" checked />
        <span class="dot-hollow"></span>
        <span id="lblCadastralCount">Cadastro GeoSampa Completo</span>
      </label>
      <div class="legend">
        <span><i class="dot" style="background:#ef4444"></i>Conflito</span>
        <span><i class="dot" style="background:#f59e0b"></i>Proximidade</span>
        <span><i class="dot" style="background:#22c55e"></i>Seguro</span>
      </div>
    </div>
  </div>
  <div id="statsbar">Carregando indicadores...</div>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const trees = {trees_js};
    const regionsCatalog = {regions_js};
    const cadastralPts = {cadastral_js};

    const canvasRenderer = L.canvas({{ padding: 0.5 }});
    const map = L.map('map', {{ preferCanvas: true }}).setView([-23.5615, -46.6875], 15);

    const esriTopo = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Tiles &copy; Esri | Árvores: GeoSampa/PMSP | Street View: &copy; Google', maxZoom: 19 }}
    );
    const esriStreet = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Tiles &copy; Esri | Árvores: GeoSampa/PMSP | Street View: &copy; Google', maxZoom: 19 }}
    );
    const esriSat = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Satélite &copy; Esri, Maxar | Árvores: GeoSampa/PMSP | Street View: &copy; Google', maxZoom: 19 }}
    );

    esriTopo.addTo(map);
    L.control.layers(
      {{
        'Mapa Viário Limpo (Esri Topo)': esriTopo,
        'Mapa de Ruas (Esri Street)': esriStreet,
        'Satélite Aéreo HD (Esri Imagery)': esriSat
      }},
      null,
      {{ position: 'topright' }}
    ).addTo(map);

    const cadastralLayer = L.layerGroup().addTo(map);
    const auditedLayer = L.layerGroup().addTo(map);

    document.getElementById('lblCadastralCount').textContent =
      `Cadastro GeoSampa (${{cadastralPts.length.toLocaleString('pt-BR')}} não auditadas)`;

    // 1. Camada de fundo: Todas as árvores do governo (pontos vazados brancos/cinza em Canvas rápido)
    cadastralPts.forEach(p => {{
      const [lat, lon, tid, rkey, insp] = p;
      const m = L.circleMarker([lat, lon], {{
        renderer: canvasRenderer,
        radius: 3.6,
        fillColor: '#ffffff',
        color: '#64748b',
        weight: 1.4,
        fillOpacity: 0.78
      }});
      m.on('click', () => {{
        const svUrl = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${{lat}},${{lon}}`;
        m.bindPopup(`
          <div style="color:#0f172a;font-size:12px;min-width:230px;">
            <b>⚪ Árvore Oficial GeoSampa #${{tid}}</b><br/>
            <span style="color:#475569">Status IA: Ainda não amostrada</span><br/>
            <span style="color:#475569">Vistoria SISGAU: <b>${{insp ? 'Registrada' : 'Nenhuma (NULL)'}}</b></span><br/>
            <a class="gmaps-btn" href="${{svUrl}}" target="_blank" rel="noopener" style="margin-top:7px;">
              🌍 Abrir este ponto no Street View
            </a>
          </div>
        `).openPopup();
      }});
      cadastralLayer.addLayer(m);
    }});

    document.getElementById('chkCadastral').addEventListener('change', e => {{
      if (e.target.checked) map.addLayer(cadastralLayer);
      else map.removeLayer(cadastralLayer);
    }});

    // 2. Camada principal: Árvores auditadas pela GPU (filtradas dinamicamente pelo Preset selecionado)
    function colorForStatus(status) {{
      if (status === 'CANDIDATO_CONFLITO') return '#ef4444';
      if (status === 'CANDIDATO_PROXIMO') return '#f59e0b';
      return '#22c55e';
    }}

    let currentRegion = 'ALL';
    let currentPreset = 'all';

    function filterTreesByPreset(list, preset) {{
      if (preset === 'train_curated') return list.filter(t => t.dataset_role === 'train_curated');
      if (preset === 'inference_only') return list.filter(t => t.dataset_role !== 'train_curated');
      if (preset === 'trained_yolo_full') return list.filter(t => Boolean(t.trained_yolo_triage));
      return list;
    }}

    function renderAuditedMarkers() {{
      auditedLayer.clearLayers();
      const visible = filterTreesByPreset(trees, currentPreset);
      visible.forEach(t => {{
        const st = currentPreset === 'trained_yolo_full'
          ? (t.trained_yolo_triage || 'PROVAVEL_SEGURO')
          : (t.consolidated_triage || 'PROVAVEL_SEGURO');
        const col = colorForStatus(st);
        const isTrain = t.dataset_role === 'train_curated';
        const marker = L.circleMarker([t.tree_lat, t.tree_lon], {{
          renderer: canvasRenderer,
          radius: 8,
          fillColor: col,
          color: isTrain ? '#0f172a' : '#1e3a8a',
          weight: 2,
          fillOpacity: 0.95
        }});

        let viewsHtml = '';
        (t.views || []).forEach(v => {{
          const ga = v.gpu_analysis || {{}};
          const ovPath = ga.overlay_file ? ('gpu_overlays/' + ga.overlay_file) : ('images/' + v.file_name);
          const rawPath = 'images/' + v.file_name;
          viewsHtml += `
            <div class="view-box">
              <div><b>${{v.perspective}}</b> (${{v.capture_date}}) · ${{v.distance_m}}m · fov=${{v.fov}}°</div>
              <img src="${{ovPath}}"
                   onmouseover="this.src='${{rawPath}}'" onmouseout="this.src='${{ovPath}}'" />
              <div style="color:#334155;font-size:11px;">
                Copa: ${{ga.copa_pct ?? 0}}% | Interseção: ${{ga.conflict_overlap_px ?? 0}} px
              </div>
              <a class="gmaps-btn" href="${{v.google_maps_url}}" target="_blank" rel="noopener">
                🌍 Ver em Alta Definição no Google Maps 360°
              </a>
            </div>`;
        }});

        const roleTag = isTrain
          ? '<span class="badge" style="background:#475569;margin-right:4px;">🏷️ LOTE TREINO</span>'
          : '<span class="badge" style="background:#2563eb;margin-right:4px;">🚀 SÓ INFERÊNCIA</span>';

        const html = `
          <div class="popup-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
              <b>Árvore GeoSampa ID #${{t.geosampa_id}} (${{t.region_label || t.distrito || 'SP'}})</b>
              <div>${{roleTag}}<span class="badge" style="background:${{col}}">${{st}}</span></div>
            </div>
            <div style="font-size:12px;color:#475569;margin-top:3px;">
              Coord: ${{t.tree_lat.toFixed(6)}}, ${{t.tree_lon.toFixed(6)}} · Vistoria SISGAU: <b>${{t.sisgau_last_inspection || 'Nenhuma (NULL)'}}</b>
            </div>
            <div class="views-grid">${{viewsHtml}}</div>
          </div>`;

        marker.bindPopup(html, {{ maxWidth: 660 }});
        auditedLayer.addLayer(marker);
      }});
    }}

    // 3. Seletor de Regiões + Preset + Barra de Estatísticas Dinâmica
    const tabsContainer = document.getElementById('regionTabs');
    const statsBar = document.getElementById('statsbar');
    const presetSelect = document.getElementById('presetSelect');
    const regionKeys = Object.keys(regionsCatalog);

    function updateStats() {{
      const regSubset = currentRegion === 'ALL'
        ? trees
        : trees.filter(t => (t.region_key || 'pinheiros') === currentRegion);
      const subset = filterTreesByPreset(regSubset, currentPreset);

      const getStatus = t => currentPreset === 'trained_yolo_full'
        ? t.trained_yolo_triage
        : t.consolidated_triage;

      const nConf = subset.filter(t => getStatus(t) === 'CANDIDATO_CONFLITO').length;
      const nProx = subset.filter(t => getStatus(t) === 'CANDIDATO_PROXIMO').length;
      const nSafe = subset.length - nConf - nProx;

      let totalGov = 0, totalInsp = 0;
      if (currentRegion === 'ALL') {{
        regionKeys.forEach(k => {{
          totalGov += regionsCatalog[k].total_geosampa_trees || 0;
          totalInsp += regionsCatalog[k].sisgau_inspected_trees || 0;
        }});
      }} else if (regionsCatalog[currentRegion]) {{
        totalGov = regionsCatalog[currentRegion].total_geosampa_trees || 0;
        totalInsp = regionsCatalog[currentRegion].sisgau_inspected_trees || 0;
      }}
      const nullPct = totalGov > 0 ? (((totalGov - totalInsp) / totalGov) * 100).toFixed(1) : '0.0';

      if (currentPreset === 'trained_yolo_full' && subset.length === 0) {{
        statsBar.innerHTML = `
          <span style="color:#fbbf24">🤖 <b>Preset Pós-Treino (Aguardando Pesos best.pt):</b> Assim que concluirmos o treino do YOLOv8s-Seg nas 139 imagens curadas, este preset exibirá a inferência automática sobre todas as <b>${{totalGov.toLocaleString('pt-BR')}}</b> árvores mapeadas no recorte!</span>
        `;
        return;
      }}

      statsBar.innerHTML = `
        <span>🌲 Cadastro Oficial GeoSampa: <b>${{totalGov.toLocaleString('pt-BR')}}</b> (<b>${{nullPct}}%</b> sem vistoria SISGAU)</span>
        <span>🔍 Exibidas neste Preset: <b>${{subset.length}}</b></span>
        <span style="color:#f87171">🔴 Conflito: <b>${{nConf}}</b></span>
        <span style="color:#fbbf24">🟡 Proximidade: <b>${{nProx}}</b></span>
        <span style="color:#4ade80">🟢 Seguro: <b>${{nSafe}}</b></span>
        <span style="color:#94a3b8">💡 Passe o mouse nas fotos: Original ↔ Segmentação</span>
      `;
    }}

    presetSelect.addEventListener('change', e => {{
      currentPreset = e.target.value;
      renderAuditedMarkers();
      updateStats();
    }});

    function selectRegion(rkey) {{
      currentRegion = rkey;
      document.querySelectorAll('.reg-btn').forEach(b => b.classList.toggle('active', b.dataset.region === rkey));
      if (rkey === 'ALL') {{
        const allLatLngs = trees.map(t => [t.tree_lat, t.tree_lon]);
        if (allLatLngs.length) map.flyToBounds(L.latLngBounds(allLatLngs), {{ padding: [30, 30], duration: 1.0 }});
      }} else if (regionsCatalog[rkey] && regionsCatalog[rkey].bbox) {{
        const [minLon, minLat, maxLon, maxLat] = regionsCatalog[rkey].bbox;
        map.flyToBounds([[minLat, minLon], [maxLat, maxLon]], {{ padding: [20, 20], duration: 1.0 }});
      }}
      updateStats();
    }}

    renderAuditedMarkers();

    regionKeys.forEach((rk, i) => {{
      const btn = document.createElement('button');
      btn.className = 'reg-btn' + (i === 0 ? ' active' : '');
      btn.dataset.region = rk;
      btn.textContent = '📍 ' + (regionsCatalog[rk].label || rk);
      btn.onclick = () => selectRegion(rk);
      tabsContainer.appendChild(btn);
    }});

    if (regionKeys.length > 1) {{
      const btnAll = document.createElement('button');
      btnAll.className = 'reg-btn';
      btnAll.dataset.region = 'ALL';
      btnAll.textContent = '🗺️ Ver Todas (' + regionKeys.length + ' regiões)';
      btnAll.onclick = () => selectRegion('ALL');
      tabsContainer.appendChild(btnAll);
    }}

    if (regionKeys.length > 0) selectRegion(regionKeys[0]);
  </script>
</body>
</html>
"""
    for p in out_html_paths:
        p.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Pré-Labeling GPU Incremental + Mapa Multi-Região")
    parser.add_argument("--recompute", action="store_true", help="Reprocessa na GPU mesmo imagens já analisadas")
    parser.add_argument("--map-only", action="store_true", help="Apenas regenera o Mapa HTML sem rodar GPU")
    args = parser.parse_args()

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_maps = [DATA_DIR / "mapa_arborgrid_sp.html"]

    # Marca explicitamente quais árvores pertencem ao lote congelado de labeling/treino (workspace_xanylabeling)
    # vs quais são exclusivas de inferência (zero trabalho adicional de labeling!)
    curated_files = {p.name for p in LABELING_WS_DIR.glob("*.jpg")} if LABELING_WS_DIR.exists() else set()
    for tree in manifest.get("trees", []):
        in_curated = any(v.get("file_name") in curated_files for v in tree.get("views", []))
        tree["dataset_role"] = "train_curated" if in_curated else "inference_only"

    if args.map_only:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        build_interactive_leaflet_map(manifest, out_maps)
        print("Mapas HTML regenerados com sucesso!")
        return

    # Descobrir quantas views precisam de GPU
    pending_views = []
    for tree in manifest["trees"]:
        for v in tree["views"]:
            ov_ok = v.get("gpu_analysis") and (OVERLAYS_DIR / v["gpu_analysis"].get("overlay_file", "")).exists()
            if args.recompute or not ov_ok:
                pending_views.append((tree, v))

    if pending_views:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("=" * 74)
        print(f"ARBORGRID | PROCESSANDO {len(pending_views)} NOVAS IMAGENS NA GPU ({device})")
        print("=" * 74)

        PRELABELS_DIR.mkdir(parents=True, exist_ok=True)
        (PRELABELS_DIR / "classes.txt").write_text("\n".join(CANONICAL_CLASSES) + "\n", encoding="utf-8")

        processor = SegformerImageProcessor.from_pretrained(SEGFORMER_ID)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID).to(device).eval()

        for i, (_, v) in enumerate(pending_views, start=1):
            img_path = IMAGES_DIR / v["file_name"]
            if not img_path.exists():
                continue
            lbl_path = PRELABELS_DIR / (img_path.stem + ".txt")
            ov_path = OVERLAYS_DIR / f"ov_{v['file_name']}"
            v["gpu_analysis"] = process_single_view(img_path, processor, model, device, ov_path, lbl_path)
            if i % 25 == 0 or i == len(pending_views):
                print(f"  -> GPU: {i}/{len(pending_views)} novas imagens concluídas...")

    # Atualizar consenso multi-perspectiva em todas as árvores
    summary_by_region: dict[str, dict[str, int]] = {}
    for tree in manifest["trees"]:
        rkey = tree.get("region_key", "pinheiros")
        view_statuses = [(v.get("gpu_analysis") or {}).get("triage", "PROVAVEL_SEGURO") for v in tree["views"]]
        n_conf = view_statuses.count("CANDIDATO_CONFLITO")
        n_prox = view_statuses.count("CANDIDATO_PROXIMO")

        if n_conf >= 2 or (n_conf == 1 and (n_prox >= 1 or len(view_statuses) == 1)):
            tree["consolidated_triage"] = "CANDIDATO_CONFLITO"
        elif n_conf >= 1 or n_prox >= 1:
            tree["consolidated_triage"] = "CANDIDATO_PROXIMO"
        else:
            tree["consolidated_triage"] = "PROVAVEL_SEGURO"

        reg_stats = summary_by_region.setdefault(
            rkey, {"CANDIDATO_CONFLITO": 0, "CANDIDATO_PROXIMO": 0, "PROVAVEL_SEGURO": 0}
        )
        reg_stats[tree["consolidated_triage"]] += 1

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_interactive_leaflet_map(manifest, out_maps)

    print("\n" + "=" * 74)
    print("RESUMO CONSOLIDADO POR REGIÃO:")
    for rk, st in summary_by_region.items():
        print(f"  [{rk}] Conflito={st['CANDIDATO_CONFLITO']} | Proximidade={st['CANDIDATO_PROXIMO']} | Seguro={st['PROVAVEL_SEGURO']}")
    print(f"Mapas atualizados: {out_maps[0]} e {out_maps[1]}")
    print("=" * 74)


if __name__ == "__main__":
    main()
