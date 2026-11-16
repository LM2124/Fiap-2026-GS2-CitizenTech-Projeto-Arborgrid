"""
ArborGrid SP — Gerador do Mapa Interativo Georreferenciado
Módulo: arborgrid-vision / scripts / generate_interactive_map.py

Gera o mapa analítico HTML com:
- Suporte ao Modelo V2 Full 1024px e marcadores bicolores de paralaxe
- Painel Lateral Retrátil com Seletor de Territórios e Diagnósticos Ricos
- Camadas Cartográficas Esri Corporativas (100% livres de autenticação e bloqueios)
- Polígono de Corredor Expandido da Av. Paulista (cobrindo Al. Santos e Rua São Carlos do Pinhal)
- Delimitações vetoriais limpas (sem interferência no cursor do mouse)
"""

import json
from pathlib import Path

CV_ROOT = Path(__file__).resolve().parent.parent
SP_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"
MANIFEST_PATH = SP_DIR / "manifest.json"
MAP_PATH = SP_DIR / "mapa_arborgrid_sp.html"


def generate_html_map(manifest: dict) -> Path:
    trees_js = json.dumps(manifest.get("trees", []), ensure_ascii=False)
    regions_js = json.dumps(manifest.get("regions_catalog", {}), ensure_ascii=False)

    html_code = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>ArborGrid SP — Auditoria Georreferenciada Multi-Perspectiva (SegFormer 1024px V3 Modelo C)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; overflow: hidden; }}
    
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

    /* Layout Estrutural: Painel Lateral + Mapa */
    #content-wrapper {{
      display: flex;
      height: calc(100vh - 84px);
      width: 100%;
      position: relative;
      overflow: hidden;
    }}
    .sidebar {{
      width: 350px;
      min-width: 350px;
      background: #0f172a;
      border-right: 1px solid #1e293b;
      display: flex;
      flex-direction: column;
      z-index: 500;
      transition: margin-left 0.22s ease;
      overflow: hidden;
    }}
    .sidebar.collapsed {{
      margin-left: -350px;
    }}
    .sidebar-header {{
      padding: 12px 14px;
      border-bottom: 1px solid #1e293b;
      background: #111827;
      flex-shrink: 0;
    }}
    .sidebar-region-cards {{
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      border-bottom: 1px solid #1e293b;
      flex-shrink: 0;
    }}
    .s-region-card {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 10px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 6px;
      color: #e2e8f0;
      cursor: pointer;
      transition: 0.15s;
      font-size: 12px;
      font-weight: 600;
    }}
    .s-region-card:hover {{
      background: #334155;
      border-color: #64748b;
    }}
    .s-region-card.active {{
      background: #1e3a8a;
      border-color: #3b82f6;
      color: #ffffff;
      box-shadow: 0 0 0 1px #3b82f6;
    }}
    .sidebar-detail-panel {{
      padding: 14px;
      flex: 1;
      overflow-y: auto;
      background: #090d16;
      font-size: 12px;
      line-height: 1.45;
    }}
    .s-metric-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin: 10px 0;
    }}
    .s-metric-box {{
      background: #1e293b;
      padding: 7px 9px;
      border-radius: 6px;
      border: 1px solid #334155;
    }}
    .s-metric-box span {{
      font-size: 10px;
      color: #94a3b8;
      display: block;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .s-metric-box b {{
      font-size: 15px;
      display: block;
      margin-top: 2px;
      color: #f8fafc;
    }}
    .grid-badge-pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}

    #map-container {{
      flex: 1;
      height: 100%;
      position: relative;
    }}
    #map {{
      height: 100%;
      width: 100%;
    }}

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

    .sidebar-toggle-btn {{
      display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px;
      border-radius: 6px; border: 1px solid #3b82f6; background: #1e3a8a;
      color: #ffffff; font-size: 12px; font-weight: 700; cursor: pointer; transition: 0.15s;
    }}
    .sidebar-toggle-btn:hover {{ background: #2563eb; }}
  </style>
</head>
<body>
  <div id="topbar">
    <div class="brand">
      <button id="toggleSidebarBtn" class="sidebar-toggle-btn" title="Exibir / Ocultar Painel Territorial de Regiões">
        <span id="sidebarToggleIcon">◀</span> Painel Territorial
      </button>
      <span>ArborGrid SP — Auditoria Georreferenciada</span>
    </div>
    <div class="region-tabs" id="regionTabs"></div>
    <div class="controls-right">
      <select id="presetSelect" class="preset-select" title="Modo de visualização dos modelos">
        <option value="v3_gold_bicolor" selected>★ Modo V3 Ouro (Modelo C — Hard Negatives + Augs)</option>
        <option value="v2_full_bicolor">Modo V2 Full (Dataset 132 Imgs + Morfologia Calibrada)</option>
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
    <div>Região: <b id="statRegion">Todas as Regiões</b></div>
    <div>Total Árvores: <b id="statTotal">0</b></div>
    <div>Crítico Confirmado (Ambos): <b id="statCritConfirm" style="color:#ef4444;">0</b></div>
    <div>Divergência de Paralaxe: <b id="statParalax" style="color:#f97316;">0</b></div>
    <div>Mono-Visada: <b id="statMono" style="color:#94a3b8;">0</b></div>
    <div>Proximidade Monitorada: <b id="statProx" style="color:#f59e0b;">0</b></div>
    <div>Seguro: <b id="statSafe" style="color:#10b981;">0</b></div>
  </div>

  <div id="content-wrapper">
    <div id="sidebar" class="sidebar">
      <div class="sidebar-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <b style="font-size:13px; color:#f8fafc;">Territórios Auditados</b>
          <span class="badge" style="background:#2563eb; font-size:10px;">4 Setores</span>
        </div>
        <div style="font-size:11px; color:#94a3b8; margin-top:3px;">
          Selecione uma área para filtrar o mapa e inspecionar os perfis cadastrais e diagnósticos de rede.
        </div>
      </div>
      <div class="sidebar-region-cards" id="sidebarRegionCards"></div>
      <div class="sidebar-detail-panel" id="sidebarDetailPanel"></div>
    </div>
    <div id="map-container">
      <div id="map"></div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const trees = {trees_js};
    const regionsCatalog = {regions_js};

    const map = L.map('map', {{ zoomControl: true }}).setView([-23.5650, -46.6450], 13);

    // Camadas Cartográficas Esri Corporativas (100% livres de autenticação, cotas ou bloqueios)
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

    esriTopo.addTo(map);

    const regionsLayer = L.layerGroup().addTo(map);
    const auditedLayer = L.layerGroup().addTo(map);

    L.control.layers(
      {{
        'Mapa Topográfico (Esri Topo)': esriTopo,
        'Mapa de Ruas (Esri Street)': esriStreet,
        'Satélite HD (Esri Imagery)': esriSat
      }},
      {{
        'Delimitações das Regiões': regionsLayer,
        'Árvores Auditadas': auditedLayer
      }},
      {{ position: 'topright' }}
    ).addTo(map);

    let currentRegion = 'ALL';
    let currentPreset = 'v3_gold_bicolor';

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

        if (currentPreset === 'v3_gold_bicolor') {{
          const v3 = t.v3_triage || t.v2_triage || {{}};
          colFront = v3.front_color || '#10b981';
          colSide  = v3.side_color || '#10b981';
          isParalax = Boolean(v3.has_paralax_divergence);
          consTriage = v3.consolidated || 'SEGURO';
        }} else if (currentPreset === 'v2_full_bicolor') {{
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
          if (currentPreset === 'v3_gold_bicolor') {{
            an = v.v3_analysis || v.v2_analysis || {{}};
            ovPath = an.v3_overlay_file ? ('v3_overlays/' + an.v3_overlay_file) : (an.v2_overlay_file ? ('v2_overlays/' + an.v2_overlay_file) : ('images/' + v.file_name));
            triageCol = an.color_hex || '#475569';
          }} else if (currentPreset === 'v2_full_bicolor') {{
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
        if (currentPreset === 'v3_gold_bicolor') {{
          cons = (t.v3_triage || t.v2_triage || {{}}).consolidated || 'SEGURO';
        }} else if (currentPreset === 'v2_full_bicolor') {{
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
          opacity: isSelected ? 1.0 : (isDimmed ? 0.20 : 0.85),
          interactive: false // Sem eventos de hover intrusivos sobre o mapa
        }};

        let shape;
        if (reg.corridor_polygon) {{
          shape = L.polygon(reg.corridor_polygon, shapeOptions);
        }} else {{
          const [minLon, minLat, maxLon, maxLat] = reg.bbox;
          shape = L.rectangle([[minLat, minLon], [maxLat, maxLon]], shapeOptions);
        }}

        regionsLayer.addLayer(shape);
      }}
    }}

    function buildRegionTabs() {{
      const container = document.getElementById('regionTabs');
      container.innerHTML = '';

      const btnAll = document.createElement('button');
      btnAll.className = 'reg-btn active';
      btnAll.innerText = 'Todos os Bairros (' + trees.length + ')';
      btnAll.onclick = () => selectRegion('ALL');
      container.appendChild(btnAll);

      for (const [key, reg] of Object.entries(regionsCatalog)) {{
        const count = trees.filter(t => (t.region_key || 'pinheiros') === key).length;
        if (count === 0) continue;
        const btn = document.createElement('button');
        btn.className = 'reg-btn';
        btn.setAttribute('data-key', key);
        btn.innerText = `${{reg.label.split('(')[0].trim()}} (${{count}})`;
        btn.onclick = () => selectRegion(key);
        container.appendChild(btn);
      }}
    }}

    function renderSidebar() {{
      const cardsContainer = document.getElementById('sidebarRegionCards');
      cardsContainer.innerHTML = '';

      // Card "Todas as Regiões"
      const allCard = document.createElement('div');
      allCard.className = 's-region-card' + (currentRegion === 'ALL' ? ' active' : '');
      allCard.setAttribute('data-key', 'ALL');
      allCard.innerHTML = `
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="width:9px; height:9px; border-radius:50%; background:#3b82f6;"></span>
          <span>Todas as Regiões (Geral SP)</span>
        </div>
        <span class="badge" style="background:#334155; font-size:10px;">${{trees.length}}</span>`;
      allCard.onclick = () => selectRegion('ALL');
      cardsContainer.appendChild(allCard);

      // Cards por região
      for (const [key, reg] of Object.entries(regionsCatalog)) {{
        const count = trees.filter(t => (t.region_key || 'pinheiros') === key).length;
        if (count === 0) continue;
        const st = regionStyles[key] || {{ color: '#64748b' }};
        const card = document.createElement('div');
        card.className = 's-region-card' + (currentRegion === key ? ' active' : '');
        card.setAttribute('data-key', key);
        card.innerHTML = `
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:9px; height:9px; border-radius:50%; background:${{st.color}};"></span>
            <span>${{reg.label.split('(')[0].trim()}}</span>
          </div>
          <span class="badge" style="background:#1e293b; border:1px solid #475569; font-size:10px;">${{count}}</span>`;
        card.onclick = () => selectRegion(key);
        cardsContainer.appendChild(card);
      }}

      renderDetailPanel();
    }}

    function renderDetailPanel() {{
      const panel = document.getElementById('sidebarDetailPanel');
      if (currentRegion === 'ALL') {{
        const critTotal = document.getElementById('statCritConfirm').innerText;
        const paralaxTotal = document.getElementById('statParalax').innerText;
        const proxTotal = document.getElementById('statProx').innerText;
        const safeTotal = document.getElementById('statSafe').innerText;

        panel.innerHTML = `
          <div style="font-size:13px; font-weight:700; color:#f8fafc; margin-bottom:4px;">
            Município de São Paulo (Consolidado)
          </div>
          <span class="grid-badge-pill" style="background:#1e3a8a; color:#93c5fd; border:1px solid #3b82f6;">
            4 Territórios · Rede Aérea + Subterrânea
          </span>
          <p style="color:#94a3b8; font-size:11px; margin:8px 0 12px 0;">
            Auditoria multi-perspectiva integrando bairros com rede elétrica aérea convencional (Pinheiros, Mooca, Vila Mariana) e um corredor de controle negativo com rede 100% subterrânea (Avenida Paulista).
          </p>
          <div style="font-weight:700; font-size:11px; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.04em;">
            Quadro Geral de Riscos (Modelo Ativo)
          </div>
          <div class="s-metric-grid">
            <div class="s-metric-box"><span style="color:#f87171;">Crítico Confirmado</span><b style="color:#ef4444;">${{critTotal}}</b></div>
            <div class="s-metric-box"><span style="color:#fb923c;">Paralaxe (Divergente)</span><b style="color:#f97316;">${{paralaxTotal}}</b></div>
            <div class="s-metric-box"><span style="color:#fbbf24;">Proximidade (Vento)</span><b style="color:#f59e0b;">${{proxTotal}}</b></div>
            <div class="s-metric-box"><span style="color:#34d399;">Seguro Certificado</span><b style="color:#10b981;">${{safeTotal}}</b></div>
          </div>
          <div style="background:#111827; padding:9px; border-radius:6px; border:1px solid #1e293b; margin-top:10px; font-size:11px; color:#cbd5e1;">
            <div>Total Árvores Auditadas: <b>${{trees.length}}</b></div>
            <div style="margin-top:2px;">Total Fotografias Street View: <b>${{trees.reduce((acc, t) => acc + (t.views ? t.views.length : 0), 0)}}</b></div>
            <div style="margin-top:2px;">Base Cadastral: <b>GeoSampa / SISGAU (PMSP)</b></div>
          </div>`;
      }} else {{
        const reg = regionsCatalog[currentRegion] || {{}};
        const regTrees = trees.filter(t => (t.region_key || 'pinheiros') === currentRegion);
        const st = regionStyles[currentRegion] || {{ color: '#3b82f6' }};

        let crit = 0, paralax = 0, prox = 0, safe = 0;
        regTrees.forEach(t => {{
          let c = 'SEGURO';
          if (currentPreset === 'v3_gold_bicolor') {{
            c = (t.v3_triage || t.v2_triage || {{}}).consolidated || 'SEGURO';
          }} else if (currentPreset === 'v2_full_bicolor') {{
            c = (t.v2_triage || {{}}).consolidated || 'SEGURO';
          }} else if (currentPreset === 'gold_segformer_bicolor') {{
            c = (t.gold_triage || {{}}).consolidated || 'SEGURO';
          }} else {{
            const r = t.consolidated_risk || 'LOW';
            c = (r === 'HIGH') ? 'CRITICO_CONFIRMADO' : (r === 'MEDIUM') ? 'PROXIMIDADE' : 'SEGURO';
          }}
          if (c === 'CRITICO_CONFIRMADO') crit++;
          else if (c === 'DIVERGENCIA_PARALAXE') paralax++;
          else if (c === 'PROXIMIDADE' || c === 'CRITICO_MONITORADO') prox++;
          else safe++;
        }});

        const gridBadgeStyle = reg.is_negative_control
          ? 'background:#064e3b; color:#6ee7b7; border:1px solid #059669;'
          : 'background:#78350f; color:#fde68a; border:1px solid #d97706;';

        const gridTitle = reg.is_negative_control
          ? '● REDE 100% SUBTERRÂNEA (CONTROLE NEGATIVO)'
          : '▲ REDE DE DISTRIBUIÇÃO AÉREA (ENEL SP)';

        panel.innerHTML = `
          <div style="font-size:13px; font-weight:700; color:${{st.color}}; margin-bottom:4px;">
            ${{reg.label}}
          </div>
          <span class="grid-badge-pill" style="${{gridBadgeStyle}}">
            ${{gridTitle}}
          </span>
          <p style="color:#cbd5e1; font-size:11px; margin:8px 0 12px 0;">
            ${{reg.context || reg.description || ''}}
          </p>

          <div style="font-weight:700; font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em;">
            Diagnóstico Territorial (Modelo Ativo)
          </div>
          <div class="s-metric-grid">
            <div class="s-metric-box"><span style="color:#f87171;">Crítico Confirmado</span><b style="color:#ef4444;">${{crit}}</b></div>
            <div class="s-metric-box"><span style="color:#fb923c;">Paralaxe (Divergente)</span><b style="color:#f97316;">${{paralax}}</b></div>
            <div class="s-metric-box"><span style="color:#fbbf24;">Proximidade (Vento)</span><b style="color:#f59e0b;">${{prox}}</b></div>
            <div class="s-metric-box"><span style="color:#34d399;">Seguro Certificado</span><b style="color:#10b981;">${{safe}}</b></div>
          </div>

          <div style="background:#111827; padding:9px; border-radius:6px; border:1px solid #1e293b; margin-top:10px; font-size:11px; color:#cbd5e1;">
            <div>Árvores Cadastradas GeoSampa: <b>${{reg.total_geosampa_trees || 'N/D'}}</b></div>
            <div style="margin-top:2px;">Amostra Auditada: <b>${{regTrees.length}} árvores</b> (${{regTrees.reduce((acc, t) => acc + (t.views ? t.views.length : 0), 0)}} fotografias)</div>
            <div style="margin-top:2px;">Proporção de Conflito Crítico: <b>${{((crit / (regTrees.length || 1)) * 100).toFixed(1)}}%</b></div>
          </div>

          <button id="btnCenterRegion" style="margin-top:12px; width:100%; padding:7px 10px; background:#1e293b; border:1px solid #3b82f6; border-radius:6px; color:#93c5fd; font-size:11px; font-weight:700; cursor:pointer;">
            Centralizar Este Setor no Mapa ↗
          </button>`;

        const btnCenter = document.getElementById('btnCenterRegion');
        if (btnCenter) {{
          btnCenter.onclick = () => focusRegion(currentRegion);
        }}
      }}
    }}

    function selectRegion(key) {{
      currentRegion = key;

      document.querySelectorAll('.reg-btn').forEach(b => {{
        const bKey = b.getAttribute('data-key') || 'ALL';
        b.classList.toggle('active', bKey === key);
      }});

      document.querySelectorAll('.s-region-card').forEach(c => {{
        const cKey = c.getAttribute('data-key') || 'ALL';
        c.classList.toggle('active', cKey === key);
      }});

      focusRegion(key);
      renderRegionBoxes();
      renderMarkers();
      renderDetailPanel();
    }}

    function focusRegion(key) {{
      if (key !== 'ALL' && regionsCatalog[key]) {{
        const reg = regionsCatalog[key];
        if (reg.corridor_polygon) {{
          const poly = L.polygon(reg.corridor_polygon);
          map.fitBounds(poly.getBounds(), {{ padding: [40, 40], maxZoom: 16, duration: 0.8 }});
        }} else if (reg.bbox) {{
          const [minLon, minLat, maxLon, maxLat] = reg.bbox;
          map.fitBounds([[minLat, minLon], [maxLat, maxLon]], {{ padding: [40, 40], maxZoom: 16, duration: 0.8 }});
        }}
        document.getElementById('statRegion').innerText = reg.label.split('(')[0].trim();
      }} else {{
        map.flyTo([-23.5650, -46.6450], 13, {{ duration: 0.8 }});
        document.getElementById('statRegion').innerText = 'Todas as Regiões';
      }}
    }}

    document.getElementById('presetSelect').onchange = (e) => {{
      currentPreset = e.target.value;
      renderMarkers();
      renderDetailPanel();
    }};

    document.getElementById('toggleBBoxes').addEventListener('change', (e) => {{
      if (e.target.checked) {{
        map.addLayer(regionsLayer);
      }} else {{
        map.removeLayer(regionsLayer);
      }}
    }});

    document.getElementById('toggleSidebarBtn').onclick = () => {{
      const sb = document.getElementById('sidebar');
      sb.classList.toggle('collapsed');
      const isCollapsed = sb.classList.contains('collapsed');
      document.getElementById('sidebarToggleIcon').innerText = isCollapsed ? '▶' : '◀';
      setTimeout(() => map.invalidateSize(), 240);
    }};

    buildRegionTabs();
    renderSidebar();
    renderRegionBoxes();
    renderMarkers();
  </script>
</body>
</html>
"""
    MAP_PATH.write_text(html_code, encoding="utf-8")
    print(f"Mapa HTML atualizado com sucesso em: {MAP_PATH}")

    return MAP_PATH


if __name__ == "__main__":
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    generate_html_map(manifest)

