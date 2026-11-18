import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { AlertCircle, CloudRain, ExternalLink, Search, Wind, X } from "lucide-react";
import type { MapBootstrapPayload, TreeRecord } from "../types/arborgrid";
import {
  ANALYSIS_GRID_PIXELS,
  formatMacroTriageShort,
  formatPriorityTier,
  formatRelativePct,
  formatTriageLabel,
} from "../utils/labels";

interface OperationalMapViewProps {
  data: MapBootstrapPayload;
  initialTreeId?: string | null;
  initialRegionKey?: string | null;
  onConsultAssistant: (geosampaId: string) => void;
}

type BaseMapKind = "topo" | "street" | "satellite";

const TRIAGE_COLORS: Record<string, string> = {
  CRITICO: "#dc2626",
  PROXIMIDADE: "#d97706",
  SEGURO: "#15803d",
};

const TRIAGE_BADGE_CLS: Record<string, string> = {
  CRITICO: "border-[#FECACA] bg-[#FEF2F2] text-[#991B1B]",
  PROXIMIDADE: "border-[#FDE68A] bg-[#FFFBEB] text-[#92400E]",
  SEGURO: "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]",
};

const BASE_TILES: Record<
  BaseMapKind,
  { label: string; url: string; attribution: string; maxZoom: number }
> = {
  topo: {
    label: "Topográfico (Esri)",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri &amp; GIS Community | GeoSampa/PMSP",
    maxZoom: 19,
  },
  street: {
    label: "Ruas (Esri)",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri | GeoSampa/PMSP | Street View",
    maxZoom: 19,
  },
  satellite: {
    label: "Satélite (Esri)",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Satélite &copy; Esri, Maxar, Earthstar Geographics | GeoSampa/PMSP",
    maxZoom: 19,
  },
};

function createTreeMarkerIcon(tree: TreeRecord, goldTier?: string): L.DivIcon {
  const isParallax = tree.v3_triage.has_paralax_divergence;
  const macroColor = TRIAGE_COLORS[tree.v3_triage.consolidated] ?? "#15803d";
  const frontColor = tree.v3_triage.front_color || macroColor;
  const sideColor = tree.v3_triage.side_color || macroColor;
  const hasUrgentGold = goldTier === "URGENTE" || goldTier === "ALTA";
  const ringBorder = hasUrgentGold ? "#18241B" : "#ffffff";

  if (isParallax) {
    const html = `
      <div style="position:relative;width:20px;height:20px;display:flex;align-items:center;justify-content:center;">
        <div class="parallax-ring" style="position:absolute;inset:-3px;border-radius:9999px;border:2px solid #d97706;"></div>
        <div style="width:16px;height:16px;border-radius:9999px;overflow:hidden;border:2px solid ${ringBorder};display:flex;">
          <div style="width:50%;height:100%;background:${frontColor};"></div>
          <div style="width:50%;height:100%;background:${sideColor};"></div>
        </div>
      </div>
    `;
    return L.divIcon({
      className: "arbor-parallax-pin",
      html,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });
  }

  const size = tree.v3_triage.consolidated === "CRITICO" ? 14 : 11;
  const html = `
    <div style="width:${size}px;height:${size}px;border-radius:9999px;background:${macroColor};border:2px solid ${ringBorder};"></div>
  `;
  return L.divIcon({
    className: "arbor-parallax-pin",
    html,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

export function OperationalMapView({
  data,
  initialTreeId = null,
  initialRegionKey = null,
  onConsultAssistant,
}: OperationalMapViewProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);

  const [baseMap, setBaseMap] = useState<BaseMapKind>("topo");
  const [regionFilter, setRegionFilter] = useState<string>(initialRegionKey || "all");
  const [triageFilter, setTriageFilter] = useState<string>("all");
  const [onlyGold, setOnlyGold] = useState<boolean>(false);
  const [searchId, setSearchId] = useState<string>("");
  const [selectedTree, setSelectedTree] = useState<TreeRecord | null>(() =>
    initialTreeId ? data.trees.find((t) => t.geosampa_id === initialTreeId) ?? null : null,
  );

  useEffect(() => {
    if (initialRegionKey) {
      setRegionFilter(initialRegionKey);
    }
  }, [initialRegionKey]);

  useEffect(() => {
    if (initialTreeId) {
      const found = data.trees.find((t) => t.geosampa_id === initialTreeId) ?? null;
      if (found) setSelectedTree(found);
    }
  }, [initialTreeId, data.trees]);

  const filteredTrees = useMemo(() => {
    const q = searchId.trim().toLowerCase();
    return data.trees.filter((tree) => {
      if (regionFilter !== "all" && tree.region_key !== regionFilter) return false;
      if (triageFilter !== "all") {
        if (
          triageFilter === "DIVERGENCIA_PARALAXE" ||
          triageFilter === "CRITICO_CONFIRMADO"
        ) {
          if (tree.v3_triage.raw_consolidated !== triageFilter) return false;
        } else if (tree.v3_triage.consolidated !== triageFilter) {
          return false;
        }
      }
      if (onlyGold && !data.gold_scores_by_tree[tree.geosampa_id]) return false;
      if (q && !tree.geosampa_id.toLowerCase().includes(q) && !tree.distrito.toLowerCase().includes(q)) {
        return false;
      }
      return true;
    });
  }, [data.trees, data.gold_scores_by_tree, regionFilter, triageFilter, onlyGold, searchId]);

  // Inicializa Leaflet uma vez
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;
    const map = L.map(mapContainerRef.current, {
      center: [-23.568, -46.652],
      zoom: 13,
      zoomControl: true,
    });
    const layerGroup = L.layerGroup().addTo(map);
    mapRef.current = map;
    markersLayerRef.current = layerGroup;

    return () => {
      map.remove();
      mapRef.current = null;
      tileLayerRef.current = null;
      markersLayerRef.current = null;
    };
  }, []);

  // Troca camada base Esri sem recriar o mapa
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (tileLayerRef.current) {
      tileLayerRef.current.remove();
    }
    const cfg = BASE_TILES[baseMap];
    const layer = L.tileLayer(cfg.url, {
      attribution: cfg.attribution,
      maxZoom: cfg.maxZoom,
    }).addTo(map);
    tileLayerRef.current = layer;
  }, [baseMap]);

  // Zoom no territorio selecionado
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (regionFilter !== "all") {
      const cat = data.regions_catalog[regionFilter];
      if (cat?.bbox) {
        const [minLon, minLat, maxLon, maxLat] = cat.bbox;
        map.fitBounds(
          [
            [minLat, minLon],
            [maxLat, maxLon],
          ],
          { padding: [24, 24] },
        );
      }
    } else if (!selectedTree) {
      map.setView([-23.568, -46.652], 13);
    }
  }, [regionFilter, data.regions_catalog]);

  // Recalcula largura do mapa ao abrir ou fechar o painel lateral
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const timer = window.setTimeout(() => {
      map.invalidateSize();
    }, 60);
    return () => window.clearTimeout(timer);
  }, [selectedTree]);

  // Atualiza zonas demarcadas e marcadores filtrados
  useEffect(() => {
    const map = mapRef.current;
    const group = markersLayerRef.current;
    if (!map || !group) return;
    group.clearLayers();

    for (const [regionKey, cat] of Object.entries(data.regions_catalog)) {
      if (regionFilter !== "all" && regionFilter !== regionKey) continue;
      if (cat.corridor_polygon && cat.corridor_polygon.length > 0) {
        L.polygon(cat.corridor_polygon, {
          color: "#1B4332",
          weight: 2,
          dashArray: "6 4",
          fillColor: "#2D6A4F",
          fillOpacity: 0.08,
        })
          .bindTooltip(`${cat.label} — Área monitorada`, { sticky: true })
          .addTo(group);
      } else if (cat.bbox && cat.bbox.length === 4) {
        const [minLon, minLat, maxLon, maxLat] = cat.bbox;
        L.rectangle(
          [
            [minLat, minLon],
            [maxLat, maxLon],
          ],
          {
            color: "#1B4332",
            weight: 2,
            dashArray: "6 4",
            fillColor: "#2D6A4F",
            fillOpacity: 0.06,
          },
        )
          .bindTooltip(`${cat.label} — Área monitorada`, { sticky: true })
          .addTo(group);
      }
    }

    for (const tree of filteredTrees) {
      const gold = data.gold_scores_by_tree[tree.geosampa_id];
      const marker = L.marker([tree.tree_lat, tree.tree_lon], {
        icon: createTreeMarkerIcon(tree, gold?.priority_tier),
      });
      const labelStatus = formatTriageLabel(tree.v3_triage.raw_consolidated);
      marker.bindTooltip(
        `<strong>Árvore #${tree.geosampa_id}</strong> · ${tree.distrito}<br/>${labelStatus}${
          gold ? `<br/>Prioridade climática: ${formatPriorityTier(gold.priority_tier)} (${gold.composite_risk_score}/100)` : ""
        }`,
        { direction: "top" },
      );
      marker.on("click", () => {
        setSelectedTree(tree);
      });
      marker.addTo(group);
    }

    if (selectedTree) {
      map.panTo([selectedTree.tree_lat, selectedTree.tree_lon], { animate: true });
    }
  }, [filteredTrees, data.gold_scores_by_tree, data.regions_catalog, regionFilter, selectedTree]);

  const selectedGold = selectedTree ? data.gold_scores_by_tree[selectedTree.geosampa_id] : undefined;

  return (
    <div className="mx-auto max-w-[1680px] px-4 py-6 sm:px-8">
      {/* Layout Fixo: Mapa maior à esquerda (7 colunas) + Painel lateral compacto à direita (5 colunas) */}
      <div className="grid items-stretch gap-4 lg:grid-cols-12">
        <div className="relative overflow-hidden rounded-md border border-[#D5E0D1] bg-white lg:col-span-7">
          <div ref={mapContainerRef} className="h-[680px] w-full" />

          {/* Legenda discreta */}
          <div className="pointer-events-none absolute bottom-3 left-3 z-[400] rounded border border-[#D5E0D1] bg-white/95 px-3.5 py-2.5 text-xs text-[#18241B]">
            <div className="mb-1 font-semibold text-[#1B4332]">
              Exibindo {filteredTrees.length} de {data.trees.length} árvores
            </div>
            <div className="flex flex-wrap items-center gap-3 text-[#3D4F41]">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#dc2626]" /> Risco Crítico
              </span>
              <span className="flex items-center gap-1.5">
                <span className="flex h-3 w-3 overflow-hidden rounded-full border border-[#d97706]">
                  <span className="w-1/2 bg-[#15803d]" />
                  <span className="w-1/2 bg-[#dc2626]" />
                </span>
                Mudança por Ângulo
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#d97706]" /> Atenção Vento
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#15803d]" /> Sem Conflito
              </span>
            </div>
          </div>
        </div>

        {/* Painel Lateral Fixo: Árvore em foco OU Seletores e regiões quando nada está em foco */}
        <aside className="flex flex-col justify-between rounded-md border border-[#D5E0D1] bg-white p-4 lg:col-span-5">
          {selectedTree ? (
            <div className="flex flex-1 flex-col justify-between gap-3">
              <div className="space-y-3">
                {/* Cabeçalho com botão X à esquerda */}
                <div className="flex items-start gap-2.5 border-b border-[#E6ECE3] pb-2.5">
                  <button
                    type="button"
                    aria-label="Fechar painel da árvore"
                    title="Voltar aos filtros e camadas"
                    onClick={() => setSelectedTree(null)}
                    className="mt-0.5 shrink-0 rounded border border-[#D5E0D1] bg-[#F6F8F5] p-1.5 text-[#4A5D4E] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC] hover:text-[#18241B]"
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="font-mono text-base font-bold text-[#18241B]">
                        Árvore GeoSampa #{selectedTree.geosampa_id}
                      </h3>
                      <span
                        className={`rounded border px-2 py-0.5 text-xs font-semibold ${
                          TRIAGE_BADGE_CLS[selectedTree.v3_triage.consolidated] ??
                          TRIAGE_BADGE_CLS.SEGURO
                        }`}
                      >
                        {formatTriageLabel(selectedTree.v3_triage.raw_consolidated)}
                      </span>
                    </div>
                    <p className="mt-0.5 text-xs text-[#4A5D4E]">
                      {selectedTree.distrito} · Subprefeitura {selectedTree.subprefeitura} · Coords{" "}
                      <span className="font-mono text-[#18241B]">
                        {selectedTree.tree_lat.toFixed(5)}, {selectedTree.tree_lon.toFixed(5)}
                      </span>{" "}
                      · SISGAU:{" "}
                      <strong className="font-medium text-[#92400E]">
                        {selectedTree.sisgau_last_inspection || "Sem registro"}
                      </strong>
                    </p>
                  </div>
                </div>

                {/* Alertas compactos (Divergência + Prioridade Climática) */}
                {(selectedTree.v3_triage.has_paralax_divergence || selectedGold) && (
                  <div className="grid gap-2 sm:grid-cols-2">
                    {selectedTree.v3_triage.has_paralax_divergence && (
                      <div className="flex items-start gap-2 rounded border border-[#FDE68A] bg-[#FFFBEB] p-2.5 text-xs leading-snug text-[#92400E]">
                        <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                        <div>
                          <strong>Diferença entre ângulos:</strong> frontal{" "}
                          <strong>{formatMacroTriageShort(selectedTree.v3_triage.front_triage)}</strong>{" "}
                          vs lateral{" "}
                          <strong>{formatMacroTriageShort(selectedTree.v3_triage.side_triage)}</strong>.
                        </div>
                      </div>
                    )}

                    {selectedGold && (
                      <div
                        className={`rounded border border-[#C6DBC0] bg-[#EEF4EC] p-2.5 text-xs leading-snug ${
                          selectedTree.v3_triage.has_paralax_divergence ? "" : "sm:col-span-2"
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="font-semibold text-[#1B4332]">
                            Prioridade Climática ({selectedGold.composite_risk_score.toFixed(1)}/100 ·{" "}
                            {formatPriorityTier(selectedGold.priority_tier)})
                          </span>
                          <span className="inline-flex items-center gap-2 text-[#3D4F41]">
                            <span className="inline-flex items-center gap-1">
                              <Wind className="h-3 w-3 text-[#2D6A4F]" />{" "}
                              {selectedGold.wind_gust_kmh ?? "—"} km/h
                            </span>
                            <span className="inline-flex items-center gap-1">
                              <CloudRain className="h-3 w-3 text-[#2D6A4F]" />{" "}
                              {selectedGold.precipitation_mm ?? "—"} mm
                            </span>
                          </span>
                        </div>
                        <p className="mt-1 text-[#18241B]">{selectedGold.recommended_action}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Fotos Analisadas lado a lado sem corte */}
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h4 className="text-xs font-semibold uppercase tracking-wide text-[#4A5D4E]">
                      Ângulos da via ({selectedTree.views.length})
                    </h4>
                    <span className="text-xs text-[#4A5D4E]">
                      Fontes: GeoSampa/SISGAU · Google Street View
                    </span>
                  </div>

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {selectedTree.views.map((view, idx) => {
                      const v3 = view.v3_analysis;
                      const angleLabel =
                        view.perspective === "center_front" ? "Ângulo frontal" : "Ângulo lateral";
                      return (
                        <div
                          key={`${view.perspective}-${idx}`}
                          className="flex flex-col rounded border border-[#D5E0D1] bg-[#FAFBF9] p-2.5 text-xs"
                        >
                          <div className="mb-2 flex items-center justify-between gap-1">
                            <span className="font-semibold text-[#18241B]">
                              {angleLabel}{" "}
                              <span className="font-normal text-[#4A5D4E]">
                                ({view.capture_date || "—"})
                              </span>
                            </span>
                            <span
                              className={`rounded border px-1.5 py-0.5 text-xs font-semibold ${
                                TRIAGE_BADGE_CLS[v3.triage] ?? TRIAGE_BADGE_CLS.SEGURO
                              }`}
                            >
                              {formatMacroTriageShort(v3.triage)}
                            </span>
                          </div>

                          {view.overlay_url && (
                            <img
                              src={view.overlay_url}
                              alt={`Análise visual árvore ${selectedTree.geosampa_id} ${angleLabel}`}
                              className="aspect-square w-full rounded border border-[#D5E0D1] bg-[#F6F8F5] object-contain"
                            />
                          )}

                          <div className="mt-2 grid grid-cols-2 gap-x-2 gap-y-0.5 text-xs text-[#4A5D4E]">
                            <div>
                              Contato:{" "}
                              <strong className="text-[#18241B]">
                                {formatRelativePct(v3.contact_direct_px, v3.copa_px)}
                              </strong>{" "}
                              copa
                            </div>
                            <div>
                              Vento:{" "}
                              <strong className="text-[#18241B]">
                                {formatRelativePct(v3.wind_buffer_px, v3.copa_px)}
                              </strong>{" "}
                              copa
                            </div>
                            <div>
                              Fiação:{" "}
                              <strong className="text-[#18241B]">
                                {formatRelativePct(v3.wire_px, ANALYSIS_GRID_PIXELS)}
                              </strong>{" "}
                              foto
                            </div>
                            <div>
                              Copa:{" "}
                              <strong className="text-[#18241B]">
                                {formatRelativePct(v3.copa_px, ANALYSIS_GRID_PIXELS)}
                              </strong>{" "}
                              foto
                            </div>
                          </div>

                          {view.google_maps_url && (
                            <a
                              href={view.google_maps_url}
                              target="_blank"
                              rel="noreferrer"
                              className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-[#1B4332] hover:underline"
                            >
                              <span>Abrir no Google Street View</span>
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={() => onConsultAssistant(selectedTree.geosampa_id)}
                className="flex w-full items-center justify-center rounded bg-[#1B4332] px-4 py-2.5 text-xs font-semibold text-white transition-colors hover:bg-[#143325]"
              >
                Analisar com IA
              </button>
            </div>
          ) : (
            <div className="flex flex-1 flex-col gap-3.5">
              <div className="border-b border-[#E6ECE3] pb-2.5">
                <h2 className="text-base font-bold text-[#18241B]">
                  Filtros e Regiões do Mapa
                </h2>
                <p className="mt-0.5 text-xs text-[#4A5D4E]">
                  Escolha uma região ou filtre por situação, ajuste o mapa base e clique em uma
                  árvore no mapa para inspecionar os dois ângulos.
                </p>
              </div>

              <div className="grid gap-2.5 sm:grid-cols-2">
                <div>
                  <label
                    htmlFor="map-search-input"
                    className="mb-1 block text-xs font-medium text-[#4A5D4E]"
                  >
                    Buscar árvore ou bairro
                  </label>
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-3 top-2.5 h-3.5 w-3.5 text-[#4A5D4E]" />
                    <input
                      id="map-search-input"
                      type="text"
                      value={searchId}
                      onChange={(e) => setSearchId(e.target.value)}
                      placeholder="ID ou bairro (ex: #635210)..."
                      className="w-full rounded border border-[#D5E0D1] bg-[#F6F8F5] py-1.5 pl-8 pr-3 text-xs text-[#18241B] placeholder-[#6B7D70] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
                    />
                  </div>
                </div>

                <div>
                  <label
                    htmlFor="map-triage-select"
                    className="mb-1 block text-xs font-medium text-[#4A5D4E]"
                  >
                    Situação identificada
                  </label>
                  <select
                    id="map-triage-select"
                    value={triageFilter}
                    onChange={(e) => setTriageFilter(e.target.value)}
                    className="w-full rounded border border-[#D5E0D1] bg-[#F6F8F5] px-3 py-1.5 text-xs text-[#18241B] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
                  >
                    <option value="all">Todas as situações</option>
                    <option value="CRITICO">Risco Crítico (Todos)</option>
                    <option value="CRITICO_CONFIRMADO">Crítico Confirmado nos 2 Ângulos</option>
                    <option value="DIVERGENCIA_PARALAXE">Visível ao Mudar o Ângulo</option>
                    <option value="PROXIMIDADE">Atenção (Proximidade ao Vento)</option>
                    <option value="SEGURO">Sem Conflito Aparente</option>
                  </select>
                </div>
              </div>

              {/* Camada do mapa à esquerda · Com Prioridade Climática à direita */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-y border-[#E6ECE3] py-2.5">
                <div className="flex flex-wrap items-center gap-1">
                  <span className="mr-1 text-xs font-medium text-[#4A5D4E]">Camada:</span>
                  {(Object.keys(BASE_TILES) as BaseMapKind[]).map((kind) => (
                    <button
                      key={kind}
                      type="button"
                      onClick={() => setBaseMap(kind)}
                      className={`rounded border px-2 py-1 text-xs font-medium transition-colors ${
                        baseMap === kind
                          ? "border-[#1B4332] bg-[#1B4332] text-white"
                          : "border-[#D5E0D1] bg-[#F6F8F5] text-[#2D4234] hover:bg-[#E7EFE4]"
                      }`}
                    >
                      {BASE_TILES[kind].label}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => setOnlyGold((v) => !v)}
                  className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors ${
                    onlyGold
                      ? "border-[#1B4332] bg-[#1B4332] text-white"
                      : "border-[#D5E0D1] bg-white text-[#2D4234] hover:bg-[#EEF4EC]"
                  }`}
                >
                  Com Prioridade Climática ({Object.keys(data.gold_scores_by_tree).length})
                </button>
              </div>

              {/* Lista de regiões monitoradas com estatísticas básicas */}
              <div className="flex flex-1 flex-col gap-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-[#18241B]">Regiões monitoradas</span>
                  {regionFilter !== "all" ? (
                    <button
                      type="button"
                      onClick={() => setRegionFilter("all")}
                      className="text-xs font-medium text-[#1B4332] hover:underline"
                    >
                      Ver todas ({data.trees.length})
                    </button>
                  ) : (
                    <span className="text-xs text-[#4A5D4E]">
                      Clique para centralizar e filtrar
                    </span>
                  )}
                </div>

                <div className="space-y-2">
                  <button
                    type="button"
                    onClick={() => setRegionFilter("all")}
                    className={`flex w-full items-center justify-between rounded border px-3 py-2 text-left text-xs transition-colors ${
                      regionFilter === "all"
                        ? "border-[#1B4332] bg-[#EEF4EC] text-[#18241B]"
                        : "border-[#D5E0D1] bg-[#FAFBF9] text-[#2D4234] hover:border-[#2D6A4F]"
                    }`}
                  >
                    <span className="font-semibold">Todas as regiões de estudo</span>
                    <span className="text-xs text-[#4A5D4E]">
                      {data.stats.total_trees} árvores avaliadas ·{" "}
                      {data.stats.tree_triage_counts.CRITICO ?? 0} críticas
                    </span>
                  </button>

                  {data.stats.district_breakdown.map((reg) => {
                    const active = regionFilter === reg.region_key;
                    return (
                      <button
                        key={reg.region_key}
                        type="button"
                        onClick={() =>
                          setRegionFilter((prev) =>
                            prev === reg.region_key ? "all" : reg.region_key,
                          )
                        }
                        className={`w-full rounded border px-3 py-2.5 text-left transition-colors ${
                          active
                            ? "border-[#1B4332] bg-[#EEF4EC]"
                            : "border-[#D5E0D1] bg-[#FAFBF9] hover:border-[#2D6A4F]"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-semibold text-[#18241B]">
                            {reg.region_label}
                          </span>
                          <span
                            className={`shrink-0 rounded border px-1.5 py-0.5 text-xs font-medium ${
                              reg.is_negative_control
                                ? "border-[#D5E0D1] bg-white text-[#4A5D4E]"
                                : "border-[#C6DBC0] bg-white text-[#1B4332]"
                            }`}
                          >
                            {reg.is_negative_control ? "Subterrânea (controle)" : "Rede aérea"}
                          </span>
                        </div>

                        <div className="mt-1.5 grid grid-cols-4 gap-2 text-xs">
                          <div>
                            <span className="text-[#4A5D4E]">Avaliadas</span>
                            <p className="font-semibold tabular-nums text-[#18241B]">
                              {reg.total_trees}
                            </p>
                          </div>
                          <div>
                            <span className="text-[#4A5D4E]">Críticas</span>
                            <p className="font-semibold tabular-nums text-[#991B1B]">
                              {reg.critico_total} ({reg.critical_rate_pct.toFixed(0)}%)
                            </p>
                          </div>
                          <div>
                            <span className="text-[#4A5D4E]">Vento</span>
                            <p className="font-semibold tabular-nums text-[#92400E]">
                              {reg.proximidade}
                            </p>
                          </div>
                          <div>
                            <span className="text-[#4A5D4E]">Sem conflito</span>
                            <p className="font-semibold tabular-nums text-[#15803d]">
                              {reg.seguro}
                            </p>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
