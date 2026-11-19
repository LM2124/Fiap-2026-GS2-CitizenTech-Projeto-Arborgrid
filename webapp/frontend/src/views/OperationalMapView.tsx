import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import {
  AlertCircle,
  CloudRain,
  ExternalLink,
  RotateCcw,
  Search,
  Sparkles,
  Wind,
  X,
} from "lucide-react";
import { OverlayPhoto } from "../components/OverlayPhoto";
import type { MapBootstrapPayload, TreeRecord, V3ViewAnalysis } from "../types/arborgrid";
import {
  APP_LABELS,
  formatDatePtBr,
  formatDistrictName,
  formatMacroTriageShort,
  formatNumberPtBr,
  formatPercentPtBr,
  formatPriorityTier,
  formatTriageLabel,
  getPriorityTierStyle,
} from "../utils/labels";

interface OperationalMapViewProps {
  data: MapBootstrapPayload;
  initialTreeId?: string | null;
  initialRegionKey?: string | null;
  onConsultAssistant: (geosampaId: string) => void;
}

type BaseMapKind = "topo" | "street" | "satellite";

// O bbox do catálogo é a área de busca; algumas árvores ficam rente à borda e o
// ícone do marcador passa dela. Expande para conter todas as árvores da área e
// soma uma folga visual (~70 m).
function studyAreaBounds(
  regionKey: string,
  bbox: number[],
  trees: TreeRecord[],
): [[number, number], [number, number]] {
  let [minLon, minLat, maxLon, maxLat] = bbox;
  for (const tree of trees) {
    if (tree.region_key !== regionKey) continue;
    if (tree.tree_lon < minLon) minLon = tree.tree_lon;
    if (tree.tree_lon > maxLon) maxLon = tree.tree_lon;
    if (tree.tree_lat < minLat) minLat = tree.tree_lat;
    if (tree.tree_lat > maxLat) maxLat = tree.tree_lat;
  }
  const margin = 0.0006;
  return [
    [minLat - margin, minLon - margin],
    [maxLat + margin, maxLon + margin],
  ];
}

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

  // Área clicável de 24x24 px mantendo o marcador visual compacto
  if (isParallax) {
    const html = `
      <div style="width:24px;height:24px;display:flex;align-items:center;justify-content:center;">
        <div style="position:relative;width:20px;height:20px;display:flex;align-items:center;justify-content:center;">
          <div class="parallax-ring" style="position:absolute;inset:-3px;border-radius:9999px;border:2px solid #d97706;"></div>
          <div style="width:16px;height:16px;border-radius:9999px;overflow:hidden;border:2px solid ${ringBorder};display:flex;">
            <div style="width:50%;height:100%;background:${frontColor};"></div>
            <div style="width:50%;height:100%;background:${sideColor};"></div>
          </div>
        </div>
      </div>
    `;
    return L.divIcon({
      className: "arbor-parallax-pin",
      html,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
  }

  const size = tree.v3_triage.consolidated === "CRITICO" ? 14 : 11;
  const html = `
    <div style="width:24px;height:24px;display:flex;align-items:center;justify-content:center;">
      <div style="width:${size}px;height:${size}px;border-radius:9999px;background:${macroColor};border:2px solid ${ringBorder};"></div>
    </div>
  `;
  return L.divIcon({
    className: "arbor-parallax-pin",
    html,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
}

function getPhotoStats(v3: V3ViewAnalysis): { primary: string; secondary: string | null } | null {
  if (typeof v3.wire_px !== "number") return null;
  if (v3.wire_px === 0) {
    return { primary: "Sem fiação visível nesta foto", secondary: null };
  }
  if (typeof v3.contact_points !== "number") {
    return null;
  }
  if (v3.contact_points === 0) {
    return { primary: "Fiação visível, sem galhos encostando", secondary: null };
  }
  // O percentual separa melhor os casos (ex.: 98% numa árvore crítica x 5% numa proximidade);
  // a contagem junta pedaços próximos do mesmo fio e entra como detalhe.
  const zones =
    v3.contact_points === 1
      ? "em 1 trecho de contato"
      : `em ${v3.contact_points.toLocaleString("pt-BR")} trechos de contato`;
  if (typeof v3.wire_contact_pct !== "number") {
    return { primary: `Galhos encostando na fiação ${zones}`, secondary: null };
  }
  return {
    primary: `${formatPercentPtBr(v3.wire_contact_pct, 0)} da fiação visível passa pela copa`,
    secondary: zones.charAt(0).toUpperCase() + zones.slice(1),
  };
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

  const unionBounds = useMemo<L.LatLngBoundsExpression | null>(() => {
    let minLat = Infinity;
    let minLon = Infinity;
    let maxLat = -Infinity;
    let maxLon = -Infinity;
    for (const cat of Object.values(data.regions_catalog)) {
      if (cat.bbox && cat.bbox.length === 4) {
        const [bMinLon, bMinLat, bMaxLon, bMaxLat] = cat.bbox;
        if (bMinLat < minLat) minLat = bMinLat;
        if (bMinLon < minLon) minLon = bMinLon;
        if (bMaxLat > maxLat) maxLat = bMaxLat;
        if (bMaxLon > maxLon) maxLon = bMaxLon;
      }
    }
    if (!Number.isFinite(minLat) || !Number.isFinite(minLon)) return null;
    return [
      [minLat, minLon],
      [maxLat, maxLon],
    ];
  }, [data.regions_catalog]);

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
      if (
        q &&
        !tree.geosampa_id.toLowerCase().includes(q) &&
        !tree.distrito.toLowerCase().includes(q) &&
        !formatDistrictName(tree.distrito).toLowerCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [data.trees, data.gold_scores_by_tree, regionFilter, triageFilter, onlyGold, searchId]);

  const hasActiveFilters =
    regionFilter !== "all" ||
    triageFilter !== "all" ||
    onlyGold ||
    searchId.trim().length > 0;

  const handleClearFilters = () => {
    setRegionFilter("all");
    setTriageFilter("all");
    setOnlyGold(false);
    setSearchId("");
    setSelectedTree(null);
  };

  // Inicializa Leaflet uma vez
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;
    const map = L.map(mapContainerRef.current, {
      center: [-23.568, -46.652],
      zoom: 13,
      zoomControl: false,
    });
    L.control.zoom({ zoomInTitle: "Aproximar", zoomOutTitle: "Afastar" }).addTo(map);
    map.attributionControl?.setPrefix("Mapa");

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

  // Posiciona câmera: árvore selecionada (zoom 17), região filtrada, ou fitBounds de todas as regiões
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const timer = window.setTimeout(() => {
      map.invalidateSize();
      if (selectedTree) {
        map.setView([selectedTree.tree_lat, selectedTree.tree_lon], 17, { animate: true });
      } else if (regionFilter !== "all") {
        const cat = data.regions_catalog[regionFilter];
        if (cat?.bbox && cat.bbox.length === 4) {
          // Folga extra embaixo para a legenda (canto inferior esquerdo) não cobrir árvores.
          map.fitBounds(studyAreaBounds(regionFilter, cat.bbox, data.trees), {
            paddingTopLeft: [24, 24],
            paddingBottomRight: [24, 72],
          });
        }
      } else if (unionBounds) {
        map.fitBounds(unionBounds, { padding: [24, 24] });
      }
    }, 60);

    return () => window.clearTimeout(timer);
  }, [selectedTree, regionFilter, unionBounds, data.regions_catalog, data.trees]);

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
          interactive: false,
          className: "arbor-study-area",
        }).addTo(group);
      } else if (cat.bbox && cat.bbox.length === 4) {
        L.rectangle(
          studyAreaBounds(regionKey, cat.bbox, data.trees),
          {
            color: "#1B4332",
            weight: 2,
            dashArray: "6 4",
            fillColor: "#2D6A4F",
            fillOpacity: 0.06,
            interactive: false,
            className: "arbor-study-area",
          },
        ).addTo(group);
      }
    }

    for (const tree of filteredTrees) {
      const gold = data.gold_scores_by_tree[tree.geosampa_id];
      const labelStatus = formatTriageLabel(tree.v3_triage.raw_consolidated);
      const marker = L.marker([tree.tree_lat, tree.tree_lon], {
        icon: createTreeMarkerIcon(tree, gold?.priority_tier),
        title: `Árvore #${tree.geosampa_id}, ${labelStatus}`,
        alt: `Árvore #${tree.geosampa_id}`,
      });
      marker.bindTooltip(
        `<strong>Árvore #${tree.geosampa_id}</strong>, ${formatDistrictName(tree.distrito)}<br/>${labelStatus}${
          gold
            ? `<br/>Prioridade climática: ${formatPriorityTier(gold.priority_tier)} (${formatNumberPtBr(gold.composite_risk_score, 1)}/100)`
            : ""
        }`,
        { direction: "top" },
      );
      marker.on("click", () => {
        setSelectedTree(tree);
      });
      marker.addTo(group);
    }
  }, [filteredTrees, data.trees, data.gold_scores_by_tree, data.regions_catalog, regionFilter]);

  const selectedGold = selectedTree ? data.gold_scores_by_tree[selectedTree.geosampa_id] : undefined;

  return (
    <div className="mx-auto max-w-[1680px] space-y-3 px-4 py-4 sm:px-8">
      {/* Faixa superior visível com título H1, contador e Limpar filtros */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-[#D5E0D1] bg-white px-4 py-2.5">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h1 className="text-base font-bold text-[#18241B] sm:text-lg">
            Mapa de Vistorias
          </h1>
          <span className="text-[13px] tabular-nums text-[#4A5D4E]">
            Exibindo{" "}
            <strong className="font-semibold text-[#1B4332]">
              {filteredTrees.length.toLocaleString("pt-BR")}
            </strong>{" "}
            de {data.trees.length.toLocaleString("pt-BR")} árvores
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {hasActiveFilters && (
            <button
              type="button"
              onClick={handleClearFilters}
              className="inline-flex items-center gap-1.5 rounded border border-[#2D6A4F] bg-[#EEF4EC] px-3 py-1.5 text-[13px] font-semibold text-[#1B4332] transition-colors hover:bg-[#DCEAD8]"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Limpar filtros</span>
            </button>
          )}
        </div>
      </div>

      {/* Layout: Mapa maior à esquerda (7 colunas) + Painel lateral à direita (5 colunas) */}
      <div className="grid items-stretch gap-4 lg:grid-cols-12">
        <div className="relative min-w-0 overflow-hidden rounded-md border border-[#D5E0D1] bg-white lg:col-span-7">
          <div
            ref={mapContainerRef}
            className="h-[calc(100vh-180px)] min-h-[480px] w-full"
          />

          {/* Estado vazio quando os filtros não retornam nenhuma árvore */}
          {filteredTrees.length === 0 && (
            <div className="pointer-events-none absolute inset-0 z-[410] flex items-center justify-center p-4">
              <div className="pointer-events-auto max-w-sm rounded-md border border-[#D5E0D1] bg-white/95 p-4 text-center shadow-sm">
                <p className="text-sm font-semibold text-[#18241B]">
                  Nenhuma árvore encontrada com esses filtros.
                </p>
                <p className="mt-1 text-[13px] text-[#4A5D4E]">
                  Limpe os filtros para voltar a exibir todas as árvores da amostra.
                </p>
                <button
                  type="button"
                  onClick={handleClearFilters}
                  className="mt-3 inline-flex items-center gap-1.5 rounded bg-[#1B4332] px-3.5 py-1.5 text-[13px] font-semibold text-white hover:bg-[#143325]"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  <span>Limpar filtros</span>
                </button>
              </div>
            </div>
          )}

          {/* Legenda */}
          <div className="pointer-events-none absolute bottom-3 left-3 z-[400] max-w-[calc(100%-24px)] rounded border border-[#D5E0D1] bg-white/95 px-3.5 py-2 text-[13px] text-[#18241B]">
            <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-[#3D4F41]">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#dc2626]" /> Risco crítico
              </span>
              <span className="flex items-center gap-1.5">
                <span className="flex h-3 w-3 overflow-hidden rounded-full border border-[#d97706]">
                  <span className="w-1/2 bg-[#15803d]" />
                  <span className="w-1/2 bg-[#dc2626]" />
                </span>
                {APP_LABELS.parallaxLabel}
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#d97706]" /> Atenção vento
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#15803d]" /> Sem conflito
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-full border-2 border-[#18241B] bg-white" /> Prioridade climática alta
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3.5 rounded-[2px] border border-dashed border-[#1B4332] bg-[#2D6A4F]/10" /> Área monitorada
              </span>
            </div>
            <p className="mt-1 text-[11px] text-[#4A5D4E]">
              Cada área mostra uma amostra das árvores cadastradas, não todas as árvores da região.
            </p>
          </div>
        </div>

        {/* Painel Lateral: Árvore em foco OU Seletores e regiões */}
        <aside className="flex min-w-0 flex-col justify-between rounded-md border border-[#D5E0D1] bg-white p-4 lg:col-span-5 lg:max-h-[calc(100vh-180px)] lg:overflow-y-auto">
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
                      <h2 className="font-mono text-base font-bold text-[#18241B]">
                        Árvore #{selectedTree.geosampa_id}
                      </h2>
                      <span
                        className={`rounded border px-2 py-0.5 text-[13px] font-semibold ${
                          TRIAGE_BADGE_CLS[selectedTree.v3_triage.consolidated] ??
                          TRIAGE_BADGE_CLS.SEGURO
                        }`}
                      >
                        {formatTriageLabel(selectedTree.v3_triage.raw_consolidated)}
                      </span>
                    </div>
                    <p className="mt-1 text-[13px] text-[#4A5D4E]">
                      {formatDistrictName(selectedTree.distrito)}, Subprefeitura{" "}
                      {formatDistrictName(selectedTree.subprefeitura)}. Última vistoria da Prefeitura
                      (SISGAU):{" "}
                      <strong className="font-medium text-[#92400E]">
                        {formatDatePtBr(selectedTree.sisgau_last_inspection)}
                      </strong>
                    </p>
                  </div>
                </div>

                {/* Alertas compactos (Visível só no 2º ângulo + Prioridade Climática) */}
                {(selectedTree.v3_triage.has_paralax_divergence || selectedGold) && (
                  <div className="grid gap-2 sm:grid-cols-2">
                    {selectedTree.v3_triage.has_paralax_divergence && (
                      <div className="flex items-start gap-2 rounded border border-[#FDE68A] bg-[#FFFBEB] p-2.5 text-[13px] leading-snug text-[#92400E]">
                        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                        <div>
                          <strong>{APP_LABELS.parallaxLabel}:</strong> frontal{" "}
                          <strong>{formatMacroTriageShort(selectedTree.v3_triage.front_triage)}</strong>{" "}
                          vs lateral{" "}
                          <strong>{formatMacroTriageShort(selectedTree.v3_triage.side_triage)}</strong>.
                        </div>
                      </div>
                    )}

                    {selectedGold &&
                      (() => {
                        const tierStyle = getPriorityTierStyle(selectedGold.priority_tier);
                        return (
                          <div
                            className={`rounded border p-2.5 text-[13px] leading-snug ${tierStyle.card} ${
                              selectedTree.v3_triage.has_paralax_divergence ? "" : "sm:col-span-2"
                            }`}
                          >
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <span className={`font-semibold tabular-nums ${tierStyle.title}`}>
                                Prioridade climática (
                                {formatNumberPtBr(selectedGold.composite_risk_score, 1)}/100,{" "}
                                {formatPriorityTier(selectedGold.priority_tier)})
                              </span>
                              <span className="inline-flex items-center gap-2 tabular-nums text-[#3D4F41]">
                                <span className="inline-flex items-center gap-1">
                                  <Wind className={`h-3.5 w-3.5 ${tierStyle.icon}`} aria-hidden="true" />{" "}
                                  {selectedGold.wind_gust_kmh !== null &&
                                  selectedGold.wind_gust_kmh !== undefined
                                    ? `${formatNumberPtBr(selectedGold.wind_gust_kmh, 1)} km/h`
                                    : "—"}
                                </span>
                                <span className="inline-flex items-center gap-1">
                                  <CloudRain
                                    className={`h-3.5 w-3.5 ${tierStyle.icon}`}
                                    aria-hidden="true"
                                  />{" "}
                                  {selectedGold.precipitation_mm !== null &&
                                  selectedGold.precipitation_mm !== undefined
                                    ? `${formatNumberPtBr(selectedGold.precipitation_mm, 1)} mm`
                                    : "—"}
                                </span>
                              </span>
                            </div>
                            <p className="mt-1 text-[#18241B]">{selectedGold.recommended_action}</p>
                          </div>
                        );
                      })()}
                  </div>
                )}

                {/* Fotos analisadas lado a lado com alternância para foto original */}
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-[13px] font-semibold text-[#4A5D4E]">
                      Ângulos da via ({selectedTree.views.length})
                    </h3>
                    <span className="text-xs text-[#4A5D4E]">
                      Fontes: Prefeitura (GeoSampa e SISGAU) e Google Street View
                    </span>
                  </div>

                  {selectedTree.views.some(
                    (v) => Boolean(v.overlay_url) && Boolean(v.image_url) && v.image_url !== v.overlay_url,
                  ) && (
                    <p className="hidden text-[11px] text-[#4A5D4E] [@media(hover:hover)]:block">
                      Passe o mouse sobre as fotos para ver a original
                    </p>
                  )}

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {selectedTree.views.map((view, idx) => {
                      const v3 = view.v3_analysis;
                      const angleLabel =
                        view.perspective === "center_front" ? "Ângulo frontal" : "Ângulo lateral";
                      const photoStats = getPhotoStats(v3);
                      return (
                        <div
                          key={`${selectedTree.geosampa_id}-${view.perspective}-${idx}`}
                          className="flex flex-col rounded border border-[#D5E0D1] bg-[#FAFBF9] p-2.5 text-[13px]"
                        >
                          <div className="mb-2 flex items-center justify-between gap-1">
                            <span className="font-semibold text-[#18241B]">
                              {angleLabel}{" "}
                              <span className="font-normal text-[#4A5D4E]">
                                ({view.capture_date ? formatDatePtBr(view.capture_date, "monthYear") : "—"})
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

                          {(view.overlay_url || view.image_url) && (
                            <OverlayPhoto
                              key={`${selectedTree.geosampa_id}-${view.perspective}-${view.overlay_url ?? view.image_url ?? idx}`}
                              overlayUrl={view.overlay_url}
                              imageUrl={view.image_url}
                              overlayAlt={`Foto com marcações da árvore #${selectedTree.geosampa_id}, ${angleLabel}`}
                              originalAlt={`Foto original sem marcações da árvore #${selectedTree.geosampa_id}, ${angleLabel}`}
                            />
                          )}

                          {photoStats && (
                            <div className="mt-2 space-y-0.5 text-[13px] text-[#3D4F41]">
                              <p className="font-medium text-[#18241B]">{photoStats.primary}</p>
                              {photoStats.secondary && (
                                <p className="text-xs tabular-nums text-[#4A5D4E]">
                                  {photoStats.secondary}
                                </p>
                              )}
                            </div>
                          )}

                          {view.google_maps_url && (
                            <a
                              href={view.google_maps_url}
                              target="_blank"
                              rel="noreferrer"
                              className="mt-2 inline-flex items-center gap-1 text-[13px] font-medium text-[#1B4332] hover:underline"
                            >
                              <span>Abrir no Google Street View</span>
                              <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
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
                className="flex w-full items-center justify-center gap-1.5 rounded bg-[#1B4332] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#143325]"
              >
                <Sparkles className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>Perguntar ao assistente</span>
              </button>
            </div>
          ) : (
            <div className="flex flex-1 flex-col gap-3.5">
              <div className="border-b border-[#E6ECE3] pb-2.5">
                <h2 className="text-base font-bold text-[#18241B]">
                  Filtros e regiões do mapa
                </h2>
                <p className="mt-0.5 text-[13px] text-[#4A5D4E]">
                  Escolha uma região ou filtre por situação e clique em uma árvore no mapa para
                  inspecionar os dois ângulos.
                </p>
              </div>

              <div className="grid gap-2.5 sm:grid-cols-2">
                <div>
                  <label
                    htmlFor="map-search-input"
                    className="mb-1 block text-[13px] font-medium text-[#4A5D4E]"
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
                      className="w-full rounded border border-[#D5E0D1] bg-[#F6F8F5] py-1.5 pl-8 pr-3 text-[13px] text-[#18241B] placeholder-[#4A5D4E] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
                    />
                  </div>
                </div>

                <div>
                  <label
                    htmlFor="map-triage-select"
                    className="mb-1 block text-[13px] font-medium text-[#4A5D4E]"
                  >
                    Situação identificada
                  </label>
                  <select
                    id="map-triage-select"
                    value={triageFilter}
                    onChange={(e) => setTriageFilter(e.target.value)}
                    className="w-full rounded border border-[#D5E0D1] bg-[#F6F8F5] px-3 py-1.5 text-[13px] text-[#18241B] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
                  >
                    <option value="all">Todas as situações</option>
                    <option value="CRITICO">Risco crítico (todos)</option>
                    <option value="CRITICO_CONFIRMADO">Crítico confirmado nos 2 ângulos</option>
                    <option value="DIVERGENCIA_PARALAXE">{APP_LABELS.parallaxLabel}</option>
                    <option value="PROXIMIDADE">Atenção (proximidade ao vento)</option>
                    <option value="SEGURO">Sem conflito aparente</option>
                  </select>
                </div>
              </div>

              {/* Camada do mapa à esquerda, com prioridade climática à direita */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-y border-[#E6ECE3] py-2.5">
                <div
                  role="group"
                  aria-labelledby="map-layer-label"
                  className="flex flex-wrap items-center gap-1"
                >
                  <span id="map-layer-label" className="mr-1 text-[13px] font-medium text-[#4A5D4E]">
                    Camada:
                  </span>
                  {(Object.keys(BASE_TILES) as BaseMapKind[]).map((kind) => (
                    <button
                      key={kind}
                      type="button"
                      aria-pressed={baseMap === kind}
                      onClick={() => setBaseMap(kind)}
                      className={`rounded border px-2.5 py-1 text-[13px] font-medium transition-colors ${
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
                  aria-pressed={onlyGold}
                  onClick={() => setOnlyGold((v) => !v)}
                  className={`rounded border px-2.5 py-1 text-[13px] font-medium transition-colors ${
                    onlyGold
                      ? "border-[#1B4332] bg-[#1B4332] text-white"
                      : "border-[#D5E0D1] bg-white text-[#2D4234] hover:bg-[#EEF4EC]"
                  }`}
                >
                  Com prioridade climática ({Object.keys(data.gold_scores_by_tree).length})
                </button>
              </div>

              {/* Lista de regiões monitoradas */}
              <div className="flex flex-1 flex-col gap-2">
                <div className="flex items-center justify-between text-[13px]">
                  <span className="font-semibold text-[#18241B]">Regiões monitoradas</span>
                  {regionFilter !== "all" ? (
                    <button
                      type="button"
                      onClick={() => setRegionFilter("all")}
                      className="text-[13px] font-medium text-[#1B4332] hover:underline"
                    >
                      Ver todas ({data.trees.length})
                    </button>
                  ) : (
                    <span className="text-[13px] text-[#4A5D4E]">
                      Clique para centralizar e filtrar
                    </span>
                  )}
                </div>

                <div className="space-y-2">
                  <button
                    type="button"
                    aria-pressed={regionFilter === "all"}
                    onClick={() => setRegionFilter("all")}
                    className={`flex w-full items-center justify-between rounded border px-3 py-2 text-left text-[13px] transition-colors ${
                      regionFilter === "all"
                        ? "border-[#1B4332] bg-[#EEF4EC] text-[#18241B]"
                        : "border-[#D5E0D1] bg-[#FAFBF9] text-[#2D4234] hover:border-[#2D6A4F]"
                    }`}
                  >
                    <span className="font-semibold">Todas as regiões de estudo</span>
                    <span className="tabular-nums text-[13px] text-[#4A5D4E]">
                      {data.stats.total_trees} árvores,{" "}
                      {data.stats.tree_triage_counts.CRITICO ?? 0} críticas
                    </span>
                  </button>

                  {data.stats.district_breakdown.map((reg) => {
                    const active = regionFilter === reg.region_key;
                    return (
                      <button
                        key={reg.region_key}
                        type="button"
                        aria-pressed={active}
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
                          <span className="text-[13px] font-semibold text-[#18241B]">
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

                        <div className="mt-1.5 grid grid-cols-4 gap-2 text-[13px]">
                          <div>
                            <span className="text-xs text-[#4A5D4E]">Avaliadas</span>
                            <p className="font-semibold tabular-nums text-[#18241B]">
                              {reg.total_trees}
                            </p>
                          </div>
                          <div>
                            <span className="text-xs text-[#4A5D4E]">Críticas</span>
                            <p className="font-semibold tabular-nums text-[#991B1B]">
                              {reg.critico_total} ({formatNumberPtBr(reg.critical_rate_pct, 0)}%)
                            </p>
                          </div>
                          <div>
                            <span className="text-xs text-[#4A5D4E]">Vento</span>
                            <p className="font-semibold tabular-nums text-[#92400E]">
                              {reg.proximidade}
                            </p>
                          </div>
                          <div>
                            <span className="text-xs text-[#4A5D4E]">Sem conflito</span>
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
