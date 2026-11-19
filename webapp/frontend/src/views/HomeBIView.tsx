import { useEffect, useMemo, useRef, useState, type FocusEvent } from "react";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Download,
  MapPin,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  Sparkles,
} from "lucide-react";
import { OverlayPhoto } from "../components/OverlayPhoto";
import type { AppTab } from "../components/Navbar";
import type {
  GoldRiskRecord,
  MapBootstrapPayload,
  PriorityTier,
  TreeRecord,
} from "../types/arborgrid";
import {
  APP_LABELS,
  HERO_TREE_IDS,
  formatDatePtBr,
  formatDistrictName,
  formatMacroTriageShort,
  formatNumberPtBr,
  formatPercentPtBr,
  formatPriorityTier,
  formatTriageLabel,
  getPriorityTierStyle,
} from "../utils/labels";

interface HomeBIViewProps {
  data: MapBootstrapPayload;
  onNavigate: (
    tab: AppTab,
    options?: { geosampaId?: string; regionKey?: string; prompt?: string },
  ) => void;
  onSeedGold: () => Promise<void>;
}

type GoldSortKey = "score" | "wind" | "inspection";
type SortDir = "asc" | "desc";

const PAGE_SIZE = 20;
const TIER_ORDER: PriorityTier[] = ["URGENTE", "ALTA", "MEDIA", "BAIXA"];

const PHOTO_TRIAGE_BADGE: Record<string, string> = {
  CRITICO: "border-[#FECACA] bg-[#FEF2F2] text-[#991B1B]",
  PROXIMIDADE: "border-[#FDE68A] bg-[#FFFBEB] text-[#92400E]",
  SEGURO: "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]",
};

function parseInspectionTimestamp(raw?: string | null): number | null {
  if (!raw) return null;
  const text = raw.trim();
  if (!text) return null;
  const ym = /^(\d{4})-(\d{2})$/.exec(text);
  if (ym) {
    return new Date(Number(ym[1]), Number(ym[2]) - 1, 1).getTime();
  }
  const normalized = /^\d{4}-\d{2}-\d{2}Z?$/i.test(text) ? `${text.slice(0, 10)}T12:00:00` : text;
  const ts = new Date(normalized).getTime();
  return Number.isNaN(ts) ? null : ts;
}

function escapeCsvCell(value: string): string {
  const escaped = value.replace(/"/g, '""');
  return /[;"\n\r]/.test(escaped) ? `"${escaped}"` : escaped;
}

function hasTwoOverlays(tree: TreeRecord): boolean {
  return tree.views.filter((v) => Boolean(v.overlay_url)).length >= 2;
}

function sumContactPoints(tree: TreeRecord): number {
  return tree.views.reduce((acc, v) => acc + (v.v3_analysis.contact_points ?? 0), 0);
}

function sumContactDirectArea(tree: TreeRecord): number {
  return tree.views.reduce((acc, v) => acc + (v.v3_analysis.contact_direct_px ?? 0), 0);
}

function selectHeroTrees(trees: TreeRecord[]): TreeRecord[] {
  const byId = new Map(trees.map((t) => [t.geosampa_id, t]));
  const curated = HERO_TREE_IDS.map((id) => byId.get(id)).filter(
    (t): t is TreeRecord => Boolean(t && hasTwoOverlays(t)),
  );
  if (curated.length > 0) {
    return curated;
  }

  const confirmedCritical = trees
    .filter((t) => t.v3_triage.raw_consolidated === "CRITICO_CONFIRMADO" && hasTwoOverlays(t))
    .sort((a, b) => {
      const diffPoints = sumContactPoints(b) - sumContactPoints(a);
      if (diffPoints !== 0) return diffPoints;
      return sumContactDirectArea(b) - sumContactDirectArea(a);
    })
    .slice(0, 5);

  if (confirmedCritical.length > 0) {
    return confirmedCritical;
  }

  return trees.filter(hasTwoOverlays).slice(0, 5);
}

function ScoreBar({ score, tier }: { score: number; tier: string }) {
  const clamped = Math.min(100, Math.max(0, score));
  const barColor = getPriorityTierStyle(tier).bar;
  return (
    <div className="flex items-center justify-end gap-2">
      <div
        className="h-2 w-20 overflow-hidden rounded-full bg-[#E6ECE3]"
        aria-hidden="true"
      >
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${clamped}%` }} />
      </div>
      <span className="w-10 text-right font-semibold tabular-nums text-[#18241B]">
        {formatNumberPtBr(score, 1)}
      </span>
    </div>
  );
}

export function HomeBIView({ data, onNavigate, onSeedGold }: HomeBIViewProps) {
  const [seeding, setSeeding] = useState(false);
  const goldSectionRef = useRef<HTMLElement | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTiers, setSelectedTiers] = useState<PriorityTier[]>([]);
  const [districtFilter, setDistrictFilter] = useState("all");
  const [sortKey, setSortKey] = useState<GoldSortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [currentPage, setCurrentPage] = useState(1);

  const [prefersReducedMotion, setPrefersReducedMotion] = useState<boolean>(() =>
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [userPaused, setUserPaused] = useState<boolean>(() => prefersReducedMotion);
  const [isHeroHovered, setIsHeroHovered] = useState(false);
  const [isHeroFocused, setIsHeroFocused] = useState(false);
  const [activeHeroIndex, setActiveHeroIndex] = useState(0);

  const { stats, gold_records, trees } = data;
  const demoSeedEnabled = data.features?.demo_seed_enabled === true;

  const critTotal = stats.tree_triage_counts.CRITICO ?? 0;
  const proxTotal = stats.tree_triage_counts.PROXIMIDADE ?? 0;
  const safeTotal = stats.tree_triage_counts.SEGURO ?? 0;
  const parallaxCount = stats.detailed_triage_counts.DIVERGENCIA_PARALAXE ?? 0;
  const critConfirmedCount = stats.detailed_triage_counts.CRITICO_CONFIRMADO ?? 0;
  const critSingleViewCount = stats.detailed_triage_counts.CRITICO_MONO_VISADA ?? 0;

  const neverInspectedPct = stats.total_trees
    ? formatPercentPtBr((stats.sisgau_never_inspected_count / stats.total_trees) * 100)
    : "0,0%";
  const critSharePct = stats.total_trees
    ? formatPercentPtBr((critTotal / stats.total_trees) * 100)
    : "0,0%";

  const paulistaRow = stats.district_breakdown.find((d) => d.is_negative_control);

  const heroTrees = useMemo(() => selectHeroTrees(trees), [trees]);
  const safeHeroIndex = heroTrees.length > 0 ? activeHeroIndex % heroTrees.length : 0;
  const currentHeroTree = heroTrees[safeHeroIndex] ?? null;

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
      if (e.matches) {
        setUserPaused(true);
      }
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const shouldAutoRotate =
    heroTrees.length > 1 &&
    !userPaused &&
    !prefersReducedMotion &&
    !isHeroHovered &&
    !isHeroFocused;

  useEffect(() => {
    if (!shouldAutoRotate) return;
    const timer = window.setInterval(() => {
      setActiveHeroIndex((prev) => (prev + 1) % heroTrees.length);
    }, 8000);
    return () => window.clearInterval(timer);
  }, [shouldAutoRotate, heroTrees.length]);

  // Pré-carrega as fotos (com marcações e originais) da próxima árvore em rotação
  useEffect(() => {
    if (heroTrees.length <= 1) return;
    const nextTree = heroTrees[(safeHeroIndex + 1) % heroTrees.length];
    if (!nextTree) return;
    const nextViews = nextTree.views.filter((v) => Boolean(v.overlay_url)).slice(0, 2);
    for (const v of nextViews) {
      if (v.overlay_url) {
        const imgOverlay = new Image();
        imgOverlay.src = v.overlay_url;
      }
      if (v.image_url) {
        const imgOriginal = new Image();
        imgOriginal.src = v.image_url;
      }
    }
  }, [heroTrees, safeHeroIndex]);

  const heroOverlayViews = currentHeroTree
    ? currentHeroTree.views.filter((v) => Boolean(v.overlay_url)).slice(0, 2)
    : [];

  const isDemoBatch =
    gold_records.some(
      (r) => r.is_demo === true || (r.batch_id ?? "").toLowerCase().includes("demo"),
    ) || (data.gold_layer.last_batch_id ?? "").toLowerCase().includes("demo");

  const weatherObservedRaw = gold_records.find((r) => Boolean(r.weather_observed_at))?.weather_observed_at;
  const weatherObservedLabel = weatherObservedRaw
    ? formatDatePtBr(weatherObservedRaw, "dateTime")
    : null;

  const topThreeGold: GoldRiskRecord[] = gold_records.slice(0, 3);

  const treeById = useMemo(
    () => new Map(trees.map((t) => [t.geosampa_id, t])),
    [trees],
  );

  const districtOptions = useMemo(() => {
    const names = new Set<string>();
    for (const rec of gold_records) {
      const formatted = formatDistrictName(rec.distrito);
      if (formatted && formatted !== "—") {
        names.add(formatted);
      }
    }
    return Array.from(names).sort((a, b) => a.localeCompare(b, "pt-BR"));
  }, [gold_records]);

  const baseFilteredGold = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return gold_records.filter((rec) => {
      if (districtFilter !== "all" && formatDistrictName(rec.distrito) !== districtFilter) {
        return false;
      }
      if (q) {
        const districtText = formatDistrictName(rec.distrito).toLowerCase();
        const rawDistrict = (rec.distrito ?? "").toLowerCase();
        if (
          !rec.geosampa_id.toLowerCase().includes(q) &&
          !districtText.includes(q) &&
          !rawDistrict.includes(q)
        ) {
          return false;
        }
      }
      return true;
    });
  }, [gold_records, districtFilter, searchQuery]);

  const tierCounts = useMemo(() => {
    const counts: Record<PriorityTier, number> = {
      URGENTE: 0,
      ALTA: 0,
      MEDIA: 0,
      BAIXA: 0,
    };
    for (const rec of baseFilteredGold) {
      if (rec.priority_tier in counts) {
        counts[rec.priority_tier] += 1;
      }
    }
    return counts;
  }, [baseFilteredGold]);

  const sortedGoldRecords = useMemo(() => {
    const filtered =
      selectedTiers.length === 0
        ? baseFilteredGold
        : baseFilteredGold.filter((rec) => selectedTiers.includes(rec.priority_tier));

    return [...filtered].sort((a, b) => {
      if (sortKey === "score") {
        const diff = a.composite_risk_score - b.composite_risk_score;
        return sortDir === "asc" ? diff : -diff;
      }
      if (sortKey === "wind") {
        const wa = a.wind_gust_kmh;
        const wb = b.wind_gust_kmh;
        if (wa == null && wb == null) return 0;
        if (wa == null) return 1;
        if (wb == null) return -1;
        return sortDir === "asc" ? wa - wb : wb - wa;
      }
      const ta = parseInspectionTimestamp(treeById.get(a.geosampa_id)?.sisgau_last_inspection);
      const tb = parseInspectionTimestamp(treeById.get(b.geosampa_id)?.sisgau_last_inspection);
      const va = ta ?? -Infinity;
      const vb = tb ?? -Infinity;
      if (va === vb) return 0;
      return sortDir === "asc" ? va - vb : vb - va;
    });
  }, [baseFilteredGold, selectedTiers, sortKey, sortDir, treeById]);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, selectedTiers, districtFilter, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sortedGoldRecords.length / PAGE_SIZE));
  const safePage = Math.min(currentPage, totalPages);
  const startIdx = (safePage - 1) * PAGE_SIZE;
  const endIdx = Math.min(sortedGoldRecords.length, startIdx + PAGE_SIZE);
  const pageGoldRecords = sortedGoldRecords.slice(startIdx, endIdx);

  const hasTableFilters =
    searchQuery.trim().length > 0 || selectedTiers.length > 0 || districtFilter !== "all";

  const handleToggleTier = (tier: PriorityTier) => {
    setSelectedTiers((prev) =>
      prev.includes(tier) ? prev.filter((item) => item !== tier) : [...prev, tier],
    );
  };

  const handleClearTableFilters = () => {
    setSearchQuery("");
    setSelectedTiers([]);
    setDistrictFilter("all");
  };

  const handleToggleSort = (key: GoldSortKey) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir(key === "inspection" ? "asc" : "desc");
    }
  };

  const renderSortIcon = (key: GoldSortKey) => {
    if (sortKey !== key) {
      return <ArrowUpDown className="h-3.5 w-3.5 shrink-0 text-[#6B7D70]" aria-hidden="true" />;
    }
    return sortDir === "asc" ? (
      <ArrowUp className="h-3.5 w-3.5 shrink-0 text-[#1B4332]" aria-hidden="true" />
    ) : (
      <ArrowDown className="h-3.5 w-3.5 shrink-0 text-[#1B4332]" aria-hidden="true" />
    );
  };

  const handleExportCsv = () => {
    const header = [
      "Arvore",
      "Distrito",
      "Prioridade",
      "Indice",
      "Vento_kmh",
      "Chuva_mm",
      "Ultima_vistoria",
      "Analise_da_foto",
      "Encaminhamento_sugerido",
    ];
    const rows = sortedGoldRecords.map((rec) => {
      const tree = treeById.get(rec.geosampa_id);
      const lastInspection = formatDatePtBr(tree?.sisgau_last_inspection);
      const photoTriage = tree ? formatTriageLabel(tree.v3_triage.raw_consolidated) : "Sem dados";
      return [
        rec.geosampa_id,
        formatDistrictName(rec.distrito),
        formatPriorityTier(rec.priority_tier),
        formatNumberPtBr(rec.composite_risk_score, 1),
        rec.wind_gust_kmh != null ? formatNumberPtBr(rec.wind_gust_kmh, 1) : "",
        rec.precipitation_mm != null ? formatNumberPtBr(rec.precipitation_mm, 1) : "",
        lastInspection,
        photoTriage,
        rec.recommended_action,
      ]
        .map((cell) => escapeCsvCell(String(cell)))
        .join(";");
    });
    const csv = `\uFEFF${[header.join(";"), ...rows].join("\r\n")}`;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "arborgrid-fila-prioridade.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleSeed = async () => {
    setSeeding(true);
    try {
      await onSeedGold();
    } finally {
      setSeeding(false);
    }
  };

  const heroHasOriginal = heroOverlayViews.some(
    (v) => Boolean(v.image_url) && v.image_url !== v.overlay_url,
  );

  return (
    <div className="mx-auto max-w-[1680px] space-y-6 px-4 py-5 sm:px-8">
      {/* 1. Hero com a evidência e indicadores principais (Cartão com borda/sombra) */}
      <section
        onMouseEnter={() => setIsHeroHovered(true)}
        onMouseLeave={() => setIsHeroHovered(false)}
        onFocus={() => setIsHeroFocused(true)}
        onBlur={(e: FocusEvent<HTMLElement>) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
            setIsHeroFocused(false);
          }
        }}
        className="rounded-md border border-[#D5E0D1] bg-white p-5 shadow-sm sm:p-6"
      >
        <div className="grid gap-5 lg:grid-cols-12 lg:items-center">
          <div className="space-y-3.5 lg:col-span-7">
            <h1 className="text-2xl font-bold leading-tight tracking-tight text-[#18241B] sm:text-3xl lg:text-4xl">
              <span className="tabular-nums text-[#991B1B]">
                {stats.sisgau_never_inspected_count.toLocaleString("pt-BR")} de{" "}
                {stats.total_trees.toLocaleString("pt-BR")}
              </span>{" "}
              árvores sem vistoria registrada
            </h1>
            <p className="max-w-2xl text-sm leading-relaxed text-[#3D4F41] sm:text-base">
              O <strong className="font-semibold text-[#18241B]">ArborGrid SP</strong> cruza o
              cadastro público municipal com fotos de rua em dois ângulos e clima para apontar onde
              agir antes das tempestades, indicando o canal oficial correto para gestores e
              moradores.
            </p>
            <div className="flex flex-wrap gap-3 pt-1">
              <button
                type="button"
                onClick={() => onNavigate("mapa")}
                className="inline-flex items-center gap-2 rounded bg-[#1B4332] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#143325]"
              >
                <span>Abrir Mapa de Vistorias</span>
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </button>
              <button
                type="button"
                onClick={() => onNavigate("visao")}
                className="inline-flex items-center gap-2 rounded border border-[#2D6A4F] bg-white px-4 py-2.5 text-sm font-medium text-[#1B4332] transition-colors hover:bg-[#EEF4EC]"
              >
                <span>Testar análise em uma foto</span>
              </button>
              <button
                type="button"
                onClick={() => onNavigate("assistente")}
                className="inline-flex items-center gap-1.5 rounded border border-[#2D6A4F] bg-white px-4 py-2.5 text-sm font-medium text-[#1B4332] transition-colors hover:bg-[#EEF4EC]"
              >
                <Sparkles className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>Abrir {APP_LABELS.assistantTab}</span>
              </button>
            </div>
          </div>

          {/* Evidência visual à direita: rotação de árvores com hover para ver foto original */}
          {currentHeroTree && heroOverlayViews.length >= 2 && (
            <div className="rounded-md border border-[#E6ECE3] bg-[#FAFBF9] p-3 lg:col-span-5">
              <div
                key={currentHeroTree.geosampa_id}
                className={prefersReducedMotion ? "" : "hero-crossfade"}
              >
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <span className="rounded border border-[#FECACA] bg-[#FEF2F2] px-2 py-0.5 text-xs font-semibold text-[#991B1B]">
                    {formatTriageLabel(currentHeroTree.v3_triage.raw_consolidated)}
                  </span>
                  <button
                    type="button"
                    onClick={() => onNavigate("mapa", { geosampaId: currentHeroTree.geosampa_id })}
                    className="text-xs font-semibold text-[#1B4332] hover:underline"
                  >
                    Árvore #{currentHeroTree.geosampa_id},{" "}
                    {formatDistrictName(currentHeroTree.distrito)}
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2.5">
                  {heroOverlayViews.map((view, idx) => {
                    const angleLabel =
                      view.perspective === "center_front" ? "Ângulo frontal" : "Ângulo lateral";
                    return (
                      <div key={`${currentHeroTree.geosampa_id}-${view.perspective}-${idx}`} className="space-y-1">
                        <OverlayPhoto
                          overlayUrl={view.overlay_url}
                          imageUrl={view.image_url}
                          overlayAlt={`Foto com marcações da árvore #${currentHeroTree.geosampa_id}, ${angleLabel}`}
                          originalAlt={`Foto original sem marcações da árvore #${currentHeroTree.geosampa_id}, ${angleLabel}`}
                        />
                        <p className="text-xs font-medium text-[#4A5D4E]">{angleLabel}</p>
                      </div>
                    );
                  })}
                </div>
              </div>

              {(heroHasOriginal || heroTrees.length > 1) && (
                <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-[#E6ECE3] pt-2">
                  {heroHasOriginal ? (
                    <p className="hidden text-[11px] text-[#4A5D4E] [@media(hover:hover)]:block">
                      Passe o mouse sobre as fotos para ver a original
                    </p>
                  ) : (
                    <span />
                  )}

                  {heroTrees.length > 1 && (
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        aria-pressed={userPaused}
                        aria-label={
                          userPaused
                            ? "Continuar troca automática de árvores"
                            : "Pausar troca automática de árvores"
                        }
                        onClick={() => setUserPaused((v) => !v)}
                        className="inline-flex min-h-[26px] items-center gap-1 rounded border border-[#D5E0D1] bg-white px-2 py-0.5 text-[11px] font-medium text-[#1B4332] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC]"
                      >
                        {userPaused ? (
                          <>
                            <Play className="h-3 w-3 shrink-0" aria-hidden="true" />
                            <span>Continuar</span>
                          </>
                        ) : (
                          <>
                            <Pause className="h-3 w-3 shrink-0" aria-hidden="true" />
                            <span>Pausar</span>
                          </>
                        )}
                      </button>

                      <div
                        role="group"
                        aria-label="Escolher árvore em destaque"
                        className="flex items-center gap-0.5"
                      >
                        {heroTrees.map((tree, idx) => {
                          const isCurrent = idx === safeHeroIndex;
                          return (
                            <button
                              key={tree.geosampa_id}
                              type="button"
                              aria-current={isCurrent ? "true" : undefined}
                              aria-label={`Mostrar árvore #${tree.geosampa_id} (${idx + 1} de ${heroTrees.length})`}
                              onClick={() => setActiveHeroIndex(idx)}
                              className="inline-flex h-6 min-w-[20px] items-center justify-center rounded-full px-0.5 hover:bg-[#E7EFE4]"
                            >
                              <span
                                aria-hidden="true"
                                className={`block h-2 rounded-full transition-all ${
                                  isCurrent ? "w-4 bg-[#1B4332]" : "w-2 bg-[#B7C9B4]"
                                }`}
                              />
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 3 indicadores objetivos com dados reais da API */}
        <div className="mt-4 grid gap-4 border-t border-[#E6ECE3] pt-3.5 sm:grid-cols-3">
          <div>
            <p className="text-2xl font-bold tabular-nums text-[#991B1B]">{neverInspectedPct}</p>
            <p className="text-sm font-semibold text-[#18241B]">Sem vistoria registrada</p>
            <p className="text-[13px] text-[#4A5D4E]">
              {stats.sisgau_never_inspected_count.toLocaleString("pt-BR")} de{" "}
              {stats.total_trees.toLocaleString("pt-BR")} árvores sem histórico de vistoria da
              Prefeitura (SISGAU)
            </p>
          </div>
          <div>
            <p className="text-2xl font-bold tabular-nums text-[#991B1B]">
              {critTotal.toLocaleString("pt-BR")} árvores críticas
            </p>
            <p className="text-sm font-semibold text-[#18241B]">Contato aparente com a fiação</p>
            <p className="text-[13px] text-[#4A5D4E]">
              {critSharePct} da amostra avaliada em 4 territórios ({stats.total_views} fotos)
            </p>
          </div>
          <div>
            <p className="text-2xl font-bold tabular-nums text-[#92400E]">
              {parallaxCount.toLocaleString("pt-BR")} casos
            </p>
            <p className="text-sm font-semibold text-[#18241B]">{APP_LABELS.parallaxLabel}</p>
            <p className="text-[13px] text-[#4A5D4E]">
              Contato escondido pela folhagem na foto frontal e revelado na visada lateral
            </p>
          </div>
        </div>

        {/* Faixa curta Como funciona com 3 passos numerados */}
        <div className="mt-4 border-t border-[#E6ECE3] pt-4">
          <h2 className="mb-2.5 text-base font-bold text-[#18241B]">Como funciona</h2>
          <ol className="grid gap-4 md:grid-cols-3">
            <li className="flex items-start gap-2.5 text-[15px] leading-relaxed text-[#4A5D4E]">
              <span
                aria-hidden="true"
                className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#EEF4EC] text-xs font-bold text-[#1B4332]"
              >
                1
              </span>
              <span>
                <strong className="font-semibold text-[#18241B]">
                  Fotos de rua em dois ângulos:
                </strong>{" "}
                cada árvore do cadastro da Prefeitura é fotografada pelo Street View de frente e de
                lado.
              </span>
            </li>
            <li className="flex items-start gap-2.5 text-[15px] leading-relaxed text-[#4A5D4E]">
              <span
                aria-hidden="true"
                className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#EEF4EC] text-xs font-bold text-[#1B4332]"
              >
                2
              </span>
              <span>
                <strong className="font-semibold text-[#18241B]">Visão computacional:</strong> um
                modelo treinado pela equipe marca, pixel a pixel, a copa (verde) e a fiação
                (amarelo); onde as duas se encostam fica vermelho.{" "}
                <button
                  type="button"
                  onClick={() => onNavigate("visao")}
                  className="font-semibold text-[#1B4332] underline underline-offset-2 hover:text-[#143325]"
                >
                  Testar a análise em uma foto
                </button>
              </span>
            </li>
            <li className="flex items-start gap-2.5 text-[15px] leading-relaxed text-[#4A5D4E]">
              <span
                aria-hidden="true"
                className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#EEF4EC] text-xs font-bold text-[#1B4332]"
              >
                3
              </span>
              <span>
                <strong className="font-semibold text-[#18241B]">Fila de prioridade:</strong> o
                resultado cruza com o tempo sem vistoria e a previsão de vento das próximas 48 horas.
              </span>
            </li>
          </ol>
        </div>
      </section>

      {/* 2. Fila de Priorização com Clima (Cartão principal de produto BI com borda/sombra) */}
      <section
        id="fila-priorizacao"
        ref={goldSectionRef}
        className="rounded-md border border-[#D5E0D1] bg-white p-6 shadow-sm"
      >
        <div className="mb-4 flex flex-wrap items-start justify-between gap-4 border-b border-[#E6ECE3] pb-4">
          <div className="max-w-3xl space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-bold text-[#18241B]">
                {isDemoBatch
                  ? "Priorização para podas e vistorias (cenário de demonstração)"
                  : "Priorização para podas e vistorias com dados climáticos"}
              </h2>
              {isDemoBatch && (
                <span className="rounded border border-[#FDE68A] bg-[#FFFBEB] px-2 py-0.5 text-xs font-semibold text-[#92400E]">
                  Cenário de demonstração
                </span>
              )}
            </div>
            <p className="text-sm text-[#4A5D4E]">
              {isDemoBatch ? (
                <>
                  Cenário de demonstração baseado em condição climática de{" "}
                  <strong className="font-semibold text-[#18241B]">
                    {weatherObservedLabel ?? "última leitura climática"}
                  </strong>{" "}
                  (serviço meteorológico Open-Meteo), aplicando valores de referência de vento e chuva por faixa de
                  prioridade para ordenar vistorias de 0 a 100.
                </>
              ) : (
                <>
                  Combina a proximidade dos galhos com a fiação e dados climáticos do serviço Open-Meteo
                  {weatherObservedLabel ? ` (${weatherObservedLabel})` : ""} para gerar o índice de
                  prioridade de 0 a 100.
                </>
              )}
            </p>
          </div>

          {demoSeedEnabled && (
            <button
              type="button"
              disabled={seeding}
              onClick={handleSeed}
              className="inline-flex items-center gap-2 rounded border border-[#2D6A4F] bg-white px-3.5 py-2 text-xs font-semibold text-[#1B4332] transition-colors hover:bg-[#EEF4EC] disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${seeding ? "animate-spin" : ""}`} />
              <span>
                {gold_records.length > 0
                  ? "Atualizar cenário de demonstração"
                  : "Carregar cenário de demonstração"}
              </span>
            </button>
          )}
        </div>

        {/* Top 3 da fila de prioridade climática */}
        {topThreeGold.length > 0 && (
          <div className="mb-4">
            <div className="mb-2.5 flex flex-wrap items-baseline gap-2">
              <h3 className="text-sm font-bold text-[#18241B]">
                Top 3 da fila de prioridade climática
              </h3>
              <span className="text-xs text-[#4A5D4E]">
                {isDemoBatch ? "Cenário de demonstração" : "Monitoramento ativo"}
                {weatherObservedLabel ? `, clima previsto para ${weatherObservedLabel}` : ""}
              </span>
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              {topThreeGold.map((rec) => (
                <div
                  key={rec.geosampa_id}
                  className="flex items-center justify-between gap-3 rounded border border-[#E6ECE3] bg-[#FAFBF9] px-3.5 py-2.5"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs font-bold text-[#18241B]">
                        #{rec.geosampa_id}
                      </span>
                      <span
                        className={`rounded border px-1.5 py-0.5 text-xs font-semibold ${
                          getPriorityTierStyle(rec.priority_tier).badge
                        }`}
                      >
                        {formatPriorityTier(rec.priority_tier)}
                      </span>
                    </div>
                    <p className="mt-0.5 text-xs text-[#4A5D4E]">
                      {formatDistrictName(rec.distrito)}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <ScoreBar score={rec.composite_risk_score} tier={rec.priority_tier} />
                    <button
                      type="button"
                      onClick={() => onNavigate("mapa", { geosampaId: rec.geosampa_id })}
                      className="text-xs font-semibold text-[#1B4332] hover:underline"
                    >
                      Ver no mapa
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {gold_records.length === 0 ? (
          <div className="rounded border border-dashed border-[#C6DBC0] bg-[#F6F8F5] p-6 text-center">
            <p className="text-sm font-medium text-[#18241B]">
              Nenhum registro de priorização climática disponível no momento.
            </p>
            {demoSeedEnabled && (
              <p className="mx-auto mt-1 max-w-lg text-[13px] text-[#4A5D4E]">
                Use o botão acima para carregar o cenário de demonstração com condições de vento e
                chuva por faixa.
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            {/* Barra de ferramentas da fila de trabalho */}
            <div className="flex flex-col gap-3 rounded-md border border-[#E6ECE3] bg-[#FAFBF9] p-3">
              <div className="flex flex-wrap items-center justify-between gap-2.5">
                <div className="flex flex-wrap items-center gap-2">
                  <div className="relative">
                    <Search
                      className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-[#4A5D4E]"
                      aria-hidden="true"
                    />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      aria-label="Buscar por número da árvore ou distrito"
                      placeholder="Buscar árvore ou distrito..."
                      className="w-56 rounded border border-[#D5E0D1] bg-white py-1.5 pl-8 pr-3 text-[13px] text-[#18241B] placeholder-[#4A5D4E] focus:border-[#2D6A4F] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
                    />
                  </div>

                  <select
                    value={districtFilter}
                    onChange={(e) => setDistrictFilter(e.target.value)}
                    aria-label="Filtrar por distrito"
                    className="rounded border border-[#D5E0D1] bg-white px-2.5 py-1.5 text-[13px] text-[#18241B] focus:border-[#2D6A4F] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
                  >
                    <option value="all">Todos os distritos</option>
                    {districtOptions.map((dist) => (
                      <option key={dist} value={dist}>
                        {dist}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[13px] font-medium tabular-nums text-[#4A5D4E]">
                    {sortedGoldRecords.length.toLocaleString("pt-BR")}{" "}
                    {sortedGoldRecords.length === 1 ? "árvore" : "árvores"}
                    {weatherObservedLabel ? `, clima de ${weatherObservedLabel}` : ""}
                  </span>

                  {hasTableFilters && (
                    <button
                      type="button"
                      onClick={handleClearTableFilters}
                      className="inline-flex items-center gap-1 rounded border border-[#2D6A4F] bg-[#EEF4EC] px-2.5 py-1 text-xs font-semibold text-[#1B4332] hover:bg-[#DCEAD8]"
                    >
                      <RotateCcw className="h-3 w-3 shrink-0" aria-hidden="true" />
                      <span>Limpar filtros</span>
                    </button>
                  )}

                  <button
                    type="button"
                    disabled={sortedGoldRecords.length === 0}
                    onClick={handleExportCsv}
                    className="inline-flex items-center gap-1.5 rounded border border-[#D5E0D1] bg-white px-3 py-1 text-xs font-semibold text-[#1B4332] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC] disabled:opacity-50"
                  >
                    <Download className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    <span>Baixar lista (CSV)</span>
                  </button>
                </div>
              </div>

              <div
                role="group"
                aria-label="Filtrar por prioridade"
                className="flex flex-wrap items-center gap-1.5"
              >
                <span className="mr-1 text-xs font-medium text-[#4A5D4E]">Prioridade:</span>
                {TIER_ORDER.map((tier) => {
                  const active = selectedTiers.includes(tier);
                  const style = getPriorityTierStyle(tier);
                  return (
                    <button
                      key={tier}
                      type="button"
                      aria-pressed={active}
                      onClick={() => handleToggleTier(tier)}
                      className={`inline-flex items-center gap-1 rounded border px-2.5 py-1 text-xs font-semibold transition-colors ${
                        active
                          ? style.badge
                          : "border-[#D5E0D1] bg-white text-[#3D4F41] hover:border-[#2D6A4F] hover:bg-[#EEF4EC]"
                      }`}
                    >
                      <span>{formatPriorityTier(tier)}</span>
                      <span className="tabular-nums opacity-80">
                        ({tierCounts[tier].toLocaleString("pt-BR")})
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {sortedGoldRecords.length === 0 ? (
              <div className="rounded border border-dashed border-[#C6DBC0] bg-[#F6F8F5] p-6 text-center">
                <p className="text-sm font-medium text-[#18241B]">
                  Nenhuma árvore encontrada com esses filtros.
                </p>
                <button
                  type="button"
                  onClick={handleClearTableFilters}
                  className="mt-3 inline-flex items-center gap-1.5 rounded bg-[#1B4332] px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-[#143325]"
                >
                  <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>Limpar filtros</span>
                </button>
              </div>
            ) : (
              <>
                <div className="max-h-[640px] overflow-auto rounded border border-[#E6ECE3]">
                  <table className="w-full min-w-[960px] border-collapse text-left text-sm">
                    <caption className="sr-only">
                      Fila de trabalho para podas e vistorias priorizadas por risco visual e previsão climática
                    </caption>
                    <thead className="sticky top-0 z-10 border-b border-[#D5E0D1] bg-[#FAFBF9] text-[13px] text-[#4A5D4E]">
                      <tr>
                        <th scope="col" className="py-2.5 pl-3 pr-3 font-semibold">
                          Árvore
                        </th>
                        <th scope="col" className="px-3 py-2.5 font-semibold">
                          Distrito
                        </th>
                        <th scope="col" className="px-3 py-2.5 font-semibold">
                          Prioridade
                        </th>
                        <th
                          scope="col"
                          aria-sort={
                            sortKey === "score"
                              ? sortDir === "asc"
                                ? "ascending"
                                : "descending"
                              : "none"
                          }
                          className="px-3 py-2.5 text-right font-semibold"
                        >
                          <button
                            type="button"
                            onClick={() => handleToggleSort("score")}
                            className="inline-flex items-center gap-1 hover:text-[#18241B]"
                          >
                            <span>Índice</span>
                            {renderSortIcon("score")}
                          </button>
                        </th>
                        <th
                          scope="col"
                          aria-sort={
                            sortKey === "wind"
                              ? sortDir === "asc"
                                ? "ascending"
                                : "descending"
                              : "none"
                          }
                          className="px-3 py-2.5 text-right font-semibold"
                        >
                          <button
                            type="button"
                            onClick={() => handleToggleSort("wind")}
                            className="inline-flex items-center gap-1 hover:text-[#18241B]"
                          >
                            <span>Vento e chuva previstos</span>
                            {renderSortIcon("wind")}
                          </button>
                        </th>
                        <th
                          scope="col"
                          aria-sort={
                            sortKey === "inspection"
                              ? sortDir === "asc"
                                ? "ascending"
                                : "descending"
                              : "none"
                          }
                          className="px-3 py-2.5 font-semibold"
                        >
                          <button
                            type="button"
                            onClick={() => handleToggleSort("inspection")}
                            className="inline-flex items-center gap-1 hover:text-[#18241B]"
                          >
                            <span>Última vistoria</span>
                            {renderSortIcon("inspection")}
                          </button>
                        </th>
                        <th scope="col" className="px-3 py-2.5 font-semibold">
                          Análise da foto
                        </th>
                        <th scope="col" className="px-3 py-2.5 font-semibold">
                          Encaminhamento sugerido
                        </th>
                        <th scope="col" className="py-2.5 pl-3 pr-3 text-right font-semibold">
                          Ações
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#EEF2EC]">
                      {pageGoldRecords.map((rec) => {
                        const tree = treeById.get(rec.geosampa_id);
                        const lastInspection = formatDatePtBr(tree?.sisgau_last_inspection);
                        const photoMacro = tree?.v3_triage.consolidated ?? "SEGURO";
                        const photoLabel = tree
                          ? formatMacroTriageShort(tree.v3_triage.consolidated)
                          : "Sem dados";
                        const photoTitle = tree
                          ? formatTriageLabel(tree.v3_triage.raw_consolidated)
                          : undefined;
                        return (
                          <tr key={rec.geosampa_id} className="align-top hover:bg-[#F6F8F5]">
                            <td className="py-3 pl-3 pr-3 font-mono text-[13px] font-semibold text-[#18241B]">
                              #{rec.geosampa_id}
                            </td>
                            <td className="px-3 py-3 text-[13px] text-[#3D4F41]">
                              {formatDistrictName(rec.distrito)}
                            </td>
                            <td className="px-3 py-3">
                              <span
                                className={`inline-block rounded border px-2 py-0.5 text-xs font-semibold ${
                                  getPriorityTierStyle(rec.priority_tier).badge
                                }`}
                              >
                                {formatPriorityTier(rec.priority_tier)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-[13px]">
                              <ScoreBar score={rec.composite_risk_score} tier={rec.priority_tier} />
                            </td>
                            <td className="px-3 py-3 text-right text-[13px] tabular-nums text-[#4A5D4E]">
                              <div>
                                {rec.wind_gust_kmh !== null && rec.wind_gust_kmh !== undefined
                                  ? `${formatNumberPtBr(rec.wind_gust_kmh, 1)} km/h`
                                  : "—"}
                              </div>
                              <div className="text-xs">
                                {rec.precipitation_mm !== null && rec.precipitation_mm !== undefined
                                  ? `${formatNumberPtBr(rec.precipitation_mm, 1)} mm`
                                  : "—"}
                              </div>
                            </td>
                            <td className="whitespace-nowrap px-3 py-3 text-[13px] tabular-nums text-[#3D4F41]">
                              {lastInspection}
                            </td>
                            <td className="px-3 py-3">
                              {tree ? (
                                <span
                                  title={photoTitle}
                                  className={`inline-block rounded border px-2 py-0.5 text-xs font-semibold ${
                                    PHOTO_TRIAGE_BADGE[photoMacro] ?? PHOTO_TRIAGE_BADGE.SEGURO
                                  }`}
                                >
                                  {photoLabel}
                                </span>
                              ) : (
                                <span className="text-xs text-[#4A5D4E]">—</span>
                              )}
                            </td>
                            <td className="px-3 py-3 text-[13px] leading-relaxed text-[#3D4F41]">
                              {rec.recommended_action}
                            </td>
                            <td className="whitespace-nowrap py-3 pl-3 pr-3 text-right text-[13px]">
                              <div className="inline-flex items-center gap-1.5">
                                <button
                                  type="button"
                                  title="Ver no mapa"
                                  aria-label={`Ver árvore #${rec.geosampa_id} no mapa`}
                                  onClick={() => onNavigate("mapa", { geosampaId: rec.geosampa_id })}
                                  className="inline-flex min-h-[30px] items-center gap-1 rounded border border-[#D5E0D1] bg-white px-2 py-1 text-xs font-medium text-[#1B4332] hover:border-[#2D6A4F] hover:bg-[#EEF4EC]"
                                >
                                  <MapPin className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                                  <span className="hidden 2xl:inline">Ver no mapa</span>
                                </button>
                                <button
                                  type="button"
                                  title="Perguntar ao assistente"
                                  aria-label={`Perguntar ao assistente sobre a árvore #${rec.geosampa_id}`}
                                  onClick={() =>
                                    onNavigate("assistente", { geosampaId: rec.geosampa_id })
                                  }
                                  className="inline-flex min-h-[30px] items-center gap-1.5 rounded border border-[#2D6A4F] bg-white px-2 py-1 text-xs font-semibold text-[#1B4332] hover:bg-[#EEF4EC]"
                                >
                                  <Sparkles className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                                  <span className="hidden 2xl:inline">Perguntar ao assistente</span>
                                </button>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Paginação */}
                <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-[13px] text-[#4A5D4E]">
                  <span className="tabular-nums">
                    Mostrando{" "}
                    <strong className="font-semibold text-[#18241B]">
                      {startIdx + 1}–{endIdx}
                    </strong>{" "}
                    de{" "}
                    <strong className="font-semibold text-[#18241B]">
                      {sortedGoldRecords.length.toLocaleString("pt-BR")}
                    </strong>
                  </span>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={safePage <= 1}
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      className="inline-flex min-h-[34px] items-center gap-1 rounded border border-[#D5E0D1] bg-white px-2.5 py-1 text-xs font-semibold text-[#1B4332] hover:border-[#2D6A4F] hover:bg-[#EEF4EC] disabled:opacity-40"
                    >
                      <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
                      <span>Anterior</span>
                    </button>
                    <span className="tabular-nums text-xs font-medium text-[#18241B]">
                      Página {safePage} de {totalPages}
                    </span>
                    <button
                      type="button"
                      disabled={safePage >= totalPages}
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      className="inline-flex min-h-[34px] items-center gap-1 rounded border border-[#D5E0D1] bg-white px-2.5 py-1 text-xs font-semibold text-[#1B4332] hover:border-[#2D6A4F] hover:bg-[#EEF4EC] disabled:opacity-40"
                    >
                      <span>Próxima</span>
                      <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </section>

      {/* 3. Comparativo das 4 Regiões Analisadas (Seção plana com divisor) */}
      <section className="space-y-4 border-t border-[#D5E0D1] pt-8">
        <div>
          <h2 className="text-lg font-bold text-[#18241B]">
            Retrato comparativo das 4 áreas analisadas
          </h2>
          <p className="text-sm text-[#4A5D4E]">
            Pinheiros, Alto da Mooca e Vila Mariana possuem rede elétrica aérea; a Av. Paulista tem
            rede subterrânea e atua como referência de controle. Cada área mostra uma amostra das
            árvores cadastradas, não todas as árvores da região. Clique em uma linha para abrir a
            região no mapa.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-[#D5E0D1] text-[13px] text-[#4A5D4E]">
                <th className="py-2.5 pr-4 font-semibold">Região / recorte</th>
                <th className="px-3 py-2.5 font-semibold">Rede elétrica</th>
                <th className="px-3 py-2.5 text-right font-semibold">Cadastro da Prefeitura (GeoSampa)</th>
                <th className="px-3 py-2.5 text-right font-semibold">Amostra avaliada</th>
                <th className="px-3 py-2.5 text-right font-semibold">Risco crítico</th>
                <th className="px-3 py-2.5 text-right font-semibold">
                  {APP_LABELS.parallaxLabel}
                </th>
                <th className="px-3 py-2.5 text-right font-semibold">Atenção vento</th>
                <th className="py-2.5 pl-4 text-right font-semibold">Sem conflito</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E2EAE0]">
              {stats.district_breakdown.map((d) => (
                <tr
                  key={d.region_key}
                  onClick={() => onNavigate("mapa", { regionKey: d.region_key })}
                  className="cursor-pointer transition-colors hover:bg-white"
                >
                  <td className="py-3.5 pr-4">
                    <div className="font-semibold text-[#1B4332] hover:underline">
                      {d.region_label}
                    </div>
                    <div className="text-xs text-[#4A5D4E]">
                      {d.sisgau_never_inspected_count} de {d.total_trees} sem vistoria registrada no
                      SISGAU
                    </div>
                  </td>
                  <td className="px-3 py-3.5 text-xs">
                    <span
                      className={`inline-block rounded border px-2 py-0.5 font-medium ${
                        d.is_negative_control
                          ? "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]"
                          : "border-[#D5E0D1] bg-white text-[#2D4234]"
                      }`}
                    >
                      {d.is_negative_control ? "Subterrânea (controle)" : "Aérea convencional"}
                    </span>
                  </td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-[#4A5D4E]">
                    {d.total_geosampa_trees.toLocaleString("pt-BR")}
                  </td>
                  <td className="px-3 py-3.5 text-right font-medium tabular-nums text-[#18241B]">
                    {d.total_trees} ({d.total_views} fotos)
                  </td>
                  <td className="px-3 py-3.5 text-right font-semibold tabular-nums text-[#991B1B]">
                    {d.critico_total} ({formatPercentPtBr(d.critical_rate_pct)})
                  </td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-[#92400E]">
                    {d.divergencia_paralaxe}
                  </td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-[#4A5D4E]">
                    {d.proximidade}
                  </td>
                  <td className="py-3.5 pl-4 text-right font-semibold tabular-nums text-[#1B4332]">
                    {d.seguro}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 4. Procedência dos Dados: Fontes Oficiais vs. Análise Própria (Seção plana com divisor) */}
      <section className="space-y-4 border-t border-[#D5E0D1] pt-8">
        <div>
          <h2 className="text-lg font-bold text-[#18241B]">
            Origem dos dados e achados da análise
          </h2>
          <p className="text-sm text-[#4A5D4E]">
            Separação entre os registros obtidos de bases públicas ou serviços externos e os
            resultados produzidos pelo ArborGrid SP.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="space-y-3">
            <span className="inline-block rounded border border-[#D5E0D1] bg-white px-2.5 py-0.5 text-xs font-semibold text-[#2D4234]">
              Fontes externas e oficiais
            </span>
            <h3 className="text-base font-semibold text-[#18241B]">
              Dados cadastrais, fotos de rua, clima e legislação vigente
            </h3>
            <ul className="space-y-2.5 text-sm leading-relaxed text-[#3D4F41]">
              <li>
                <strong className="font-semibold text-[#18241B]">
                  GeoSampa e SISGAU (Prefeitura de São Paulo):
                </strong>{" "}
                cadastro público de localização das árvores (GeoSampa) e histórico oficial de vistorias (SISGAU). Nos 4 trechos selecionados, o cadastro
                soma mais de 8 mil árvores; na amostra avaliada,{" "}
                <strong>
                  {neverInspectedPct} ({stats.sisgau_never_inspected_count} de {stats.total_trees})
                </strong>{" "}
                não trazem data de vistoria registrada e as datas presentes vão de 2012 a 2017.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">Google Street View:</strong>{" "}
                fotografias ao nível da rua capturadas a partir de pontos distintos da via pública
                para observar a copa da árvore em relação aos cabos.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">Open-Meteo:</strong> serviço de dados
                meteorológicos com previsão e leitura de rajadas de vento (km/h) e volume de chuva (mm) por região.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">Normas e canais oficiais:</strong>{" "}
                Lei Municipal SP nº 17.794/2022, Decreto nº 61.859/2022, Manual Técnico de Poda SVMA,
                ABNT NBR 16246-1/2/3 e atendimentos SP156, Enel (0800 72 72 196), Defesa Civil (199)
                e Bombeiros (193).
              </li>
            </ul>
          </div>

          <div className="space-y-3 border-t border-[#E6ECE3] pt-5 md:border-l md:border-t-0 md:pl-8 md:pt-0">
            <span className="inline-block rounded border border-[#C6DBC0] bg-[#EEF4EC] px-2.5 py-0.5 text-xs font-semibold text-[#1B4332]">
              Análise do projeto (amostra de {stats.total_trees} árvores)
            </span>
            <h3 className="text-base font-semibold text-[#18241B]">
              Diagnóstico visual multi-ângulo e índice de prioridade
            </h3>
            <ul className="space-y-2.5 text-sm leading-relaxed text-[#3D4F41]">
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Contato aparente com a fiação ({critTotal} árvores):
                </strong>{" "}
                {critConfirmedCount} confirmadas em dois ângulos diferentes da mesma árvore,{" "}
                {parallaxCount} visíveis só no 2º ângulo e {critSingleViewCount} em registro único.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Proximidade ao vento ({proxTotal} árvores) e sem conflito ({safeTotal}):
                </strong>{" "}
                diferencia galhos que já encostam nos fios daqueles em que a folhagem fica dentro da
                margem de balanço durante ventanias.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Por que dois ângulos da mesma rua importam:
                </strong>{" "}
                em {parallaxCount} situações, uma foto de frente escondia os fios atrás da folhagem,
                e apenas o olhar lateral mostrou o contato real, evitando subestimar riscos na
                triagem.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Controle negativo na Av. Paulista:
                </strong>{" "}
                {paulistaRow
                  ? `Na Av. Paulista a fiação é subterrânea, então o esperado é quase nenhum alerta. Das ${paulistaRow.total_trees} árvores, o modelo atual marca ${paulistaRow.critico_confirmado} como críticas nos dois ângulos, todas em cruzamentos com ruas que ainda têm fiação aérea, e outras ${Math.max(0, paulistaRow.critico_total - paulistaRow.critico_confirmado)} em um ângulo só. A versão anterior marcava 14. Nas 65 árvores que o modelo nunca viu no treino, os falsos alertas de fiação caíram 41,8%.`
                  : "Na Av. Paulista a fiação é subterrânea, então o esperado é quase nenhum alerta. Das 74 árvores, o modelo atual marca 3 como críticas nos dois ângulos, todas em cruzamentos com ruas que ainda têm fiação aérea, e outras 6 em um ângulo só. A versão anterior marcava 14. Nas 65 árvores que o modelo nunca viu no treino, os falsos alertas de fiação caíram 41,8%."}
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. Impacto Urbano e ODS ONU (Seção plana com divisor) */}
      <section className="space-y-4 border-t border-[#D5E0D1] pt-8">
        <h2 className="text-base font-bold text-[#18241B]">
          Contribuição urbana e Objetivos de Desenvolvimento Sustentável (ONU)
        </h2>
        <div className="grid gap-6 md:grid-cols-3">
          <div className="space-y-1.5">
            <p className="text-[13px] font-semibold text-[#2D6A4F]">
              ODS 9: Infraestrutura resiliente
            </p>
            <p className="text-sm font-semibold text-[#18241B]">
              Menos interrupções na rede elétrica aérea
            </p>
            <p className="text-[13px] leading-relaxed text-[#4A5D4E]">
              Identificar o encontro entre galhos e cabos antes do rompimento ajuda a direcionar
              manutenção e soluções como cabos protegidos nos trechos mais críticos.
            </p>
          </div>
          <div className="space-y-1.5">
            <p className="text-[13px] font-semibold text-[#2D6A4F]">
              ODS 11: Cidades sustentáveis
            </p>
            <p className="text-sm font-semibold text-[#18241B]">
              Preservação das árvores e atendimento correto ao cidadão
            </p>
            <p className="text-[13px] leading-relaxed text-[#4A5D4E]">
              Evita podas drásticas desnecessárias ao mesmo tempo em que informa à população quando
              registrar um pedido preventivo no SP156 e quando ligar para Enel ou Bombeiros.
            </p>
          </div>
          <div className="space-y-1.5">
            <p className="text-[13px] font-semibold text-[#2D6A4F]">
              ODS 13: Ação climática
            </p>
            <p className="text-sm font-semibold text-[#18241B]">
              Planejamento preventivo antes de temporais
            </p>
            <p className="text-[13px] leading-relaxed text-[#4A5D4E]">
              Ao cruzar árvores próximas aos fios com previsões de vento forte e chuva intensa, o
              sistema apoia ações antes da ocorrência de emergências.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
