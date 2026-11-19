import { useMemo, useRef, useState } from "react";
import { ArrowRight, ChevronDown, ChevronUp, RefreshCw } from "lucide-react";
import { CroppedOverlayImage } from "../components/CroppedOverlayImage";
import type { AppTab } from "../components/Navbar";
import type { GoldRiskRecord, MapBootstrapPayload } from "../types/arborgrid";
import {
  APP_LABELS,
  formatDatePtBr,
  formatDistrictName,
  formatNumberPtBr,
  formatPercentPtBr,
  formatPriorityTier,
} from "../utils/labels";

interface HomeBIViewProps {
  data: MapBootstrapPayload;
  onNavigate: (
    tab: AppTab,
    options?: { geosampaId?: string; regionKey?: string; prompt?: string },
  ) => void;
  onSeedGold: () => Promise<void>;
}

const PRIORITY_BADGE: Record<string, string> = {
  URGENTE: "bg-[#FEF2F2] text-[#991B1B] border-[#FECACA]",
  ALTA: "bg-[#FFFBEB] text-[#92400E] border-[#FDE68A]",
  MEDIA: "bg-[#EEF4EC] text-[#1B4332] border-[#C6DBC0]",
  BAIXA: "bg-[#F4F7F4] text-[#4A5D4E] border-[#DCE5D8]",
};

const PRIORITY_BAR_COLOR: Record<string, string> = {
  URGENTE: "bg-[#dc2626]",
  ALTA: "bg-[#d97706]",
  MEDIA: "bg-[#2D6A4F]",
  BAIXA: "bg-[#6B7D70]",
};

function ScoreBar({ score, tier }: { score: number; tier: string }) {
  const clamped = Math.min(100, Math.max(0, score));
  const barColor = PRIORITY_BAR_COLOR[tier] ?? PRIORITY_BAR_COLOR.BAIXA;
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
  const [showAllGold, setShowAllGold] = useState(false);
  const goldSectionRef = useRef<HTMLElement | null>(null);

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

  // Seleciona árvore crítica confirmada nos dois ângulos com overlays disponíveis para o hero
  const heroEvidenceTree = useMemo(() => {
    const hasTwoOverlays = (t: (typeof trees)[number]) =>
      t.views.filter((v) => Boolean(v.overlay_url)).length >= 2;

    const topGoldId = gold_records[0]?.geosampa_id;
    if (topGoldId) {
      const fromGold = trees.find(
        (t) =>
          t.geosampa_id === topGoldId &&
          t.v3_triage.raw_consolidated === "CRITICO_CONFIRMADO" &&
          hasTwoOverlays(t),
      );
      if (fromGold) return fromGold;
    }

    return (
      trees.find(
        (t) => t.v3_triage.raw_consolidated === "CRITICO_CONFIRMADO" && hasTwoOverlays(t),
      ) ??
      trees.find((t) => hasTwoOverlays(t)) ??
      null
    );
  }, [trees, gold_records]);

  const heroOverlayViews = heroEvidenceTree
    ? heroEvidenceTree.views.filter((v) => Boolean(v.overlay_url)).slice(0, 2)
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
  const visibleGoldRecords = showAllGold ? gold_records : gold_records.slice(0, 12);

  const handleSeed = async () => {
    setSeeding(true);
    try {
      await onSeedGold();
    } finally {
      setSeeding(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1680px] space-y-8 px-4 py-6 sm:px-8">
      {/* 1. Hero com a evidência e indicadores principais (Cartão com borda/sombra) */}
      <section className="rounded-md border border-[#D5E0D1] bg-white p-5 shadow-sm sm:p-7">
        <div className="grid gap-6 lg:grid-cols-12 lg:items-center">
          <div className="space-y-4 lg:col-span-7">
            <p className="text-[13px] font-semibold text-[#2D6A4F]">
              Inteligência urbana · São Paulo
            </p>
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
              agir antes das tempestades — indicando o canal oficial correto para gestores e
              moradores.
            </p>
            <div className="flex flex-wrap gap-3 pt-1">
              <button
                type="button"
                onClick={() => onNavigate("mapa")}
                className="inline-flex items-center gap-2 rounded bg-[#1B4332] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#143325]"
              >
                <span>Abrir Mapa de Vistorias</span>
                <ArrowRight className="h-4 w-4" />
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
                className="inline-flex items-center gap-2 rounded border border-[#D5E0D1] bg-[#F6F8F5] px-4 py-2.5 text-sm font-medium text-[#2D4234] transition-colors hover:bg-[#E7EFE4]"
              >
                <span>Abrir {APP_LABELS.assistantTab}</span>
              </button>
            </div>
          </div>

          {/* Evidência visual à direita: duas fotos com tarja de debug cortada */}
          {heroEvidenceTree && heroOverlayViews.length >= 2 && (
            <div className="rounded-md border border-[#E6ECE3] bg-[#FAFBF9] p-3.5 lg:col-span-5">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <span className="rounded border border-[#FECACA] bg-[#FEF2F2] px-2 py-0.5 text-xs font-semibold text-[#991B1B]">
                  Crítico confirmado nos dois ângulos
                </span>
                <button
                  type="button"
                  onClick={() => onNavigate("mapa", { geosampaId: heroEvidenceTree.geosampa_id })}
                  className="text-xs font-semibold text-[#1B4332] hover:underline"
                >
                  Árvore #{heroEvidenceTree.geosampa_id} ·{" "}
                  {formatDistrictName(heroEvidenceTree.distrito)}
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2.5">
                {heroOverlayViews.map((view, idx) => {
                  const angleLabel =
                    view.perspective === "center_front" ? "Ângulo frontal" : "Ângulo lateral";
                  return (
                    <div key={`${view.perspective}-${idx}`} className="space-y-1">
                      {view.overlay_url && (
                        <CroppedOverlayImage
                          src={view.overlay_url}
                          alt={`Árvore #${heroEvidenceTree.geosampa_id} — ${angleLabel}`}
                        />
                      )}
                      <p className="text-xs font-medium text-[#4A5D4E]">{angleLabel}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* 3 indicadores objetivos com dados reais da API */}
        <div className="mt-6 grid gap-4 border-t border-[#E6ECE3] pt-5 sm:grid-cols-3">
          <div>
            <p className="text-2xl font-bold tabular-nums text-[#991B1B]">{neverInspectedPct}</p>
            <p className="text-sm font-semibold text-[#18241B]">Sem vistoria registrada</p>
            <p className="text-[13px] text-[#4A5D4E]">
              {stats.sisgau_never_inspected_count.toLocaleString("pt-BR")} de{" "}
              {stats.total_trees.toLocaleString("pt-BR")} árvores sem histórico no SISGAU
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

        {/* Top 3 do Gold acima da dobra em 1920x1080 */}
        {topThreeGold.length > 0 && (
          <div className="mt-5 border-t border-[#E6ECE3] pt-4">
            <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-baseline gap-2">
                <h2 className="text-sm font-bold text-[#18241B]">
                  Top 3 da fila de prioridade climática
                </h2>
                <span className="text-xs text-[#4A5D4E]">
                  {isDemoBatch ? "Cenário de demonstração" : "Lote ativo"}
                  {weatherObservedLabel ? ` · Clima de ${weatherObservedLabel}` : ""}
                </span>
              </div>
              <button
                type="button"
                onClick={() =>
                  goldSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
                }
                className="text-[13px] font-semibold text-[#1B4332] hover:underline"
              >
                Ver fila completa ({gold_records.length})
              </button>
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
                          PRIORITY_BADGE[rec.priority_tier] ?? PRIORITY_BADGE.BAIXA
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
                  ? "Priorização para podas e vistorias — cenário de demonstração"
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
                    {weatherObservedLabel ?? "data registrada no lote"}
                  </strong>{" "}
                  (Open-Meteo), aplicando valores de referência de vento e chuva por faixa de
                  prioridade para ordenar vistorias de 0 a 100.
                </>
              ) : (
                <>
                  Combina a proximidade dos galhos com a fiação e dados climáticos do Open-Meteo
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
            <div className="flex flex-wrap items-center justify-between gap-2 text-[13px] text-[#4A5D4E]">
              <span className="tabular-nums">
                Exibindo{" "}
                <strong className="font-semibold text-[#18241B]">
                  {visibleGoldRecords.length} de {gold_records.length}
                </strong>{" "}
                árvores priorizadas
                {weatherObservedLabel ? ` · Condição climática de ${weatherObservedLabel}` : ""}
              </span>
              {gold_records.length > 12 && (
                <button
                  type="button"
                  onClick={() => setShowAllGold((v) => !v)}
                  className="inline-flex items-center gap-1 rounded border border-[#D5E0D1] bg-[#FAFBF9] px-3 py-1 text-[13px] font-semibold text-[#1B4332] hover:border-[#2D6A4F] hover:bg-[#EEF4EC]"
                >
                  {showAllGold ? (
                    <>
                      <span>Mostrar apenas 12</span>
                      <ChevronUp className="h-4 w-4" />
                    </>
                  ) : (
                    <>
                      <span>Ver todas ({gold_records.length})</span>
                      <ChevronDown className="h-4 w-4" />
                    </>
                  )}
                </button>
              )}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[820px] border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-[#D5E0D1] text-[13px] text-[#4A5D4E]">
                    <th className="py-2.5 pr-3 font-semibold">Árvore</th>
                    <th className="px-3 py-2.5 font-semibold">Distrito</th>
                    <th className="px-3 py-2.5 font-semibold">Prioridade</th>
                    <th className="px-3 py-2.5 text-right font-semibold">Índice (0–100)</th>
                    <th className="px-3 py-2.5 text-right font-semibold">Vento / chuva</th>
                    <th className="px-3 py-2.5 font-semibold">Encaminhamento sugerido</th>
                    <th className="py-2.5 pl-3 text-right font-semibold">Ações</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#EEF2EC]">
                  {visibleGoldRecords.map((rec) => (
                    <tr key={rec.geosampa_id} className="align-top hover:bg-[#F6F8F5]">
                      <td className="py-3 pr-3 font-mono text-[13px] font-semibold text-[#18241B]">
                        #{rec.geosampa_id}
                      </td>
                      <td className="px-3 py-3 text-[13px] text-[#3D4F41]">
                        {formatDistrictName(rec.distrito)}
                      </td>
                      <td className="px-3 py-3">
                        <span
                          className={`inline-block rounded border px-2 py-0.5 text-xs font-semibold ${
                            PRIORITY_BADGE[rec.priority_tier] ?? PRIORITY_BADGE.BAIXA
                          }`}
                        >
                          {formatPriorityTier(rec.priority_tier)}
                        </span>
                      </td>
                      <td className="px-3 py-3 text-[13px]">
                        <ScoreBar score={rec.composite_risk_score} tier={rec.priority_tier} />
                      </td>
                      <td className="px-3 py-3 text-right text-[13px] tabular-nums text-[#4A5D4E]">
                        {rec.wind_gust_kmh !== null && rec.wind_gust_kmh !== undefined
                          ? `${formatNumberPtBr(rec.wind_gust_kmh, 1)} km/h`
                          : "—"}{" "}
                        ·{" "}
                        {rec.precipitation_mm !== null && rec.precipitation_mm !== undefined
                          ? `${formatNumberPtBr(rec.precipitation_mm, 1)} mm`
                          : "—"}
                      </td>
                      <td className="px-3 py-3 text-[13px] leading-relaxed text-[#3D4F41]">
                        {rec.recommended_action}
                      </td>
                      <td className="whitespace-nowrap py-3 pl-3 text-right text-[13px]">
                        <div className="inline-flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => onNavigate("mapa", { geosampaId: rec.geosampa_id })}
                            className="font-medium text-[#1B4332] hover:underline"
                          >
                            Ver no mapa
                          </button>
                          <button
                            type="button"
                            onClick={() => onNavigate("assistente", { geosampaId: rec.geosampa_id })}
                            className="rounded border border-[#2D6A4F] px-2.5 py-1 text-xs font-semibold text-[#1B4332] hover:bg-[#EEF4EC]"
                          >
                            Perguntar ao assistente
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
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
            rede subterrânea e atua como referência de controle. Clique em uma linha para abrir a
            região no mapa.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-[#D5E0D1] text-[13px] text-[#4A5D4E]">
                <th className="py-2.5 pr-4 font-semibold">Região / recorte</th>
                <th className="px-3 py-2.5 font-semibold">Rede elétrica</th>
                <th className="px-3 py-2.5 text-right font-semibold">Cadastro GeoSampa</th>
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
                  GeoSampa / SISGAU (Prefeitura de São Paulo):
                </strong>{" "}
                coordenadas oficiais e histórico de vistorias. Nos 4 trechos selecionados, o cadastro
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
                <strong className="font-semibold text-[#18241B]">Open-Meteo:</strong> dados
                meteorológicos de rajadas de vento (km/h) e volume de chuva (mm) por coordenada.
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
                e apenas o olhar lateral mostrou o contato real — evitando subestimar riscos na
                triagem.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Controle negativo na Av. Paulista:
                </strong>{" "}
                {paulistaRow
                  ? `na Av. Paulista, onde a rede é subterrânea, o modelo apontou ${formatPercentPtBr(paulistaRow.critical_rate_pct)} como críticas (esperado: 0,0%). Nas 65 árvores que o modelo nunca viu no treino, a versão atual reduziu os falsos alertas de fiação em 41,8%, mantendo ${paulistaRow.seguro} de ${paulistaRow.total_trees} árvores classificadas como sem conflito.`
                  : "na Av. Paulista, onde a rede é subterrânea, a versão atual do modelo reduziu em 41,8% os falsos alertas de fiação nas 65 árvores que nunca viu no treino."}
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
              ODS 9 · Infraestrutura resiliente
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
              ODS 11 · Cidades sustentáveis
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
              ODS 13 · Ação climática
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
