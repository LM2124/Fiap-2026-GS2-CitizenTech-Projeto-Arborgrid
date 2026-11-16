import { useState } from "react";
import { ArrowRight, RefreshCw } from "lucide-react";
import type { AppTab } from "../components/Navbar";
import type { MapBootstrapPayload } from "../types/arborgrid";
import { formatPriorityTier } from "../utils/labels";

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

export function HomeBIView({ data, onNavigate, onSeedGold }: HomeBIViewProps) {
  const [seeding, setSeeding] = useState(false);
  const { stats, gold_records } = data;

  const critTotal = stats.tree_triage_counts.CRITICO ?? 0;
  const proxTotal = stats.tree_triage_counts.PROXIMIDADE ?? 0;
  const safeTotal = stats.tree_triage_counts.SEGURO ?? 0;
  const parallaxCount = stats.detailed_triage_counts.DIVERGENCIA_PARALAXE ?? 0;
  const critConfirmedCount = stats.detailed_triage_counts.CRITICO_CONFIRMADO ?? 0;
  const neverInspectedPct = stats.total_trees
    ? ((stats.sisgau_never_inspected_count / stats.total_trees) * 100).toFixed(1)
    : "0.0";
  const critSharePct = stats.total_trees
    ? ((critTotal / stats.total_trees) * 100).toFixed(1)
    : "0.0";

  const handleSeed = async () => {
    setSeeding(true);
    try {
      await onSeedGold();
    } finally {
      setSeeding(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1680px] space-y-10 px-4 py-8 sm:px-8">
      {/* 1. Abertura de Produto */}
      <section className="rounded-md border border-[#D5E0D1] bg-white p-6 sm:p-8">
        <div className="grid gap-8 lg:grid-cols-12 lg:items-start">
          <div className="space-y-4 lg:col-span-8">
            <p className="text-xs font-semibold uppercase tracking-wider text-[#2D6A4F]">
              Inteligência Urbana · São Paulo
            </p>
            <h1 className="text-2xl font-bold leading-tight tracking-tight text-[#18241B] sm:text-3xl">
              Mapeamento preventivo de árvores próximas à rede elétrica antes das tempestades
            </h1>
            <p className="max-w-3xl text-sm leading-relaxed text-[#3D4F41] sm:text-base">
              Durante chuvas e rajadas de vento em São Paulo, o contato de galhos com a fiação aérea
              provoca interrupções de energia e riscos nas calçadas. O{" "}
              <strong className="font-semibold text-[#18241B]">ArborGrid SP</strong> reúne dados do
              cadastro público municipal, fotografias ao nível da rua em mais de um ângulo e
              previsão do tempo para apontar quais pontos exigem vistoria e poda preventiva com
              maior urgência — orientando tanto gestores quanto moradores sobre o canal oficial
              correto.
            </p>
            <div className="flex flex-wrap gap-3 pt-2">
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
                <span>Testar Análise em uma Foto</span>
              </button>
              <button
                type="button"
                onClick={() => onNavigate("assistente")}
                className="inline-flex items-center gap-2 rounded border border-[#D5E0D1] bg-[#F6F8F5] px-4 py-2.5 text-sm font-medium text-[#2D4234] transition-colors hover:bg-[#E7EFE4]"
              >
                <span>Consultar Normas e Telefones</span>
              </button>
            </div>
          </div>

          <div className="border-t border-[#E6ECE3] pt-5 lg:col-span-4 lg:border-l lg:border-t-0 lg:pl-7 lg:pt-0">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#4A5D4E]">
              Resumo da Amostra Avaliada
            </h2>
            <dl className="mt-4 space-y-3.5 text-sm">
              <div className="flex items-baseline justify-between border-b border-[#EEF2EC] pb-2.5">
                <dt className="text-[#4A5D4E]">Árvores analisadas em 4 áreas</dt>
                <dd className="text-lg font-bold text-[#18241B]">
                  {stats.total_trees}{" "}
                  <span className="text-xs font-normal text-[#4A5D4E]">
                    ({stats.total_views} fotos)
                  </span>
                </dd>
              </div>
              <div className="flex items-baseline justify-between border-b border-[#EEF2EC] pb-2.5">
                <dt className="text-[#4A5D4E]">Sem vistoria registrada (SISGAU)</dt>
                <dd className="text-lg font-bold text-[#991B1B]">
                  {stats.sisgau_never_inspected_count}{" "}
                  <span className="text-xs font-normal text-[#4A5D4E]">
                    ({neverInspectedPct}%)
                  </span>
                </dd>
              </div>
              <div className="flex items-baseline justify-between border-b border-[#EEF2EC] pb-2.5">
                <dt className="text-[#4A5D4E]">Com contato aparente na fiação</dt>
                <dd className="text-lg font-bold text-[#991B1B]">
                  {critTotal}{" "}
                  <span className="text-xs font-normal text-[#4A5D4E]">({critSharePct}%)</span>
                </dd>
              </div>
              <div className="flex items-baseline justify-between">
                <dt className="text-[#4A5D4E]">Casos revelados só em outro ângulo</dt>
                <dd className="text-lg font-bold text-[#92400E]">{parallaxCount}</dd>
              </div>
            </dl>
          </div>
        </div>
      </section>

      {/* 2. Procedência dos Dados: Fontes Oficiais vs. Análise Própria */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-bold text-[#18241B]">
            Transparência dos dados: fontes externas e análise do estudo
          </h2>
          <p className="text-sm text-[#4A5D4E]">
            Separação clara entre os registros obtidos de bases públicas ou serviços externos e os
            achados produzidos pela avaliação do ArborGrid SP.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded-md border border-[#D5E0D1] bg-white p-6">
            <span className="inline-block rounded border border-[#D5E0D1] bg-[#F6F8F5] px-2.5 py-0.5 text-xs font-semibold text-[#2D4234]">
              Fontes externas e oficiais
            </span>
            <h3 className="mt-3 text-base font-semibold text-[#18241B]">
              Dados cadastrais, fotos de rua, clima e legislação vigente
            </h3>
            <ul className="mt-4 space-y-3 text-sm leading-relaxed text-[#3D4F41]">
              <li>
                <strong className="font-semibold text-[#18241B]">
                  GeoSampa / SISGAU (Prefeitura de São Paulo):
                </strong>{" "}
                coordenadas oficiais e histórico de vistorias. Nos 4 trechos selecionados, o cadastro
                soma mais de 8 mil árvores; na amostra avaliada,{" "}
                <strong>88,2% ({stats.sisgau_never_inspected_count} de {stats.total_trees})</strong>{" "}
                não trazem data de vistoria registrada e as 32 datas presentes vão de 2012 a 2017.
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

          <div className="rounded-md border border-[#C6DBC0] bg-white p-6">
            <span className="inline-block rounded border border-[#C6DBC0] bg-[#EEF4EC] px-2.5 py-0.5 text-xs font-semibold text-[#1B4332]">
              Análise do projeto (amostra de {stats.total_trees} árvores)
            </span>
            <h3 className="mt-3 text-base font-semibold text-[#18241B]">
              Diagnóstico visual multi-ângulo e índice de prioridade
            </h3>
            <ul className="mt-4 space-y-3 text-sm leading-relaxed text-[#3D4F41]">
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Contato aparente com a fiação ({critTotal} árvores):
                </strong>{" "}
                {critConfirmedCount} confirmadas em dois ângulos diferentes da mesma árvore,{" "}
                {parallaxCount} identificadas apenas ao trocar o ângulo da câmera na rua e 2 em
                registro único.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Proximidade para atenção ao vento ({proxTotal} árvores) e sem conflito ({safeTotal}):
                </strong>{" "}
                diferencia galhos que já encostam nos fios daqueles em que a folhagem fica dentro da
                margem de balanço durante ventanias.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Por que dois ângulos da mesma rua importam:
                </strong>{" "}
                em {parallaxCount} situações, uma foto de frente escondia os fios atrás da folhagem, e
                apenas o olhar lateral mostrou o contato real — evitando subestimar riscos em uma
                triagem automática.
              </li>
              <li>
                <strong className="font-semibold text-[#18241B]">
                  Controle negativo na Av. Paulista (-56,1% em falsos alertas):
                </strong>{" "}
                como o corredor da Av. Paulista tem fiação 100% subterrânea, serviu para testar se a
                análise confundia fachadas, postes e marquises com cabos aéreos, classificando 81,1%
                (60 de 74) como livres de conflito.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3. Comparativo das 4 Regiões Analisadas */}
      <section className="rounded-md border border-[#D5E0D1] bg-white p-6">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-2 border-b border-[#E6ECE3] pb-4">
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
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-[#D5E0D1] text-xs uppercase tracking-wide text-[#4A5D4E]">
                <th className="py-2.5 pr-4 font-semibold">Região / Recorte</th>
                <th className="px-3 py-2.5 font-semibold">Rede Elétrica</th>
                <th className="px-3 py-2.5 text-right font-semibold">Cadastro GeoSampa</th>
                <th className="px-3 py-2.5 text-right font-semibold">Amostra Avaliada</th>
                <th className="px-3 py-2.5 text-right font-semibold">Risco Crítico</th>
                <th className="px-3 py-2.5 text-right font-semibold">Só em 2º Ângulo</th>
                <th className="px-3 py-2.5 text-right font-semibold">Atenção Vento</th>
                <th className="py-2.5 pl-4 text-right font-semibold">Sem Conflito</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#EEF2EC]">
              {stats.district_breakdown.map((d) => (
                <tr
                  key={d.region_key}
                  onClick={() => onNavigate("mapa", { regionKey: d.region_key })}
                  className="cursor-pointer transition-colors hover:bg-[#F6F8F5]"
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
                          : "border-[#D5E0D1] bg-[#F6F8F5] text-[#2D4234]"
                      }`}
                    >
                      {d.is_negative_control ? "Subterrânea (Controle)" : "Aérea Convencional"}
                    </span>
                  </td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-[#4A5D4E]">
                    {d.total_geosampa_trees.toLocaleString("pt-BR")}
                  </td>
                  <td className="px-3 py-3.5 text-right font-medium tabular-nums text-[#18241B]">
                    {d.total_trees} ({d.total_views} fotos)
                  </td>
                  <td className="px-3 py-3.5 text-right font-semibold tabular-nums text-[#991B1B]">
                    {d.critico_total} ({d.critical_rate_pct}%)
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

      {/* 4. Fila de Priorização com Clima (Produto BI) */}
      <section className="rounded-md border border-[#D5E0D1] bg-white p-6">
        <div className="mb-5 flex flex-wrap items-start justify-between gap-4 border-b border-[#E6ECE3] pb-4">
          <div className="max-w-2xl space-y-1">
            <h2 className="text-lg font-bold text-[#18241B]">
              Priorização para podas e vistorias com previsão do tempo
            </h2>
            <p className="text-sm text-[#4A5D4E]">
              Para decidir onde agir primeiro antes de um temporal, o produto combina a proximidade
              dos galhos com a fiação e a previsão de rajadas de vento e chuva (Open-Meteo),
              gerando um índice de 0 a 100 e a recomendação de atendimento.
            </p>
          </div>
          <button
            type="button"
            disabled={seeding}
            onClick={handleSeed}
            className="inline-flex items-center gap-2 rounded border border-[#2D6A4F] bg-white px-3.5 py-2 text-xs font-semibold text-[#1B4332] transition-colors hover:bg-[#EEF4EC] disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${seeding ? "animate-spin" : ""}`} />
            <span>
              {gold_records.length > 0
                ? "Atualizar Exemplo de Priorização"
                : "Carregar Exemplo de Priorização Climática"}
            </span>
          </button>
        </div>

        {gold_records.length === 0 ? (
          <div className="rounded border border-dashed border-[#C6DBC0] bg-[#F6F8F5] p-6 text-center">
            <p className="text-sm font-medium text-[#18241B]">
              Nenhum lote de priorização climática carregado nesta sessão.
            </p>
            <p className="mx-auto mt-1 max-w-lg text-xs text-[#4A5D4E]">
              Clique no botão acima para carregar o exemplo demonstrativo com os pontos avaliados e
              suas condições simuladas de vento e chuva.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-[#D5E0D1] text-xs uppercase tracking-wide text-[#4A5D4E]">
                  <th className="py-2.5 pr-3 font-semibold">Árvore</th>
                  <th className="px-3 py-2.5 font-semibold">Distrito</th>
                  <th className="px-3 py-2.5 font-semibold">Prioridade</th>
                  <th className="px-3 py-2.5 text-right font-semibold">Índice</th>
                  <th className="px-3 py-2.5 text-right font-semibold">Vento / Chuva</th>
                  <th className="px-3 py-2.5 font-semibold">Encaminhamento Sugerido</th>
                  <th className="py-2.5 pl-3 text-right font-semibold">Ações</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#EEF2EC]">
                {gold_records.slice(0, 12).map((rec) => (
                  <tr key={rec.geosampa_id} className="align-top hover:bg-[#F6F8F5]">
                    <td className="py-3 pr-3 font-mono text-xs font-semibold text-[#18241B]">
                      #{rec.geosampa_id}
                    </td>
                    <td className="px-3 py-3 text-xs text-[#3D4F41]">{rec.distrito}</td>
                    <td className="px-3 py-3">
                      <span
                        className={`inline-block rounded border px-2 py-0.5 text-[11px] font-semibold ${
                          PRIORITY_BADGE[rec.priority_tier] ?? PRIORITY_BADGE.BAIXA
                        }`}
                      >
                        {formatPriorityTier(rec.priority_tier)}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-right font-semibold tabular-nums text-[#18241B]">
                      {rec.composite_risk_score.toFixed(1)}/100
                    </td>
                    <td className="px-3 py-3 text-right text-xs tabular-nums text-[#4A5D4E]">
                      {rec.wind_gust_kmh ?? "—"} km/h · {rec.precipitation_mm ?? "—"} mm
                    </td>
                    <td className="px-3 py-3 text-xs leading-relaxed text-[#3D4F41]">
                      {rec.recommended_action}
                    </td>
                    <td className="whitespace-nowrap py-3 pl-3 text-right text-xs">
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
                          className="rounded border border-[#2D6A4F] px-2 py-1 font-semibold text-[#1B4332] hover:bg-[#EEF4EC]"
                        >
                          Analisar com IA
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 5. Impacto Urbano e ODS ONU */}
      <section className="rounded-md border border-[#D5E0D1] bg-white p-6">
        <h2 className="text-base font-bold text-[#18241B]">
          Contribuição urbana e Objetivos de Desenvolvimento Sustentável (ONU)
        </h2>
        <div className="mt-4 grid gap-6 border-t border-[#E6ECE3] pt-5 md:grid-cols-3">
          <div className="space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-[#2D6A4F]">
              ODS 9 · Infraestrutura Resiliente
            </p>
            <p className="text-sm font-semibold text-[#18241B]">
              Menos interrupções na rede elétrica aérea
            </p>
            <p className="text-xs leading-relaxed text-[#4A5D4E]">
              Identificar o encontro entre galhos e cabos antes do rompimento ajuda a direcionar
              manutenção e soluções como cabos protegidos nos trechos mais críticos.
            </p>
          </div>
          <div className="space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-[#2D6A4F]">
              ODS 11 · Cidades Sustentáveis
            </p>
            <p className="text-sm font-semibold text-[#18241B]">
              Preservação das árvores e atendimento correto ao cidadão
            </p>
            <p className="text-xs leading-relaxed text-[#4A5D4E]">
              Evita podas drásticas desnecessárias ao mesmo tempo em que informa à população quando
              registrar um pedido preventivo no SP156 e quando ligar para Enel ou Bombeiros.
            </p>
          </div>
          <div className="space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-[#2D6A4F]">
              ODS 13 · Ação Climática
            </p>
            <p className="text-sm font-semibold text-[#18241B]">
              Planejamento preventivo antes de temporais
            </p>
            <p className="text-xs leading-relaxed text-[#4A5D4E]">
              Ao cruzar árvores próximas aos fios com previsões de vento forte e chuva intensa, o
              sistema apoia ações antes da ocorrência de emergências.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
