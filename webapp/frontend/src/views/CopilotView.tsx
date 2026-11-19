import { useEffect, useRef, useState } from "react";
import { AlertCircle, Loader2, RotateCcw, Send, X } from "lucide-react";
import { askAssistant, fetchAssistantStatus, formatFriendlyError } from "../services/api";
import {
  ANALYSIS_GRID_PIXELS,
  APP_LABELS,
  formatDatePtBr,
  formatDistrictName,
  formatDurationSeconds,
  formatPercentPtBr,
  formatRelativePct,
  formatRelevanceLabel,
  formatTriageLabel,
  isInternalChunkId,
} from "../utils/labels";
import type {
  AssistantAskResponse,
  AssistantStatusPayload,
  TreeRecord,
} from "../types/arborgrid";

interface CopilotViewProps {
  trees: TreeRecord[];
  initialTreeId?: string | null;
  initialPrompt?: string | null;
  onConsumeInitialPrompt?: () => void;
  onClearTreeContext?: () => void;
  embedded?: boolean;
}

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: AssistantAskResponse;
}

interface FailedQuestion {
  text: string;
  treeId: string;
  errorMsg: string;
}

const RISK_STYLE: Record<string, { label: string; cls: string }> = {
  emergencia_imediata: {
    label: "Emergência Imediata",
    cls: "border-[#FECACA] bg-[#FEF2F2] text-[#991B1B]",
  },
  urgente_vistoria: {
    label: "Vistoria Urgente",
    cls: "border-[#FDE68A] bg-[#FFFBEB] text-[#92400E]",
  },
  preventivo_rotina: {
    label: "Preventivo / Rotina SP156",
    cls: "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]",
  },
  fora_de_escopo: {
    label: "Fora do Escopo",
    cls: "border-[#D5E0D1] bg-[#F6F8F5] text-[#4A5D4E]",
  },
};

const SCOPE_HELP_TEXT =
  "Posso ajudar com dúvidas sobre poda de árvores em São Paulo, proximidade com a fiação elétrica, canais oficiais (SP156, Enel, Defesa Civil, Bombeiros) e árvores cadastradas no GeoSampa.";

function isMissingValue(val?: string | null): boolean {
  if (!val) return true;
  const norm = val.trim().toLowerCase();
  return !norm || norm === "não consta" || norm === "nao consta";
}

export function CopilotView({
  trees,
  initialTreeId = null,
  initialPrompt = null,
  onConsumeInitialPrompt,
  onClearTreeContext,
  embedded = false,
}: CopilotViewProps) {
  const [statusInfo, setStatusInfo] = useState<AssistantStatusPayload | null>(null);
  const [selectedTreeId, setSelectedTreeId] = useState<string>(initialTreeId || "");
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [failedState, setFailedState] = useState<FailedQuestion | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<AssistantAskResponse | null>(null);

  const autoFiredRef = useRef<string | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const latestTurnRef = useRef<HTMLDivElement>(null);
  const loadingIndicatorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchAssistantStatus()
      .then(setStatusInfo)
      .catch(() => {});
  }, []);

  useEffect(() => {
    setSelectedTreeId(initialTreeId || "");
  }, [initialTreeId]);

  useEffect(() => {
    if (!loading) {
      setElapsedSec(0);
      return;
    }
    setElapsedSec(0);
    const id = window.setInterval(() => {
      setElapsedSec((s) => s + 1);
    }, 1000);
    return () => window.clearInterval(id);
  }, [loading]);

  // Rola para o início da nova resposta (ou indicador de espera) sem esconder o topo
  useEffect(() => {
    if (loading) {
      loadingIndicatorRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    if (turns.length > 0) {
      latestTurnRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [turns.length, loading]);

  const turnsRef = useRef<ChatTurn[]>([]);
  turnsRef.current = turns;

  const clearTreeBinding = () => {
    setSelectedTreeId("");
    onClearTreeContext?.();
  };

  const submitQuestion = async (
    questionText: string,
    gid?: string,
    options?: { skipHistory?: boolean },
  ) => {
    const clean = questionText.trim();
    if (!clean || loading) return;
    const useTreeId = gid !== undefined ? gid : selectedTreeId;
    const historyPayload = options?.skipHistory
      ? []
      : turnsRef.current.slice(-6).map((t) => ({
          role: t.role,
          content: t.text,
        }));
    const userTurnId = `u-${Date.now()}`;
    const userTurn: ChatTurn = {
      id: userTurnId,
      role: "user",
      text: clean,
    };
    setFailedState(null);
    setTurns((prev) => [...prev, userTurn]);
    setInput("");
    setLoading(true);
    try {
      const resp = await askAssistant(clean, useTreeId || null, historyPayload);
      const botTurn: ChatTurn = {
        id: `a-${Date.now()}`,
        role: "assistant",
        text: resp.answer,
        response: resp,
      };
      setTurns((prev) => [...prev, botTurn]);
      setActiveEvidence(resp);
    } catch (err) {
      setTurns((prev) => prev.filter((t) => t.id !== userTurnId));
      setInput(clean);
      setFailedState({
        text: clean,
        treeId: useTreeId,
        errorMsg: formatFriendlyError(err),
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!initialPrompt) {
      autoFiredRef.current = null;
      return;
    }
    if (autoFiredRef.current !== initialPrompt) {
      autoFiredRef.current = initialPrompt;
      void submitQuestion(initialPrompt, initialTreeId || "");
      onConsumeInitialPrompt?.();
    }
  }, [initialPrompt, initialTreeId]);

  const handleSuggestedPromptClick = (prompt: string) => {
    const treeMatch = prompt.match(/#(\d{4,7})\b/);
    if (treeMatch) {
      const mentionedId = treeMatch[1];
      setSelectedTreeId(mentionedId);
      void submitQuestion(prompt, mentionedId, { skipHistory: true });
    } else {
      clearTreeBinding();
      void submitQuestion(prompt, "", { skipHistory: true });
    }
  };

  const criticalSampleTrees = trees.filter((t) => t.v3_triage.consolidated === "CRITICO").slice(0, 25);

  const visibleEvidenceCitations = activeEvidence
    ? activeEvidence.citations.filter((c) => !isInternalChunkId(c.chunk_id))
    : [];
  const isEvidenceOutOfScope = activeEvidence?.structured.nivel_risco === "fora_de_escopo";

  return (
    <div
      className={
        embedded
          ? "space-y-4"
          : "mx-auto max-w-[1680px] space-y-4 px-4 py-4 sm:px-8"
      }
    >
      {/* Cabeçalho compacto fundido com aviso de IA Responsável */}
      {embedded ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-[#D5E0D1] bg-white px-5 py-3.5">
          <div>
            <h2 className="text-base font-bold text-[#18241B]">
              {APP_LABELS.assistantTab} — {APP_LABELS.assistantSubtitle}
            </h2>
            <p className="text-[13px] text-[#4A5D4E]">
              Orientações com base na Lei Municipal SP nº 17.794/2022, ABNT NBR 16246, Manual SVMA e canais oficiais.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded border border-[#C6DBC0] bg-[#EEF4EC] px-2.5 py-1 font-medium text-[#1B4332]">
              {statusInfo?.llm_online
                ? "Assistente IA pronto"
                : "Modo rápido por normas"}
            </span>
          </div>
        </div>
      ) : (
        <div className="rounded-md border border-[#D5E0D1] bg-white px-5 py-3.5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="flex flex-wrap items-baseline gap-2">
                <h1 className="text-lg font-bold text-[#18241B] sm:text-xl">
                  {APP_LABELS.assistantTab}
                </h1>
                <span className="text-[13px] font-medium text-[#4A5D4E]">
                  · {APP_LABELS.assistantSubtitle}
                </span>
              </div>
              <p className="mt-0.5 text-[13px] text-[#4A5D4E]">
                Lei Municipal SP nº 17.794/2022, ABNT NBR 16246-1/2/3, Manual SVMA e canais SP156, Enel, Defesa Civil e Bombeiros.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded border border-[#C6DBC0] bg-[#EEF4EC] px-2.5 py-1 font-medium text-[#1B4332]">
                {statusInfo?.llm_online
                  ? "Assistente IA pronto"
                  : "Modo rápido por normas"}
              </span>
              <span className="rounded border border-[#D5E0D1] bg-[#F6F8F5] px-2.5 py-1 tabular-nums text-[#4A5D4E]">
                {statusInfo?.chunks_count ?? 16} normas e guias oficiais
              </span>
            </div>
          </div>

          <div
            role="note"
            className="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-[#E6ECE3] pt-2 text-[13px] text-[#92400E]"
          >
            <strong className="font-semibold">IA Responsável:</strong>
            <span>
              triagem automática — não substitui vistoria presencial. Em emergência:
            </span>
            <strong className="font-semibold">
              Enel 0800 72 72 196 · Bombeiros 193 · Defesa Civil 199
            </strong>
          </div>
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-12">
        {/* Coluna Esquerda: Pergunta + Resposta */}
        <div
          className={`flex min-w-0 flex-col rounded-md border border-[#D5E0D1] bg-white lg:col-span-7 ${
            embedded ? "h-[520px]" : "h-[calc(100vh-260px)] min-h-[440px]"
          }`}
        >
          <div className="space-y-2.5 border-b border-[#E6ECE3] p-3.5">
            <div className="flex flex-wrap items-center gap-2">
              <label
                htmlFor="assistant-tree-select"
                className="text-[13px] font-medium text-[#4A5D4E]"
              >
                Árvore cadastrada:
              </label>
              <select
                id="assistant-tree-select"
                value={selectedTreeId}
                onChange={(e) => {
                  const next = e.target.value;
                  setSelectedTreeId(next);
                  if (!next) onClearTreeContext?.();
                }}
                className="min-w-0 max-w-full flex-1 rounded border border-[#D5E0D1] bg-[#F6F8F5] px-3 py-1.5 text-[13px] text-[#18241B] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
              >
                <option value="">Sem árvore específica (dúvida geral)</option>
                {selectedTreeId && !criticalSampleTrees.some((t) => t.geosampa_id === selectedTreeId) && (
                  <option value={selectedTreeId}>Árvore selecionada #{selectedTreeId}</option>
                )}
                {criticalSampleTrees.map((t) => (
                  <option key={t.geosampa_id} value={t.geosampa_id}>
                    #{t.geosampa_id} — {formatDistrictName(t.distrito)} ({formatTriageLabel(t.v3_triage.raw_consolidated)})
                  </option>
                ))}
              </select>
            </div>

            {selectedTreeId && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded border border-[#C6DBC0] bg-[#EEF4EC] px-2.5 py-1 text-[13px] font-medium text-[#1B4332]">
                  <span>Pergunta sobre a árvore #{selectedTreeId}</span>
                  <button
                    type="button"
                    onClick={clearTreeBinding}
                    aria-label={`Remover vínculo com a árvore #${selectedTreeId}`}
                    className="inline-flex h-5 w-5 items-center justify-center rounded hover:bg-[#D5E6D0]"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </span>
              </div>
            )}

            <div className="flex flex-wrap gap-1.5">
              {(statusInfo?.suggested_prompts ?? []).map((prompt, idx) => (
                <button
                  key={idx}
                  type="button"
                  disabled={loading}
                  onClick={() => handleSuggestedPromptClick(prompt)}
                  className="rounded border border-[#D5E0D1] bg-[#FAFBF9] px-2.5 py-1 text-left text-[13px] text-[#2D4234] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC] disabled:opacity-60"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          <div
            ref={chatContainerRef}
            aria-live="polite"
            className="flex-1 space-y-4 overflow-y-auto bg-[#FAFBF9] p-4"
          >
            {turns.length === 0 && !loading && !failedState && (
              <div className="flex h-full flex-col items-center justify-center text-center text-[#4A5D4E]">
                <p className="text-sm font-semibold text-[#18241B]">
                  Descreva a situação observada na rua ou clique em uma pergunta sugerida acima
                </p>
                <p className="mt-1.5 max-w-md text-[13px] leading-relaxed">
                  A resposta indica o nível de risco, o órgão responsável, o telefone ou portal
                  correto (SP156, Enel, Defesa Civil ou Bombeiros) e as normas aplicáveis.
                </p>
              </div>
            )}

            {turns.map((turn, idx) => {
              const isLatest = idx === turns.length - 1;
              if (turn.role === "user") {
                return (
                  <div
                    key={turn.id}
                    ref={isLatest ? latestTurnRef : undefined}
                    className="flex justify-end"
                  >
                    <div className="max-w-[85%] rounded-md bg-[#1B4332] px-4 py-2.5 text-sm text-white">
                      {turn.text}
                    </div>
                  </div>
                );
              }

              const resp = turn.response;
              const s = resp?.structured;
              const riskMeta = s ? RISK_STYLE[s.nivel_risco] ?? RISK_STYLE.preventivo_rotina : null;
              const isOutOfScope = s?.nivel_risco === "fora_de_escopo";
              const isEmergency = s?.nivel_risco === "emergencia_imediata";
              const isLlmMode = resp?.mode === "qwen3_gguf";
              const modeLabel = isLlmMode
                ? "Análise IA"
                : "Resposta pelas normas (modo rápido)";

              const showOrgao = s && !isOutOfScope && !isMissingValue(s.orgao_responsavel);
              const showCanal = s && !isOutOfScope && !isMissingValue(s.telefone_ou_canal);

              const filteredSources = s && !isOutOfScope
                ? s.fontes.filter((fid) => !isInternalChunkId(fid))
                : [];

              const citationById = new Map(
                (resp?.citations ?? []).map((c) => [c.chunk_id, c]),
              );

              return (
                <div
                  key={turn.id}
                  ref={isLatest ? latestTurnRef : undefined}
                  className="rounded-md border border-[#D5E0D1] bg-white p-4"
                >
                  {s && riskMeta && (
                    <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`rounded border px-2 py-0.5 text-[13px] font-bold ${riskMeta.cls}`}
                        >
                          {riskMeta.label}
                        </span>
                        <span
                          className={`rounded border px-2 py-0.5 text-[13px] font-medium ${
                            isLlmMode
                              ? "border-[#C6DBC0] bg-[#EEF4EC] text-[#1B4332]"
                              : "border-[#D5E0D1] bg-[#F6F8F5] text-[#4A5D4E]"
                          }`}
                        >
                          {modeLabel}
                          {resp?.latency_ms !== undefined
                            ? ` · ${formatDurationSeconds(resp.latency_ms)}`
                            : ""}
                        </span>
                      </div>

                      {!isEmergency && !isOutOfScope && s.pode_podar_por_conta === false && (
                        <span className="text-[13px] font-semibold text-[#991B1B]">
                          Proibida poda por conta própria em área pública
                        </span>
                      )}
                      {!isEmergency && !isOutOfScope && s.pode_podar_por_conta === true && (
                        <span className="text-[13px] font-semibold text-[#92400E]">
                          Área privada: laudo ART + aviso 10 dias úteis
                        </span>
                      )}
                    </div>
                  )}

                  <p className="text-sm leading-relaxed text-[#18241B]">{turn.text}</p>

                  {isOutOfScope && (
                    <p className="mt-2.5 rounded border border-[#E6ECE3] bg-[#F6F8F5] px-3 py-2 text-[13px] text-[#3D4F41]">
                      {SCOPE_HELP_TEXT}
                    </p>
                  )}

                  {(showOrgao || showCanal) && s && (
                    <div
                      className={`mt-3 grid gap-2 rounded border border-[#E6ECE3] bg-[#F6F8F5] p-3 text-[13px] ${
                        showOrgao && showCanal ? "sm:grid-cols-2" : ""
                      }`}
                    >
                      {showOrgao && (
                        <div>
                          <span className="text-[#4A5D4E]">Órgão responsável</span>
                          <p className="font-semibold text-[#18241B]">{s.orgao_responsavel}</p>
                        </div>
                      )}
                      {showCanal && (
                        <div>
                          <span className="text-[#4A5D4E]">Canal / telefone oficial</span>
                          <p className="font-mono font-semibold text-[#1B4332]">
                            {s.telefone_ou_canal}
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {filteredSources.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                      <span className="text-[13px] text-[#4A5D4E]">Fontes normativas:</span>
                      {filteredSources.map((fid) => {
                        const cit = citationById.get(fid);
                        const label = cit?.titulo || fid;
                        return (
                          <span
                            key={fid}
                            className="rounded border border-[#C6DBC0] bg-[#EEF4EC] px-2 py-0.5 text-[13px] font-medium text-[#1B4332]"
                          >
                            {label}
                          </span>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}

            {loading && (
              <div
                ref={loadingIndicatorRef}
                className="rounded-md border border-[#C6DBC0] bg-white p-3.5 text-[13px] text-[#1B4332]"
              >
                <div className="flex items-center gap-2.5 font-semibold">
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
                  <span className="tabular-nums">Analisando… {elapsedSec} s</span>
                </div>
                <p className="mt-1 pl-6 text-[#4A5D4E]">
                  O modelo de IA roda em CPU; a resposta pode levar até 2 minutos.
                </p>
              </div>
            )}

            {failedState && (
              <div
                role="alert"
                className="rounded-md border border-[#FECACA] bg-[#FEF2F2] p-3.5 text-[13px] text-[#991B1B]"
              >
                <div className="flex items-start gap-2.5">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <div className="flex-1 space-y-2">
                    <p className="font-medium text-[#7F1D1D]">{failedState.errorMsg}</p>
                    <button
                      type="button"
                      onClick={() => void submitQuestion(failedState.text, failedState.treeId)}
                      className="inline-flex items-center gap-1.5 rounded bg-[#1B4332] px-3 py-1.5 text-[13px] font-semibold text-white hover:bg-[#143325]"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      <span>Tentar de novo</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submitQuestion(input);
            }}
            className="flex items-center gap-2 border-t border-[#E6ECE3] bg-white p-3.5"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              aria-label="Pergunta para o assistente"
              placeholder="Pergunte sobre poda, fiação caída, SP156, Enel ou informe um número GeoSampa..."
              className="min-w-0 flex-1 rounded border border-[#D5E0D1] bg-[#F6F8F5] px-3.5 py-2 text-sm text-[#18241B] placeholder-[#4A5D4E] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="inline-flex shrink-0 items-center gap-2 rounded bg-[#1B4332] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#143325] disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              <span>{loading ? "Analisando..." : APP_LABELS.assistantSubmit}</span>
            </button>
          </form>
        </div>

        {/* Coluna Direita: Fontes Citadas */}
        <div
          className={`flex min-w-0 flex-col rounded-md border border-[#D5E0D1] bg-white p-4 lg:col-span-5 ${
            embedded ? "h-[520px]" : "h-[calc(100vh-260px)] min-h-[440px]"
          }`}
        >
          <div className="border-b border-[#E6ECE3] pb-2.5">
            <h2 className="text-sm font-bold text-[#18241B]">
              Base de normas consultada e dados da árvore
            </h2>
            <p className="text-[13px] text-[#4A5D4E]">
              Trechos utilizados para fundamentar a resposta
            </p>
          </div>

          {!activeEvidence ? (
            <div className="flex flex-1 flex-col items-center justify-center text-center text-[13px] text-[#4A5D4E]">
              Envie uma pergunta para visualizar os trechos oficiais da legislação e os dados
              cadastrais usados na resposta.
            </div>
          ) : isEvidenceOutOfScope ? (
            <div className="flex flex-1 flex-col items-center justify-center px-4 text-center text-[13px] text-[#4A5D4E]">
              <p className="font-semibold text-[#18241B]">Assunto fora do escopo do assistente</p>
              <p className="mt-1.5 max-w-sm leading-relaxed">{SCOPE_HELP_TEXT}</p>
            </div>
          ) : (
            <div className="mt-3 flex-1 space-y-4 overflow-y-auto pr-1">
              {activeEvidence.tree_context && (
                <div className="rounded border border-[#C6DBC0] bg-[#EEF4EC] p-3.5 text-[13px]">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono font-bold text-[#1B4332]">
                      GeoSampa #{activeEvidence.tree_context.geosampa_id}
                    </span>
                    <span className="rounded border border-[#C6DBC0] bg-white px-2 py-0.5 text-xs font-semibold text-[#18241B]">
                      {formatTriageLabel(activeEvidence.tree_context.triage_consolidated)}
                    </span>
                  </div>
                  <p className="mt-1 text-[#3D4F41]">
                    {formatDistrictName(activeEvidence.tree_context.distrito)} · Subprefeitura{" "}
                    {formatDistrictName(activeEvidence.tree_context.subprefeitura)} · SISGAU:{" "}
                    {formatDatePtBr(activeEvidence.tree_context.sisgau_last_inspection)}
                  </p>
                  <div className="mt-2 grid grid-cols-3 gap-2 text-[13px] tabular-nums text-[#18241B]">
                    <div>
                      Contato direto:{" "}
                      <strong>
                        {activeEvidence.tree_context.contact_copa_pct !== undefined
                          ? formatPercentPtBr(activeEvidence.tree_context.contact_copa_pct)
                          : formatRelativePct(
                              activeEvidence.tree_context.max_contact_direct_px,
                              activeEvidence.tree_context.max_copa_px || ANALYSIS_GRID_PIXELS,
                            )}{" "}
                        da copa
                      </strong>
                    </div>
                    <div>
                      Zona de vento:{" "}
                      <strong>
                        {activeEvidence.tree_context.wind_copa_pct !== undefined
                          ? formatPercentPtBr(activeEvidence.tree_context.wind_copa_pct)
                          : formatRelativePct(
                              activeEvidence.tree_context.max_wind_buffer_px,
                              activeEvidence.tree_context.max_copa_px || ANALYSIS_GRID_PIXELS,
                            )}{" "}
                        da copa
                      </strong>
                    </div>
                    <div>
                      Fiação:{" "}
                      <strong>
                        {activeEvidence.tree_context.wire_frame_pct !== undefined
                          ? formatPercentPtBr(activeEvidence.tree_context.wire_frame_pct)
                          : formatRelativePct(
                              activeEvidence.tree_context.max_wire_px,
                              ANALYSIS_GRID_PIXELS,
                            )}{" "}
                        da foto
                      </strong>
                    </div>
                  </div>
                </div>
              )}

              {visibleEvidenceCitations.length > 0 && (
                <div className="space-y-2.5">
                  <h3 className="text-[13px] font-semibold text-[#4A5D4E]">
                    Trechos normativos ({visibleEvidenceCitations.length})
                  </h3>
                  {visibleEvidenceCitations.map((chunk) => {
                    const rel = formatRelevanceLabel(chunk.score);
                    return (
                      <div
                        key={chunk.chunk_id}
                        className={`rounded border p-3.5 text-[13px] ${
                          chunk.cited
                            ? "border-[#2D6A4F] bg-[#EEF4EC]"
                            : "border-[#D5E0D1] bg-[#FAFBF9]"
                        }`}
                      >
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <p className="font-semibold text-[#18241B]">{chunk.titulo}</p>
                          <div className="flex shrink-0 items-center gap-1.5">
                            <span
                              className={`rounded border px-2 py-0.5 text-xs font-medium ${rel.cls}`}
                            >
                              {rel.label}
                            </span>
                            {chunk.cited && (
                              <span className="rounded border border-[#2D6A4F] bg-white px-2 py-0.5 text-xs font-semibold text-[#1B4332]">
                                Citado
                              </span>
                            )}
                          </div>
                        </div>
                        <p className="mt-0.5 text-xs text-[#4A5D4E]">{chunk.fonte}</p>
                        <p className="mt-1.5 leading-relaxed text-[#3D4F41]">{chunk.conteudo}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
