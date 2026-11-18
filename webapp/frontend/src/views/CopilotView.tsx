import { useEffect, useRef, useState } from "react";
import { Loader2, Send } from "lucide-react";
import { askAssistant, fetchAssistantStatus } from "../services/api";
import {
  ANALYSIS_GRID_PIXELS,
  formatDurationSeconds,
  formatRelativePct,
  formatTriageLabel,
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
  embedded?: boolean;
}

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: AssistantAskResponse;
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

export function CopilotView({
  trees,
  initialTreeId = null,
  initialPrompt = null,
  onConsumeInitialPrompt,
  embedded = false,
}: CopilotViewProps) {
  const [statusInfo, setStatusInfo] = useState<AssistantStatusPayload | null>(null);
  const [selectedTreeId, setSelectedTreeId] = useState<string>(initialTreeId || "");
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [activeEvidence, setActiveEvidence] = useState<AssistantAskResponse | null>(null);

  const autoFiredRef = useRef<string | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchAssistantStatus()
      .then(setStatusInfo)
      .catch(() => {});
  }, []);

  useEffect(() => {
    setSelectedTreeId(initialTreeId || "");
  }, [initialTreeId]);

  useEffect(() => {
    const el = chatContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns, loading]);

  const turnsRef = useRef<ChatTurn[]>([]);
  turnsRef.current = turns;

  const submitQuestion = async (questionText: string, gid?: string) => {
    const clean = questionText.trim();
    if (!clean) return;
    const useTreeId = gid !== undefined ? gid : selectedTreeId;
    const historyPayload = turnsRef.current.slice(-6).map((t) => ({
      role: t.role,
      content: t.text,
    }));
    const userTurn: ChatTurn = {
      id: `u-${Date.now()}`,
      role: "user",
      text: clean,
    };
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
      setTurns((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "assistant",
          text: `Erro ao realizar a Análise IA: ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
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

  const criticalSampleTrees = trees.filter((t) => t.v3_triage.consolidated === "CRITICO").slice(0, 25);

  return (
    <div
      className={
        embedded
          ? "space-y-4"
          : "mx-auto max-w-[1680px] space-y-5 px-4 py-8 sm:px-8"
      }
    >
      {/* Cabeçalho */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-md border border-[#D5E0D1] bg-white p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-bold text-[#18241B]">
            Consulta de Normas e Análise IA
          </h1>
          <p className="max-w-3xl text-sm text-[#4A5D4E]">
            Orientações baseadas na{" "}
            <strong className="font-semibold text-[#18241B]">Lei Municipal SP nº 17.794/2022</strong>
            , <strong className="font-semibold text-[#18241B]">ABNT NBR 16246-1/2/3</strong>,{" "}
            <strong className="font-semibold text-[#18241B]">Manual Técnico SVMA</strong> e canais
            oficiais (<strong className="font-semibold text-[#18241B]">SP156, Enel 24h, Defesa Civil 199 e Bombeiros 193</strong>).
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded border border-[#C6DBC0] bg-[#EEF4EC] px-3 py-1 font-medium text-[#1B4332]">
            {statusInfo?.llm_online
              ? "Assistente IA conectado"
              : "Consulta normativa ativa"}
          </span>
          <span className="rounded border border-[#D5E0D1] bg-[#F6F8F5] px-3 py-1 text-[#4A5D4E]">
            {statusInfo?.chunks_count ?? 16} normas e guias oficiais
          </span>
        </div>
      </div>

      <div
        role="note"
        className="rounded-md border border-[#FDE68A] bg-[#FFFBEB] px-4 py-2.5 text-xs text-[#92400E]"
      >
        <strong>IA Responsável:</strong> triagem automática — não substitui vistoria técnica presencial. Em emergência elétrica ou risco iminente: <strong>Enel 0800 72 72 196</strong> · <strong>Bombeiros 193</strong> · <strong>Defesa Civil 199</strong>.
      </div>

      <div className="grid gap-6 lg:grid-cols-12">
        {/* Coluna Esquerda: Pergunta + Resposta */}
        <div className="flex h-[640px] flex-col rounded-md border border-[#D5E0D1] bg-white lg:col-span-7">
          <div className="space-y-3 border-b border-[#E6ECE3] p-4">
            <div className="flex flex-wrap items-center gap-2">
              <label
                htmlFor="assistant-tree-select"
                className="text-xs font-medium text-[#4A5D4E]"
              >
                Vincular árvore cadastrada (opcional):
              </label>
              <select
                id="assistant-tree-select"
                value={selectedTreeId}
                onChange={(e) => setSelectedTreeId(e.target.value)}
                className="flex-1 rounded border border-[#D5E0D1] bg-[#F6F8F5] px-3 py-1.5 text-xs text-[#18241B] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
              >
                <option value="">Sem árvore específica (dúvida geral)</option>
                {selectedTreeId && !criticalSampleTrees.some((t) => t.geosampa_id === selectedTreeId) && (
                  <option value={selectedTreeId}>Árvore selecionada #{selectedTreeId}</option>
                )}
                {criticalSampleTrees.map((t) => (
                  <option key={t.geosampa_id} value={t.geosampa_id}>
                    #{t.geosampa_id} — {t.distrito} ({formatTriageLabel(t.v3_triage.raw_consolidated)})
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {(statusInfo?.suggested_prompts ?? []).map((prompt, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => submitQuestion(prompt)}
                  className="rounded border border-[#D5E0D1] bg-[#FAFBF9] px-2.5 py-1 text-left text-xs text-[#2D4234] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC]"
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
            {turns.length === 0 && (
              <div className="flex h-full flex-col items-center justify-center text-center text-[#4A5D4E]">
                <p className="text-sm font-semibold text-[#18241B]">
                  Descreva a situação observada na rua ou clique em uma pergunta pronta acima
                </p>
                <p className="mt-1.5 max-w-md text-xs leading-relaxed">
                  A resposta indica o nível de risco, o órgão responsável, o telefone ou portal
                  correto (SP156, Enel, Defesa Civil ou Bombeiros) e a legislação aplicável.
                </p>
              </div>
            )}

            {turns.map((turn) => {
              if (turn.role === "user") {
                return (
                  <div key={turn.id} className="flex justify-end">
                    <div className="max-w-[85%] rounded-md bg-[#1B4332] px-4 py-2.5 text-sm text-white">
                      {turn.text}
                    </div>
                  </div>
                );
              }
              const s = turn.response?.structured;
              const riskMeta = s ? RISK_STYLE[s.nivel_risco] ?? RISK_STYLE.preventivo_rotina : null;

              return (
                <div
                  key={turn.id}
                  className="rounded-md border border-[#D5E0D1] bg-white p-4"
                >
                  {s && riskMeta && (
                    <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`rounded border px-2 py-0.5 text-xs font-bold ${riskMeta.cls}`}
                        >
                          {riskMeta.label}
                        </span>
                        <span className="rounded border border-[#D5E0D1] bg-[#F6F8F5] px-2 py-0.5 text-xs font-medium text-[#4A5D4E]">
                          Análise IA · {formatDurationSeconds(turn.response?.latency_ms)}
                        </span>
                      </div>
                      {s.pode_podar_por_conta === false && (
                        <span className="text-xs font-semibold text-[#991B1B]">
                          Proibida poda por conta própria em área pública
                        </span>
                      )}
                      {s.pode_podar_por_conta === true && (
                        <span className="text-xs font-semibold text-[#92400E]">
                          Área privada: laudo ART + aviso 10 dias úteis
                        </span>
                      )}
                    </div>
                  )}

                  <p className="text-sm leading-relaxed text-[#18241B]">{turn.text}</p>

                  {s && (
                    <div className="mt-3 grid gap-2 rounded border border-[#E6ECE3] bg-[#F6F8F5] p-3 text-xs sm:grid-cols-2">
                      <div>
                        <span className="text-[#4A5D4E]">Órgão Responsável</span>
                        <p className="font-semibold text-[#18241B]">{s.orgao_responsavel}</p>
                      </div>
                      <div>
                        <span className="text-[#4A5D4E]">Canal / Telefone Oficial</span>
                        <p className="font-mono font-semibold text-[#1B4332]">
                          {s.telefone_ou_canal}
                        </p>
                      </div>
                    </div>
                  )}

                  {s && s.fontes.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                      <span className="text-xs text-[#4A5D4E]">Fontes normativas:</span>
                      {s.fontes.map((fid) => (
                        <span
                          key={fid}
                          className="rounded border border-[#C6DBC0] bg-[#EEF4EC] px-1.5 py-0.5 font-mono text-xs font-semibold text-[#1B4332]"
                        >
                          [{fid}]
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}

            {loading && (
              <div className="flex items-center gap-2.5 rounded border border-[#C6DBC0] bg-white p-3.5 text-xs font-medium text-[#1B4332]">
                <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
                <span>Analisando sua pergunta e consultando as normas...</span>
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submitQuestion(input);
            }}
            className="flex items-center gap-2 border-t border-[#E6ECE3] bg-white p-4"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              aria-label="Pergunta para o assistente normativo"
              placeholder="Pergunte sobre poda, fiação caída, SP156, Enel ou informe um número GeoSampa..."
              className="flex-1 rounded border border-[#D5E0D1] bg-[#F6F8F5] px-4 py-2.5 text-sm text-[#18241B] placeholder-[#6B7D70] focus:border-[#2D6A4F] focus:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="inline-flex items-center gap-2 rounded bg-[#1B4332] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#143325] disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              <span>{loading ? "Analisando..." : "Analisar com IA"}</span>
            </button>
          </form>
        </div>

        {/* Coluna Direita: Fontes Citadas */}
        <div className="flex h-[640px] flex-col rounded-md border border-[#D5E0D1] bg-white p-5 lg:col-span-5">
          <div className="border-b border-[#E6ECE3] pb-3">
            <h2 className="text-sm font-bold text-[#18241B]">
              Base de Normas Consultada e Dados da Árvore
            </h2>
            <p className="text-xs text-[#4A5D4E]">
              Trechos exatos utilizados para fundamentar a resposta acima
            </p>
          </div>

          {!activeEvidence ? (
            <div className="flex flex-1 flex-col items-center justify-center text-center text-xs text-[#4A5D4E]">
              Envie uma pergunta para visualizar os trechos oficiais da legislação e os dados
              cadastrais usados na resposta.
            </div>
          ) : (
            <div className="mt-3 flex-1 space-y-4 overflow-y-auto pr-1">
              {activeEvidence.tree_context && (
                <div className="rounded border border-[#C6DBC0] bg-[#EEF4EC] p-3.5 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-[#1B4332]">
                      GeoSampa #{activeEvidence.tree_context.geosampa_id}
                    </span>
                    <span className="rounded border border-[#C6DBC0] bg-white px-2 py-0.5 text-xs font-semibold text-[#18241B]">
                      {formatTriageLabel(activeEvidence.tree_context.triage_consolidated)}
                    </span>
                  </div>
                  <p className="mt-1 text-[#3D4F41]">
                    {activeEvidence.tree_context.distrito} · Subprefeitura{" "}
                    {activeEvidence.tree_context.subprefeitura} · SISGAU:{" "}
                    {activeEvidence.tree_context.sisgau_last_inspection || "Sem registro"}
                  </p>
                  <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-[#18241B]">
                    <div>
                      Contato direto:{" "}
                      <strong>
                        {activeEvidence.tree_context.contact_copa_pct !== undefined
                          ? `${activeEvidence.tree_context.contact_copa_pct.toFixed(1).replace(".", ",")}%`
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
                          ? `${activeEvidence.tree_context.wind_copa_pct.toFixed(1).replace(".", ",")}%`
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
                          ? `${activeEvidence.tree_context.wire_frame_pct.toFixed(1).replace(".", ",")}%`
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

              <div className="space-y-2.5">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-[#4A5D4E]">
                  Trechos Normativos ({activeEvidence.citations.length})
                </h3>
                {activeEvidence.citations.map((chunk) => (
                  <div
                    key={chunk.chunk_id}
                    className={`rounded border p-3.5 text-xs ${
                      chunk.cited
                        ? "border-[#2D6A4F] bg-[#EEF4EC]"
                        : "border-[#D5E0D1] bg-[#FAFBF9]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono font-bold text-[#1B4332]">
                        [{chunk.chunk_id}]
                      </span>
                      <span className="font-mono text-xs text-[#4A5D4E]">
                        relevância {(chunk.score * 100).toFixed(0)}% {chunk.cited ? "· Citado" : ""}
                      </span>
                    </div>
                    <p className="mt-1 font-semibold text-[#18241B]">{chunk.titulo}</p>
                    <p className="text-xs text-[#4A5D4E]">{chunk.fonte}</p>
                    <p className="mt-1.5 leading-relaxed text-[#3D4F41]">{chunk.conteudo}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
