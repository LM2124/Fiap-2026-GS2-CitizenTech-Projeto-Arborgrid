import { useEffect, useRef, useState, type DragEvent } from "react";
import { AlertTriangle, Loader2, Play, Upload } from "lucide-react";
import { fetchVisionSamples } from "../services/api";
import type { TreeRecord, VisionSampleItem, VisionSamplesPayload } from "../types/arborgrid";
import {
  ensureOnnxSession,
  renderResultToCanvas,
  runOnnxInference,
  type VisionInferenceResult,
  type VisionLayerVisibility,
} from "../utils/visionOrt";
import { CopilotView } from "./CopilotView";
import {
  ANALYSIS_GRID_PIXELS,
  formatDurationSeconds,
  formatMacroTriageShort,
  formatRelativePct,
} from "../utils/labels";

interface VisionLabViewProps {
  trees: TreeRecord[];
  onSendToAssistant?: (geosampaId: string, prompt: string) => void;
}

const BADGE_CLASS: Record<string, string> = {
  CRITICO: "border-[#FECACA] bg-[#FEF2F2] text-[#991B1B]",
  PROXIMIDADE: "border-[#FDE68A] bg-[#FFFBEB] text-[#92400E]",
  SEGURO: "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]",
};

const TRIAGE_EXPLANATION: Record<string, { title: string; subtitle: string }> = {
  CRITICO: {
    title: "Risco Crítico — Contato aparente com a fiação",
    subtitle:
      "A análise visual detectou interseção direta entre os galhos da árvore e cabos aéreos. Sem faíscas ou fios rompidos, o registro preventivo deve ser aberto pelo SP156.",
  },
  PROXIMIDADE: {
    title: "Atenção — Galhos próximos à rede elétrica",
    subtitle:
      "Os galhos não tocam os fios neste ângulo, mas estão na faixa de alcance em dias de vento forte. Recomenda-se vistoria preventiva pelo SP156.",
  },
  SEGURO: {
    title: "Sem Conflito Aparente neste ângulo",
    subtitle:
      "Não foi detectada sobreposição direta entre copa e cabos aéreos nesta fotografia. Recomenda-se observar também outro ângulo da via quando possível.",
  },
};

export function VisionLabView({ trees }: VisionLabViewProps) {
  const [meta, setMeta] = useState<VisionSamplesPayload | null>(null);
  const [selectedSample, setSelectedSample] = useState<VisionSampleItem | null>(null);
  const [activeImageSrc, setActiveImageSrc] = useState<string | null>(null);
  const [customTitle, setCustomTitle] = useState<string | null>(null);

  const [sessionReady, setSessionReady] = useState(false);
  const [loadingText, setLoadingText] = useState<string | null>(null);
  const [progressPct, setProgressPct] = useState<number>(0);
  const [running, setRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [result, setResult] = useState<VisionInferenceResult | null>(null);
  const [layers, setLayers] = useState<VisionLayerVisibility>({
    copa: true,
    fiacao: true,
    vento: false,
    opacity: 0.42,
    borderThickness: 2,
  });

  // Estado integrado para a seção Consulta de Normas e Análise IA logo abaixo
  const [embeddedTreeId, setEmbeddedTreeId] = useState<string | null>(null);
  const [embeddedPrompt, setEmbeddedPrompt] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const assistantSectionRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    fetchVisionSamples()
      .then((res) => {
        setMeta(res);
        if (res.samples.length > 0) {
          setSelectedSample(res.samples[0]);
          setActiveImageSrc(res.samples[0].image_url);
          setEmbeddedTreeId(res.samples[0].geosampa_id);
        }
        if (res.model_available) {
          void ensureOnnxSession(res.model_url, (loadedMb, totalMb, statusText) => {
            setLoadingText(statusText);
            setProgressPct(totalMb > 0 ? Math.min(100, Math.round((loadedMb / totalMb) * 100)) : 0);
          })
            .then(() => {
              setSessionReady(true);
              setLoadingText(null);
            })
            .catch(() => {
              setLoadingText(null);
            });
        }
      })
      .catch((err) => setErrorMsg(String(err)));

    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!activeImageSrc || !canvasRef.current) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      loadedImageRef.current = img;
      if (canvasRef.current) {
        renderResultToCanvas(canvasRef.current, img, result, layers);
      }
    };
    img.src = activeImageSrc;
  }, [activeImageSrc, result, layers]);

  const handleSelectSample = (sample: VisionSampleItem) => {
    setSelectedSample(sample);
    setCustomTitle(null);
    setResult(null);
    setErrorMsg(null);
    setActiveImageSrc(sample.image_url);
    setEmbeddedTreeId(sample.geosampa_id);
  };

  const handleUploadFile = (file: File) => {
    const validExt = /\.(jpe?g|png|webp)$/i.test(file.name);
    if (!file.type.startsWith("image/") && !validExt) {
      setErrorMsg("Formato inválido. Envie uma imagem JPG, PNG ou WebP.");
      return;
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setSelectedSample(null);
    setCustomTitle(file.name);
    setResult(null);
    setErrorMsg(null);
    setActiveImageSrc(url);
    setEmbeddedTreeId(null);
  };

  const handleDragOver = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    if (
      e.clientX <= rect.left ||
      e.clientX >= rect.right ||
      e.clientY <= rect.top ||
      e.clientY >= rect.bottom
    ) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) handleUploadFile(file);
  };

  const handleRunInference = async () => {
    if (!meta?.model_available) {
      setErrorMsg("Modelo de análise visual não disponível no servidor.");
      return;
    }
    if (!loadedImageRef.current) {
      setErrorMsg("Aguarde o carregamento da imagem na tela.");
      return;
    }

    setErrorMsg(null);
    setRunning(true);
    try {
      const session = await ensureOnnxSession(meta.model_url, (loadedMb, totalMb, statusText) => {
        setLoadingText(statusText);
        setProgressPct(totalMb > 0 ? Math.min(100, Math.round((loadedMb / totalMb) * 100)) : 0);
      });
      setSessionReady(true);
      setLoadingText("Identificando galhos e fiação elétrica na imagem...");
      await new Promise((r) => setTimeout(r, 20));

      const res = await runOnnxInference(session, loadedImageRef.current);
      setResult(res);
      if (canvasRef.current && loadedImageRef.current) {
        renderResultToCanvas(canvasRef.current, loadedImageRef.current, res, layers);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
      setLoadingText(null);
    }
  };

  const handleAskIntegratedAssistant = () => {
    if (!result) return;
    const gid = selectedSample?.geosampa_id || "";
    const statusHuman = formatMacroTriageShort(result.triage);
    const contactPct = formatRelativePct(result.contactDirectPx, result.copaPx);
    const windPct = formatRelativePct(result.windBufferPx, result.copaPx);
    const prompt = selectedSample
      ? `Analise a árvore #${selectedSample.geosampa_id} (${selectedSample.distrito}): a avaliação visual indicou ${statusHuman}, com ${contactPct} da copa em contato direto com a fiação e ${windPct} na zona de vento. Qual o encaminhamento oficial indicado?`
      : `Enviei a foto "${customTitle || "da rua"}" e a análise visual indicou situação de ${statusHuman}, com ${contactPct} da copa em contato direto com a fiação e ${windPct} na zona de vento (sem faíscas relatadas). Quais procedimentos e canais oficiais devo seguir segundo as normas de São Paulo?`;
    setEmbeddedTreeId(gid || null);
    setEmbeddedPrompt(prompt);
    assistantSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const summaryInfo = result ? TRIAGE_EXPLANATION[result.triage] : null;

  return (
    <div className="mx-auto max-w-[1680px] space-y-8 px-4 py-8 sm:px-8">
      {/* Cabeçalho explicativo para visitantes */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-md border border-[#D5E0D1] bg-white p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-bold text-[#18241B]">
            Análise Visual de Árvores e Fiação Elétrica
          </h1>
          <p className="max-w-4xl text-sm leading-relaxed text-[#4A5D4E]">
            Escolha um dos exemplos da cidade, clique para enviar ou arraste e solte uma foto
            tirada na rua sobre a área abaixo para identificar galhos, cabos elétricos e áreas de
            contato direto. Todo o processamento ocorre localmente no navegador e você pode
            consultar as normas aplicáveis logo abaixo.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`inline-flex items-center gap-2 rounded border px-3 py-1 text-xs font-medium ${
              sessionReady
                ? "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]"
                : "border-[#D5E0D1] bg-[#F6F8F5] text-[#4A5D4E]"
            }`}
          >
            {sessionReady ? (
              <span className="h-2 w-2 rounded-full bg-[#15803d]" />
            ) : (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-[#92400E]" />
            )}
            {sessionReady ? "Pronto para analisar fotos" : "Preparando análise visual..."}
          </span>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-12">
        {/* Coluna Esquerda: Amostras + Drag-and-Drop + Camadas */}
        <div className="space-y-4 lg:col-span-4">
          <div className="rounded-md border border-[#D5E0D1] bg-white p-5">
            <h2 className="text-sm font-semibold text-[#18241B]">
              1. Escolha um exemplo ou envie sua foto
            </h2>
            <p className="mt-1 text-xs text-[#4A5D4E]">
              Exemplos das quatro regiões estudadas em São Paulo:
            </p>

            <div className="mt-3 grid grid-cols-2 gap-2.5">
              {meta?.samples.map((s) => {
                const active = selectedSample?.id === s.id;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => handleSelectSample(s)}
                    className={`overflow-hidden rounded border text-left transition-colors ${
                      active
                        ? "border-[#1B4332] bg-[#EEF4EC]"
                        : "border-[#D5E0D1] bg-[#FAFBF9] hover:border-[#2D6A4F]"
                    }`}
                  >
                    <img
                      src={s.image_url}
                      alt={s.region_label}
                      className="h-24 w-full object-cover"
                    />
                    <div className="p-2">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-mono text-[11px] font-semibold text-[#18241B]">
                          #{s.geosampa_id}
                        </span>
                        <span
                          className={`rounded border px-1 py-0.5 text-[9px] font-semibold ${
                            BADGE_CLASS[s.reference_v3.triage] ?? BADGE_CLASS.SEGURO
                          }`}
                        >
                          {formatMacroTriageShort(s.reference_v3.triage)}
                        </span>
                      </div>
                      <p className="mt-0.5 truncate text-[11px] text-[#4A5D4E]">{s.distrito}</p>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-4 border-t border-[#E6ECE3] pt-4">
              <label
                htmlFor="vision-file-upload"
                onDragEnter={handleDragOver}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded border border-dashed px-4 py-4 text-center text-xs transition-colors ${
                  isDragging
                    ? "border-[#1B4332] bg-[#E7EFE4] text-[#18241B]"
                    : "border-[#2D6A4F] bg-[#F6F8F5] text-[#1B4332] hover:bg-[#EEF4EC]"
                }`}
              >
                <div className="flex items-center gap-2 font-medium">
                  <Upload className="h-4 w-4" />
                  <span>
                    {isDragging
                      ? "Solte a imagem aqui"
                      : "Clique ou arraste sua foto aqui (.jpg, .png, .webp)"}
                  </span>
                </div>
                <input
                  id="vision-file-upload"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="sr-only"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleUploadFile(file);
                  }}
                />
              </label>
            </div>
          </div>

          <div className="space-y-4 rounded-md border border-[#D5E0D1] bg-white p-5">
            <button
              type="button"
              disabled={running || !activeImageSrc}
              onClick={handleRunInference}
              className="flex w-full items-center justify-center gap-2 rounded bg-[#1B4332] px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-[#143325] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {running ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span>{running ? "Analisando foto..." : "2. Analisar Foto"}</span>
            </button>

            {loadingText && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-2 text-xs text-[#2D4234]">
                  <span className="inline-flex items-center gap-1.5">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-[#1B4332]" />
                    <span>{loadingText}</span>
                  </span>
                  <span>{progressPct}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded bg-[#E6ECE3]">
                  <div
                    className="h-full bg-[#2D6A4F] transition-all duration-200"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>
            )}

            {errorMsg && (
              <div className="flex items-start gap-2 rounded border border-[#FECACA] bg-[#FEF2F2] p-3 text-xs text-[#991B1B]">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            <div className="space-y-3 border-t border-[#E6ECE3] pt-3">
              <span className="block text-xs font-semibold text-[#18241B]">
                Camadas sobre a foto
              </span>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="vision-opacity-slider" className="text-[#4A5D4E]">
                      Opacidade
                    </label>
                    <span className="font-mono text-[#18241B]">
                      {Math.round(layers.opacity * 100)}%
                    </span>
                  </div>
                  <input
                    id="vision-opacity-slider"
                    type="range"
                    min={0}
                    max={1}
                    step={0.02}
                    value={layers.opacity}
                    onChange={(e) =>
                      setLayers((prev) => ({ ...prev, opacity: Number(e.target.value) }))
                    }
                    className="w-full accent-[#1B4332]"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="vision-border-slider" className="text-[#4A5D4E]">
                      Espessura da borda
                    </label>
                    <span className="font-mono text-[#18241B]">
                      {(layers.borderThickness ?? 2) === 0
                        ? "Sem borda"
                        : `Nível ${layers.borderThickness ?? 2} / 5`}
                    </span>
                  </div>
                  <input
                    id="vision-border-slider"
                    type="range"
                    min={0}
                    max={5}
                    step={1}
                    value={layers.borderThickness ?? 2}
                    onChange={(e) =>
                      setLayers((prev) => ({
                        ...prev,
                        borderThickness: Number(e.target.value),
                      }))
                    }
                    className="w-full accent-[#1B4332]"
                  />
                </div>
              </div>
              <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-3">
                {[
                  { key: "copa" as const, label: "Copa (verde)", dot: "bg-[#2dc33c]" },
                  { key: "fiacao" as const, label: "Fiação (amarelo)", dot: "bg-[#facc15]" },
                  {
                    key: "vento" as const,
                    label: "Margem de vento (laranja)",
                    dot: "bg-[#f97316]",
                  },
                ].map((item) => (
                  <label
                    key={item.key}
                    className="flex cursor-pointer items-center gap-2 rounded border border-[#D5E0D1] bg-[#FAFBF9] px-2.5 py-1.5 text-[#18241B]"
                  >
                    <input
                      type="checkbox"
                      checked={Boolean(layers[item.key])}
                      onChange={(e) =>
                        setLayers((prev) => ({ ...prev, [item.key]: e.target.checked }))
                      }
                      className="accent-[#1B4332]"
                    />
                    <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${item.dot}`} />
                    <span className="truncate">{item.label}</span>
                  </label>
                ))}
              </div>
              <p className="flex items-center gap-1.5 text-[11px] leading-snug text-[#4A5D4E]">
                <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full bg-[#dc2626]" />
                <span>
                  Com <strong>Copa</strong> e <strong>Fiação</strong> ativas, a sobreposição direta
                  entre galhos e cabos é destacada em <strong>vermelho</strong>.
                </span>
              </p>
            </div>
          </div>
        </div>

        {/* Coluna Direita: Imagem (com Drag-and-Drop direto) + Resultado */}
        <div className="space-y-4 lg:col-span-8">
          <div
            onDragEnter={handleDragOver}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative rounded-md border bg-white p-4 transition-colors ${
              isDragging ? "border-[#1B4332] bg-[#EEF4EC]" : "border-[#D5E0D1]"
            }`}
          >
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="text-xs text-[#4A5D4E]">
                {selectedSample ? (
                  <>
                    <strong className="font-semibold text-[#18241B]">
                      Árvore GeoSampa #{selectedSample.geosampa_id}
                    </strong>{" "}
                    · {selectedSample.region_label}
                  </>
                ) : customTitle ? (
                  <>
                    Foto enviada:{" "}
                    <strong className="font-semibold text-[#18241B]">{customTitle}</strong>
                  </>
                ) : (
                  "Selecione uma imagem ou arraste para cá"
                )}
              </div>
              <span className="text-[11px] text-[#4A5D4E]">
                Arraste outra foto para esta área ou escolha um exemplo ao lado
              </span>
            </div>

            <div className="relative flex min-h-[320px] justify-center overflow-hidden rounded border border-[#E6ECE3] bg-[#F6F8F5]">
              <canvas
                ref={canvasRef}
                className="max-h-[580px] w-auto max-w-full object-contain"
              />
              {running && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/75 backdrop-blur-[1px]">
                  <div className="flex items-center gap-2.5 rounded-md border border-[#C6DBC0] bg-white px-4 py-3 text-sm font-semibold text-[#1B4332]">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Identificando galhos e fiação elétrica...</span>
                  </div>
                </div>
              )}
              {isDragging && (
                <div className="absolute inset-0 flex items-center justify-center bg-[#1B4332]/85 text-sm font-semibold text-white">
                  Solte a foto aqui para analisá-la
                </div>
              )}
            </div>
          </div>

          {result && summaryInfo ? (
            <div className="rounded-md border border-[#D5E0D1] bg-white p-5">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`rounded border px-2.5 py-0.5 text-xs font-semibold ${
                        BADGE_CLASS[result.triage] ?? BADGE_CLASS.SEGURO
                      }`}
                    >
                      {formatMacroTriageShort(result.triage)}
                    </span>
                    <span className="text-xs text-[#4A5D4E]">
                      Análise concluída em{" "}
                      {formatDurationSeconds(result.inferenceMs + result.postprocessMs)}
                    </span>
                  </div>
                  <h3 className="text-base font-bold text-[#18241B]">{summaryInfo.title}</h3>
                  <p className="max-w-3xl text-xs leading-relaxed text-[#4A5D4E]">
                    {summaryInfo.subtitle}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={handleAskIntegratedAssistant}
                  className="rounded bg-[#1B4332] px-4 py-2.5 text-xs font-semibold text-white transition-colors hover:bg-[#143325]"
                >
                  Analisar com IA
                </button>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 border-t border-[#E6ECE3] pt-4 text-xs sm:grid-cols-4">
                <div>
                  <span className="text-[#4A5D4E]">Contato direto (da copa)</span>
                  <p className="mt-0.5 font-mono text-sm font-bold text-[#991B1B]">
                    {formatRelativePct(result.contactDirectPx, result.copaPx)}
                  </p>
                </div>
                <div>
                  <span className="text-[#4A5D4E]">Zona de vento (da copa)</span>
                  <p className="mt-0.5 font-mono text-sm font-bold text-[#92400E]">
                    {formatRelativePct(result.windBufferPx, result.copaPx)}
                  </p>
                </div>
                <div>
                  <span className="text-[#4A5D4E]">Fiação (da foto)</span>
                  <p className="mt-0.5 font-mono text-sm font-bold text-[#18241B]">
                    {formatRelativePct(result.wirePx, ANALYSIS_GRID_PIXELS)}
                  </p>
                </div>
                <div>
                  <span className="text-[#4A5D4E]">Copa arbórea (da foto)</span>
                  <p className="mt-0.5 font-mono text-sm font-bold text-[#1B4332]">
                    {formatRelativePct(result.copaPx, ANALYSIS_GRID_PIXELS)}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            selectedSample && (
              <div className="rounded-md border border-[#D5E0D1] bg-white p-4 text-xs text-[#4A5D4E]">
                Clique em <strong className="text-[#18241B]">2. Analisar Foto</strong> para
                executar a análise visual nesta imagem (referência cadastrada:{" "}
                <strong className="text-[#18241B]">
                  {formatMacroTriageShort(selectedSample.reference_v3.triage)}
                </strong>{" "}
                ·{" "}
                {formatRelativePct(
                  selectedSample.reference_v3.contact_direct_px,
                  selectedSample.reference_v3.copa_px,
                )}{" "}
                da copa em contato direto).
              </div>
            )
          )}
        </div>
      </div>

      {/* Seção Integrada: Consulta de Normas e Análise IA */}
      <section
        id="consulta-normas-integrada"
        ref={assistantSectionRef}
        className="border-t border-[#D5E0D1] pt-8"
      >
        <CopilotView
          embedded
          trees={trees}
          initialTreeId={embeddedTreeId}
          initialPrompt={embeddedPrompt}
          onConsumeInitialPrompt={() => setEmbeddedPrompt(null)}
        />
      </section>
    </div>
  );
}
