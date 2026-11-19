import { useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import { AlertTriangle, Download, Loader2, Play, Share2, Sparkles, Upload } from "lucide-react";
import { fetchVisionSamples, formatFriendlyError } from "../services/api";
import type { VisionSampleItem, VisionSamplesPayload } from "../types/arborgrid";
import {
  ensureOnnxSession,
  isModelCached,
  isOnnxSessionReady,
  renderResultToCanvas,
  runOnnxInference,
  subscribeOnnxProgress,
  type VisionInferenceResult,
  type VisionLayerVisibility,
} from "../utils/visionOrt";
import {
  ANALYSIS_GRID_PIXELS,
  formatDistrictName,
  formatDurationSeconds,
  formatMacroTriageShort,
  formatNumberPtBr,
  formatRelativePct,
} from "../utils/labels";

interface VisionLabViewProps {
  onSendToAssistant?: (geosampaId: string, prompt: string) => void;
}

const BADGE_CLASS: Record<string, string> = {
  CRITICO: "border-[#FECACA] bg-[#FEF2F2] text-[#991B1B]",
  PROXIMIDADE: "border-[#FDE68A] bg-[#FFFBEB] text-[#92400E]",
  SEGURO: "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]",
};

// Frase do resultado na mensagem de compartilhamento, em linguagem simples
const SHARE_PHRASE: Record<string, string> = {
  CRITICO: "risco crítico, a copa encosta na fiação",
  PROXIMIDADE: "atenção, galhos perto da fiação",
  SEGURO: "sem conflito aparente com a fiação nesta foto",
};

const TRIAGE_EXPLANATION: Record<string, { title: string; subtitle: string }> = {
  CRITICO: {
    title: "Risco crítico: contato aparente com a fiação",
    subtitle:
      "A análise visual detectou interseção direta entre os galhos da árvore e cabos aéreos. Sem faíscas ou fios rompidos, o registro preventivo deve ser aberto pelo SP156.",
  },
  PROXIMIDADE: {
    title: "Atenção: galhos próximos à rede elétrica",
    subtitle:
      "Os galhos não tocam os fios neste ângulo, mas estão na faixa de alcance em dias de vento forte. Recomenda-se vistoria preventiva pelo SP156.",
  },
  SEGURO: {
    title: "Sem conflito aparente neste ângulo",
    subtitle:
      "Não foi detectada sobreposição direta entre copa e cabos aéreos nesta fotografia. Recomenda-se observar também outro ângulo da via quando possível.",
  },
};

export function VisionLabView({ onSendToAssistant }: VisionLabViewProps) {
  const [meta, setMeta] = useState<VisionSamplesPayload | null>(null);
  const [selectedSample, setSelectedSample] = useState<VisionSampleItem | null>(null);
  const [activeImageSrc, setActiveImageSrc] = useState<string | null>(null);
  const [customTitle, setCustomTitle] = useState<string | null>(null);

  const [sessionReady, setSessionReady] = useState<boolean>(() => isOnnxSessionReady());
  const [needsModelDownload, setNeedsModelDownload] = useState(false);
  const [preparingModel, setPreparingModel] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [loadingText, setLoadingText] = useState<string | null>(null);
  const [progressPct, setProgressPct] = useState<number>(0);
  const [downloadProgress, setDownloadProgress] = useState<{
    loadedMb: number;
    totalMb: number;
  } | null>(null);
  const [running, setRunning] = useState(false);
  const [inferenceElapsedSec, setInferenceElapsedSec] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [result, setResult] = useState<VisionInferenceResult | null>(null);
  const [layers, setLayers] = useState<VisionLayerVisibility>({
    copa: true,
    fiacao: true,
    vento: false,
    opacity: 0.2,
    borderThickness: 2,
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const startModelPreparation = (payload: VisionSamplesPayload) => {
    const versionTag = payload.model_etag || payload.model_version || payload.model_variant;
    setModelLoadError(null);
    setPreparingModel(true);
    return ensureOnnxSession(
      payload.model_url,
      (loadedMb, totalMb, statusText) => {
        setLoadingText(statusText);
        setDownloadProgress({ loadedMb, totalMb });
        setProgressPct(totalMb > 0 ? Math.min(100, Math.round((loadedMb / totalMb) * 100)) : 0);
      },
      versionTag,
    )
      .then(() => {
        setSessionReady(true);
        setNeedsModelDownload(false);
        setModelLoadError(null);
        setLoadingText(null);
        setDownloadProgress(null);
      })
      .catch(() => {
        setLoadingText(null);
        setDownloadProgress(null);
        setModelLoadError("Não foi possível preparar a análise. Tente novamente.");
      })
      .finally(() => {
        setPreparingModel(false);
      });
  };

  useEffect(() => {
    const unsubProgress = subscribeOnnxProgress((loadedMb, totalMb, statusText) => {
      setLoadingText(statusText);
      setDownloadProgress({ loadedMb, totalMb });
      setProgressPct(totalMb > 0 ? Math.min(100, Math.round((loadedMb / totalMb) * 100)) : 0);
    });

    fetchVisionSamples()
      .then(async (res) => {
        setMeta(res);
        if (res.samples.length > 0) {
          setSelectedSample(res.samples[0]);
          setActiveImageSrc(res.samples[0].image_url);
        }
        if (!res.model_available) {
          setModelLoadError("Ferramenta de análise visual indisponível no servidor.");
          return;
        }
        if (isOnnxSessionReady()) {
          setSessionReady(true);
          setNeedsModelDownload(false);
          setLoadingText(null);
          return;
        }
        const versionTag = res.model_etag || res.model_version || res.model_variant;
        const cached = await isModelCached(res.model_url, versionTag);
        if (cached) {
          void startModelPreparation(res);
        } else {
          setNeedsModelDownload(true);
        }
      })
      .catch((err) => setErrorMsg(formatFriendlyError(err)));

    return () => {
      unsubProgress();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!running) {
      setInferenceElapsedSec(0);
      return;
    }
    setInferenceElapsedSec(0);
    const id = window.setInterval(() => {
      setInferenceElapsedSec((s) => s + 1);
    }, 1000);
    return () => window.clearInterval(id);
  }, [running]);

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
    if (running) return;
    setSelectedSample(sample);
    setCustomTitle(null);
    setResult(null);
    setErrorMsg(null);
    setActiveImageSrc(sample.image_url);
  };

  const handleUploadFile = (file: File) => {
    if (running) return;
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
  };

  const handleDragOver = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (running) {
      if (e.dataTransfer) e.dataTransfer.dropEffect = "none";
      if (isDragging) setIsDragging(false);
      return;
    }
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (running) {
      if (isDragging) setIsDragging(false);
      return;
    }
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
    if (running) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) handleUploadFile(file);
  };

  const handleRunInference = async () => {
    if (!meta?.model_available) {
      setErrorMsg("Ferramenta de análise visual indisponível no servidor.");
      return;
    }
    if (!loadedImageRef.current) {
      setErrorMsg("Aguarde o carregamento da imagem na tela.");
      return;
    }

    setErrorMsg(null);
    setModelLoadError(null);
    setRunning(true);
    try {
      const versionTag = meta.model_etag || meta.model_version || meta.model_variant;
      const session = await ensureOnnxSession(
        meta.model_url,
        (loadedMb, totalMb, statusText) => {
          setLoadingText(statusText);
          setProgressPct(totalMb > 0 ? Math.min(100, Math.round((loadedMb / totalMb) * 100)) : 0);
        },
        versionTag,
      );
      setSessionReady(true);
      setLoadingText(null);
      await new Promise((r) => setTimeout(r, 20));

      const res = await runOnnxInference(session, loadedImageRef.current);
      setResult(res);
      if (canvasRef.current && loadedImageRef.current) {
        renderResultToCanvas(canvasRef.current, loadedImageRef.current, res, layers);
      }
    } catch (err) {
      setErrorMsg(
        err instanceof Error && err.message
          ? err.message
          : "Não foi possível analisar a imagem. Tente novamente.",
      );
    } finally {
      setRunning(false);
      setLoadingText(null);
    }
  };

  const handleAskAssistant = () => {
    if (!result || !onSendToAssistant) return;
    const gid = selectedSample?.geosampa_id || "";
    const statusHuman = formatMacroTriageShort(result.triage);
    const contactPct = formatRelativePct(result.contactDirectPx, result.copaPx);
    const windPct = formatRelativePct(result.windBufferPx, result.copaPx);
    const prompt = selectedSample
      ? `Analise a árvore #${selectedSample.geosampa_id} (${formatDistrictName(selectedSample.distrito)}): a avaliação visual indicou ${statusHuman}, com ${contactPct} da copa em contato direto com a fiação e ${windPct} na zona de vento. Qual o encaminhamento oficial indicado?`
      : `Enviei a foto "${customTitle || "da rua"}" e a análise visual indicou situação de ${statusHuman}, com ${contactPct} da copa em contato direto com a fiação e ${windPct} na zona de vento (sem faíscas relatadas). Quais procedimentos e canais oficiais devo seguir segundo as normas de São Paulo?`;
    onSendToAssistant(gid, prompt);
  };

  const summaryInfo = result ? TRIAGE_EXPLANATION[result.triage] : null;

  const canShare = useMemo(
    () => typeof navigator !== "undefined" && typeof navigator.share === "function",
    [],
  );

  const exportFilename = useMemo(() => {
    if (selectedSample) {
      return `arborgrid-analise-${selectedSample.geosampa_id}.png`;
    }
    if (customTitle) {
      const base = customTitle
        .replace(/\.[^.]+$/, "")
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, "-")
        .replace(/^-+|-+$/g, "");
      return `arborgrid-analise-${base || "foto"}.png`;
    }
    return "arborgrid-analise-foto.png";
  }, [selectedSample, customTitle]);

  const handleDownloadImage = () => {
    if (!result || running || !canvasRef.current) return;
    canvasRef.current.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = exportFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  const buildShareText = () => {
    const siteUrl = window.location.origin + "/";
    const riskSummary =
      SHARE_PHRASE[result?.triage ?? ""] ?? formatMacroTriageShort(result?.triage).toLowerCase();
    if (selectedSample) {
      return `Análise de imagem do ArborGrid SP: árvore #${selectedSample.geosampa_id} (${formatDistrictName(selectedSample.distrito)}), ${riskSummary}. Veja em ${siteUrl}`;
    }
    return `Análise de imagem do ArborGrid SP: ${riskSummary}. Faça a sua em ${siteUrl}`;
  };

  const handleShareImage = async () => {
    if (!result || running || !canShare) return;
    const shareTitle = "Análise de imagem do ArborGrid SP";
    const text = buildShareText();

    try {
      if (canvasRef.current && typeof navigator.canShare === "function") {
        const blob = await new Promise<Blob | null>((resolve) =>
          canvasRef.current?.toBlob((b) => resolve(b), "image/png"),
        );
        if (blob) {
          const file = new File([blob], exportFilename, { type: "image/png" });
          if (navigator.canShare({ files: [file], text })) {
            await navigator.share({ title: shareTitle, text, files: [file] });
            return;
          }
        }
      }
      await navigator.share({ title: shareTitle, text });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
    }
  };

  return (
    <div className="mx-auto max-w-[1680px] space-y-3 px-4 py-3.5 sm:px-8">
      {/* Cabeçalho compacto fundido com aviso de IA Responsável */}
      <div className="rounded-md border border-[#D5E0D1] bg-white px-4 py-2.5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <h1 className="text-lg font-bold text-[#18241B] sm:text-xl">
              Análise Visual de Árvores e Fiação Elétrica
            </h1>
            <p className="mt-0.5 text-[13px] text-[#4A5D4E]">
              Escolha um exemplo da cidade ou envie uma foto de rua para identificar galhos, cabos e áreas de contato direto.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {modelLoadError ? (
              <div className="flex flex-wrap items-center gap-2">
                <span
                  role="alert"
                  className="inline-flex items-center gap-1.5 rounded border border-[#FECACA] bg-[#FEF2F2] px-3 py-1 text-[13px] font-medium text-[#991B1B]"
                >
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span>{modelLoadError}</span>
                </span>
                {meta?.model_available && (
                  <button
                    type="button"
                    onClick={() => void startModelPreparation(meta)}
                    className="rounded border border-[#2D6A4F] bg-white px-2.5 py-1 text-xs font-semibold text-[#1B4332] hover:bg-[#EEF4EC]"
                  >
                    Tentar novamente
                  </button>
                )}
              </div>
            ) : sessionReady ? (
              <span className="inline-flex items-center gap-2 rounded border border-[#A7F3D0] bg-[#ECFDF5] px-3 py-1 text-[13px] font-medium text-[#065F46]">
                <span className="h-2 w-2 rounded-full bg-[#15803d]" aria-hidden="true" />
                <span>Pronto para analisar fotos</span>
              </span>
            ) : null}
          </div>
        </div>

        {!sessionReady && !modelLoadError && preparingModel && (
          <div className="mt-2.5 rounded-md border border-[#C6DBC0] bg-[#FAFBF9] px-4 py-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2 text-[13px] text-[#18241B]">
              <span className="inline-flex items-center gap-2 font-semibold">
                <Loader2 className="h-4 w-4 animate-spin text-[#1B4332]" aria-hidden="true" />
                <span>{loadingText || "Preparando análise visual..."}</span>
              </span>
              <span className="tabular-nums font-medium text-[#3D4F41]">
                {downloadProgress && downloadProgress.totalMb > 0
                  ? `${formatNumberPtBr(downloadProgress.loadedMb, 1)} MB de ${formatNumberPtBr(downloadProgress.totalMb, 1)} MB (${progressPct}%)`
                  : `${progressPct}%`}
              </span>
            </div>
            <div className="mt-2 h-2 w-full overflow-hidden rounded bg-[#E6ECE3]">
              <div
                className="h-full bg-[#2D6A4F] transition-all duration-200"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}

        {!sessionReady && !modelLoadError && !preparingModel && needsModelDownload && (
          <div className="mt-2.5 flex flex-wrap items-center justify-between gap-3 rounded-md border border-[#C6DBC0] bg-[#FAFBF9] px-4 py-2.5">
            <div className="min-w-0 flex-1 space-y-0.5">
              <h2 className="text-sm font-bold text-[#18241B]">
                A análise roda no seu computador
              </h2>
              <p className="text-[13px] text-[#4A5D4E]">
                A foto não é enviada para nenhum servidor: o modelo de visão computacional é baixado uma vez (cerca de 55 MB) e roda no seu navegador.
              </p>
            </div>
            <button
              type="button"
              disabled={!meta}
              onClick={() => meta && void startModelPreparation(meta)}
              className="inline-flex shrink-0 items-center gap-2 rounded bg-[#1B4332] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#143325] disabled:opacity-50"
            >
              <Download className="h-4 w-4 shrink-0" aria-hidden="true" />
              <span>Preparar análise no meu computador</span>
            </button>
          </div>
        )}

        <div
          role="note"
          className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-[#E6ECE3] pt-2 text-[13px] text-[#92400E]"
        >
          <strong className="font-semibold">IA Responsável:</strong>
          <span>
            análise local no navegador (a foto não sai do dispositivo). Não substitui vistoria presencial. Emergência:
          </span>
          <strong className="font-semibold">
            Enel 0800 72 72 196, Bombeiros 193, Defesa Civil 199
          </strong>
        </div>
      </div>

      <div className="flex flex-col gap-4 lg:grid lg:grid-cols-12 lg:items-start">
        {/* Coluna Esquerda (desktop: 4 colunas empilhadas sem lacunas | celular: ordem 1, 2, 5) */}
        <div className="contents lg:col-span-4 lg:flex lg:flex-col lg:gap-4">
          {/* Card 1 (mobile: 1º | desktop: esquerda topo): Amostras + Upload */}
          <div className="order-1 min-w-0 rounded-md border border-[#D5E0D1] bg-white p-3 lg:order-none">
            <h2 className="text-sm font-semibold text-[#18241B]">
              1. Escolha um exemplo ou envie sua foto
            </h2>

            <div className="mt-2 grid min-h-[220px] grid-cols-2 gap-2">
              {meta?.samples.map((s) => {
                const active = selectedSample?.id === s.id;
                return (
                  <button
                    key={s.id}
                    type="button"
                    disabled={running}
                    onClick={() => handleSelectSample(s)}
                    className={`flex flex-col overflow-hidden rounded border text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                      active
                        ? "border-[#1B4332] bg-[#EEF4EC] ring-1 ring-[#1B4332]"
                        : "border-[#D5E0D1] bg-[#FAFBF9] hover:border-[#2D6A4F] disabled:hover:border-[#D5E0D1]"
                    }`}
                  >
                    <img
                      src={s.image_url}
                      alt={`Foto Street View da árvore #${s.geosampa_id} (${s.region_label})`}
                      className="keep-colors h-[80px] w-full object-cover"
                    />
                    <div className="px-2 py-1">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-mono text-xs font-semibold text-[#18241B]">
                          #{s.geosampa_id}
                        </span>
                        <span
                          className={`rounded border px-1.5 py-0.5 text-[11px] font-semibold ${
                            BADGE_CLASS[s.reference_v3.triage] ?? BADGE_CLASS.SEGURO
                          }`}
                        >
                          {formatMacroTriageShort(s.reference_v3.triage)}
                        </span>
                      </div>
                      <p className="mt-0.5 text-xs text-[#4A5D4E]">
                        {formatDistrictName(s.distrito)}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-2 border-t border-[#E6ECE3] pt-2">
              <label
                htmlFor="vision-file-upload"
                aria-disabled={running}
                onDragEnter={handleDragOver}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`flex flex-col items-center justify-center gap-1 rounded border border-dashed px-3 py-1.5 text-center text-[13px] transition-colors ${
                  running
                    ? "cursor-not-allowed border-[#D5E0D1] bg-[#F6F8F5] text-[#4A5D4E] opacity-50"
                    : isDragging
                      ? "cursor-pointer border-[#1B4332] bg-[#E7EFE4] text-[#18241B]"
                      : "cursor-pointer border-[#2D6A4F] bg-[#F6F8F5] text-[#1B4332] hover:bg-[#EEF4EC]"
                }`}
              >
                <div className="flex items-center gap-2 font-medium">
                  <Upload className="h-4 w-4 shrink-0" />
                  <span>
                    {isDragging && !running
                      ? "Solte a imagem aqui"
                      : "Enviar ou arrastar foto (.jpg, .png, .webp)"}
                  </span>
                </div>
                <input
                  id="vision-file-upload"
                  type="file"
                  disabled={running}
                  aria-disabled={running}
                  accept="image/jpeg,image/png,image/webp"
                  className="sr-only"
                  onChange={(e) => {
                    if (running) return;
                    const file = e.target.files?.[0];
                    if (file) handleUploadFile(file);
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
          </div>

          {/* Card 2 (mobile: 2º | desktop: esquerda meio): Faixa "2. Analisar foto" com altura fixa */}
          <div aria-live="polite" className="order-2 flex min-h-[108px] flex-col justify-between rounded-md border border-[#D5E0D1] bg-white px-3.5 py-2.5 lg:order-none lg:h-[108px]">
            {result && summaryInfo && !running ? (
              <>
                <div className="min-w-0 space-y-1">
                  <div className="flex items-center gap-2">
                    <span
                      className={`shrink-0 rounded border px-2 py-0.5 text-xs font-semibold ${
                        BADGE_CLASS[result.triage] ?? BADGE_CLASS.SEGURO
                      }`}
                    >
                      {formatMacroTriageShort(result.triage)}
                    </span>
                  </div>
                  <h2
                    className="truncate text-[13px] font-bold text-[#18241B]"
                    title={summaryInfo.title}
                  >
                    {summaryInfo.title}
                  </h2>
                </div>

                <div className="flex items-center gap-2">
                  {onSendToAssistant && (
                    <button
                      type="button"
                      onClick={handleAskAssistant}
                      className="inline-flex flex-1 items-center justify-center gap-1.5 rounded bg-[#1B4332] px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-[#143325]"
                    >
                      <Sparkles className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                      <span>Consultar assistente</span>
                    </button>
                  )}
                  <button
                    type="button"
                    disabled={running}
                    onClick={() => setResult(null)}
                    className={`rounded border border-[#D5E0D1] bg-white px-3 py-1.5 text-xs font-semibold text-[#1B4332] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC] disabled:cursor-not-allowed disabled:opacity-50 ${
                      onSendToAssistant ? "shrink-0" : "w-full"
                    }`}
                  >
                    Analisar outra foto
                  </button>
                </div>
              </>
            ) : running ? (
              <>
                <div className="min-w-0 space-y-0.5">
                  <span className="block text-sm font-semibold text-[#18241B]">
                    2. Analisar foto
                  </span>
                  <p className="truncate text-[13px] text-[#4A5D4E]">
                    Identificando galhos e fiação na foto...
                  </p>
                </div>

                <button
                  type="button"
                  disabled
                  className="inline-flex w-full items-center justify-center gap-2 rounded bg-[#1B4332] px-4 py-1.5 text-sm font-semibold text-white opacity-70"
                >
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  <span className="tabular-nums">Analisando… {inferenceElapsedSec} s</span>
                </button>
              </>
            ) : (
              <>
                <div className="min-w-0 space-y-0.5">
                  <div className="flex items-center justify-between gap-2">
                    <h2 className="text-sm font-semibold text-[#18241B]">
                      2. Analisar foto
                    </h2>
                    {!sessionReady && (
                      <span className="text-[11px] font-medium text-[#92400E]">
                        Prepare a análise primeiro
                      </span>
                    )}
                  </div>
                  <p className="truncate text-[13px] text-[#4A5D4E]">
                    {selectedSample ? (
                      <>
                        Levantamento:{" "}
                        <strong className="text-[#18241B]">
                          {formatMacroTriageShort(selectedSample.reference_v3.triage)}
                        </strong>
                        . Analise o ângulo atual.
                      </>
                    ) : (
                      "Identifique galhos, cabos elétricos e áreas de contato."
                    )}
                  </p>
                </div>

                <button
                  type="button"
                  disabled={!activeImageSrc || !sessionReady}
                  title={!sessionReady ? "Prepare a análise primeiro" : undefined}
                  onClick={handleRunInference}
                  className="inline-flex w-full items-center justify-center gap-2 rounded bg-[#1B4332] px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-[#143325] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Play className="h-4 w-4" aria-hidden="true" />
                  <span>2. Analisar foto</span>
                </button>
              </>
            )}
          </div>

          {/* Card 5 (mobile: 5º após resultado | desktop: esquerda abaixo): Camadas */}
          <div className="order-5 min-w-0 space-y-2.5 rounded-md border border-[#D5E0D1] bg-white p-3.5 lg:order-none">
            {((!sessionReady && loadingText) || errorMsg) && (
              <div aria-live="polite" className="space-y-2">
                {!running && !sessionReady && loadingText && (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between gap-2 text-[13px] text-[#2D4234]">
                      <span className="inline-flex items-center gap-1.5">
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-[#1B4332]" />
                        <span>{loadingText}</span>
                      </span>
                      <span className="tabular-nums">{progressPct}%</span>
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
                  <div className="flex items-start gap-2 rounded border border-[#FECACA] bg-[#FEF2F2] p-3 text-[13px] text-[#991B1B]">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>{errorMsg}</span>
                  </div>
                )}
              </div>
            )}

            <div className="space-y-2.5">
              <span className="block text-[13px] font-semibold text-[#18241B]">
                Camadas sobre a foto
              </span>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[13px]">
                    <label htmlFor="vision-opacity-slider" className="text-[#4A5D4E]">
                      Opacidade
                    </label>
                    <span className="font-mono tabular-nums text-[#18241B]">
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
                  <div className="flex items-center justify-between text-[13px]">
                    <label htmlFor="vision-border-slider" className="text-[#4A5D4E]">
                      Espessura da borda
                    </label>
                    <span className="font-mono tabular-nums text-[#18241B]">
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
              <div className="grid grid-cols-1 gap-2 text-[13px] sm:grid-cols-1 xl:grid-cols-3">
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
                    <span>{item.label}</span>
                  </label>
                ))}
              </div>
              <p className="flex items-center gap-1.5 text-[13px] leading-snug text-[#4A5D4E]">
                <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full bg-[#dc2626]" />
                <span>
                  Com <strong>Copa</strong> e <strong>Fiação</strong> ativas, a sobreposição direta
                  entre galhos e cabos é destacada em <strong>vermelho</strong>.
                </span>
              </p>
            </div>
          </div>
        </div>

        {/* Coluna Direita (desktop: 8 colunas | celular: ordem 3, 4) */}
        <div className="contents lg:col-span-8 lg:flex lg:flex-col lg:gap-4">
          {/* Card 3 (mobile: 3º | desktop: direita topo): Imagem/canvas */}
          <div
            onDragEnter={handleDragOver}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`order-3 relative min-w-0 rounded-md border bg-white p-3.5 transition-colors lg:order-none ${
              isDragging && !running ? "border-[#1B4332] bg-[#EEF4EC]" : "border-[#D5E0D1]"
            }`}
          >
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="text-[13px] text-[#4A5D4E]">
                {selectedSample ? (
                  <>
                    <strong className="font-semibold text-[#18241B]">
                      Árvore #{selectedSample.geosampa_id}
                    </strong>
                    , {selectedSample.region_label}
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
            </div>

            <div className="keep-colors relative flex min-h-[320px] max-h-[calc(100vh-290px)] items-center justify-center overflow-hidden rounded border border-[#E6ECE3] bg-[#F6F8F5]">
              <canvas
                ref={canvasRef}
                className="keep-colors max-h-[calc(100vh-290px)] w-auto max-w-full object-contain"
              />
              {running && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur-[1px]">
                  <div className="flex flex-col items-center gap-1.5 rounded-md border border-[#C6DBC0] bg-white px-5 py-3.5 text-center shadow-sm">
                    <div className="flex items-center gap-2 text-sm font-semibold text-[#1B4332]">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="tabular-nums">
                        Identificando galhos e fiação… {inferenceElapsedSec} s
                      </span>
                    </div>
                    <span className="text-xs text-[#4A5D4E]">
                      Processamento local no navegador
                    </span>
                  </div>
                </div>
              )}
              {isDragging && !running && (
                <div className="absolute inset-0 flex items-center justify-center bg-[#1B4332]/85 text-sm font-semibold text-white">
                  Solte a foto aqui para analisá-la
                </div>
              )}
            </div>

            <p className="mt-2 text-right text-xs text-[#4A5D4E]">
              Arraste outra foto para esta área ou escolha um exemplo ao lado
            </p>
          </div>

          {/* Card 4 (mobile: 4º | desktop: direita abaixo): Detalhes, ações e métricas com altura reservada */}
          <div className="order-4 flex min-h-[144px] flex-col justify-center rounded-md border border-[#D5E0D1] bg-white p-3.5 lg:order-none lg:h-[144px]">
            {result && summaryInfo ? (
              <div className="flex h-full flex-col justify-between gap-2">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <p className="min-w-0 flex-1 text-[13px] leading-snug text-[#4A5D4E]">
                    {summaryInfo.subtitle}
                  </p>

                  <div className="flex shrink-0 flex-col items-end gap-1">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        disabled={!result || running}
                        onClick={handleDownloadImage}
                        className="inline-flex min-h-[30px] items-center gap-1.5 rounded border border-[#D5E0D1] bg-white px-2.5 py-1 text-xs font-semibold text-[#1B4332] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC] disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        <Download className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                        <span>Baixar imagem</span>
                      </button>
                      {canShare && (
                        <button
                          type="button"
                          disabled={!result || running}
                          onClick={() => void handleShareImage()}
                          className="inline-flex min-h-[30px] items-center gap-1.5 rounded border border-[#D5E0D1] bg-white px-2.5 py-1 text-xs font-semibold text-[#1B4332] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC] disabled:cursor-not-allowed disabled:opacity-45"
                        >
                          <Share2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                          <span>Compartilhar</span>
                        </button>
                      )}
                    </div>
                    <span className="text-[11px] tabular-nums text-[#4A5D4E]">
                      Análise concluída em{" "}
                      {formatDurationSeconds(result.inferenceMs + result.postprocessMs)}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 border-t border-[#E6ECE3] pt-2 text-[13px] tabular-nums sm:grid-cols-4">
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
              <p className="text-center text-[13px] text-[#4A5D4E]">
                O resultado aparece aqui depois da análise.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
