import type {
  AssistantAskResponse,
  AssistantStatusPayload,
  MapBootstrapPayload,
  VisionSamplesPayload,
} from "../types/arborgrid";

export class ApiError extends Error {
  readonly status?: number;
  readonly isNetworkError: boolean;

  constructor(message: string, options?: { status?: number; isNetworkError?: boolean }) {
    super(message);
    this.name = "ApiError";
    this.status = options?.status;
    this.isNetworkError = Boolean(options?.isNetworkError);
  }
}

export function mapHttpStatusToMessage(status: number): string {
  if (status === 429) {
    return "Muitas perguntas seguidas. Aguarde 1 minuto e tente de novo.";
  }
  if (status >= 500) {
    return "O servidor está temporariamente indisponível. Tente de novo em instantes.";
  }
  if (status === 403) {
    return "Operação não habilitada neste ambiente.";
  }
  if (status === 404) {
    return "Recurso não encontrado no servidor.";
  }
  return "Não foi possível concluir a solicitação. Tente de novo.";
}

export function formatFriendlyError(err: unknown): string {
  if (err instanceof ApiError) {
    return err.message;
  }
  if (err instanceof TypeError) {
    return "Sem conexão com o servidor. Verifique a internet e tente de novo.";
  }
  if (err instanceof Error) {
    const lower = err.message.toLowerCase();
    if (
      lower.includes("failed to fetch") ||
      lower.includes("networkerror") ||
      lower.includes("network error") ||
      lower.includes("load failed")
    ) {
      return "Sem conexão com o servidor. Verifique a internet e tente de novo.";
    }
    if (/http\s*429/i.test(err.message)) {
      return "Muitas perguntas seguidas. Aguarde 1 minuto e tente de novo.";
    }
    if (/http\s*5\d\d/i.test(err.message)) {
      return "O servidor está temporariamente indisponível. Tente de novo em instantes.";
    }
    return err.message;
  }
  return "Ocorreu um erro inesperado. Tente de novo.";
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError("Sem conexão com o servidor. Verifique a internet e tente de novo.", {
      isNetworkError: true,
    });
  }

  if (!res.ok) {
    throw new ApiError(mapHttpStatusToMessage(res.status), { status: res.status });
  }
  return (await res.json()) as T;
}

export function fetchBootstrap(): Promise<MapBootstrapPayload> {
  return requestJson<MapBootstrapPayload>("/api/v1/public/map-bootstrap");
}

export function seedGoldDemo(): Promise<{
  batch_id: string;
  inserted: number;
  updated: number;
  total_gold_records: number;
}> {
  return requestJson("/api/v1/gold/seed-demo", { method: "POST" });
}

export function fetchVisionSamples(): Promise<VisionSamplesPayload> {
  return requestJson<VisionSamplesPayload>("/api/v1/vision/samples");
}

export function fetchAssistantStatus(): Promise<AssistantStatusPayload> {
  return requestJson<AssistantStatusPayload>("/api/v1/assistant/status");
}

export interface AssistantHistoryPayloadItem {
  role: "user" | "assistant";
  content: string;
}

export function askAssistant(
  question: string,
  geosampaId?: string | null,
  history: AssistantHistoryPayloadItem[] = [],
): Promise<AssistantAskResponse> {
  return requestJson<AssistantAskResponse>("/api/v1/assistant/ask", {
    method: "POST",
    body: JSON.stringify({
      question,
      geosampa_id: geosampaId || null,
      history,
    }),
  });
}
