import type {
  AssistantAskResponse,
  AssistantStatusPayload,
  MapBootstrapPayload,
  VisionSamplesPayload,
} from "../types/arborgrid";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${msg}`);
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
