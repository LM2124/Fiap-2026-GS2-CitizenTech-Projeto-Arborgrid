export const TRIAGE_LABELS: Record<string, string> = {
  CRITICO_CONFIRMADO: "Crítico confirmado nos dois ângulos",
  DIVERGENCIA_PARALAXE: "Crítico visível ao mudar o ângulo",
  CRITICO_MONO_VISADA: "Crítico (foto única)",
  CRITICO: "Risco Crítico",
  PROXIMIDADE: "Atenção (Proximidade ao Vento)",
  PROXIMIDADE_MONO_VISADA: "Atenção (foto única)",
  SEGURO: "Sem Conflito Aparente",
  SEGURO_MONO_VISADA: "Sem Conflito (foto única)",
};

export const MACRO_TRIAGE_SHORT_LABELS: Record<string, string> = {
  CRITICO: "Risco Crítico",
  PROXIMIDADE: "Atenção (Vento)",
  SEGURO: "Sem Conflito",
};

export const PRIORITY_TIER_LABELS: Record<string, string> = {
  URGENTE: "Urgente",
  ALTA: "Alta",
  MEDIA: "Média",
  BAIXA: "Baixa",
};

export function formatTriageLabel(raw?: string | null): string {
  if (!raw) return "Sem Conflito Aparente";
  const key = raw.trim().toUpperCase();
  return TRIAGE_LABELS[key] ?? "Sem Conflito Aparente";
}

export function formatMacroTriageShort(raw?: string | null): string {
  if (!raw) return "Sem Conflito";
  const key = raw.trim().toUpperCase();
  return MACRO_TRIAGE_SHORT_LABELS[key] ?? formatTriageLabel(key);
}

export function formatPriorityTier(tier?: string | null): string {
  if (!tier) return "Baixa";
  const key = tier.trim().toUpperCase();
  return PRIORITY_TIER_LABELS[key] ?? tier;
}

export const ANALYSIS_GRID_PIXELS = 640 * 640;

export function formatRelativePct(part: number, total: number): string {
  if (!total || total <= 0 || !part || part <= 0) return "0,0%";
  const pct = Math.min(100, (part / total) * 100);
  if (pct < 0.1) return "< 0,1%";
  return `${pct.toFixed(1).replace(".", ",")}%`;
}

export function formatDurationSeconds(ms?: number | null): string {
  if (ms === undefined || ms === null || Number.isNaN(ms)) return "";
  return `${(ms / 1000).toFixed(1).replace(".", ",")} s`;
}

