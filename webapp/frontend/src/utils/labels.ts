export const APP_LABELS = {
  assistantTab: "Assistente",
  assistantSubtitle: "Normas e canais oficiais",
  assistantSubmit: "Perguntar",
  parallaxLabel: "Visível só no 2º ângulo",
  apiDocs: "Documentação da API",
} as const;

export const TRIAGE_LABELS: Record<string, string> = {
  CRITICO_CONFIRMADO: "Crítico confirmado nos dois ângulos",
  DIVERGENCIA_PARALAXE: APP_LABELS.parallaxLabel,
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

const DISTRICT_ACCENT_MAP: Record<string, string> = {
  PINHEIROS: "Pinheiros",
  MOOCA: "Mooca",
  "ALTO DA MOOCA": "Alto da Mooca",
  "VILA MARIANA": "Vila Mariana",
  CONSOLACAO: "Consolação",
  CONSOLAÇÃO: "Consolação",
  "BELA VISTA": "Bela Vista",
  "JARDIM PAULISTA": "Jardim Paulista",
  SE: "Sé",
  SÉ: "Sé",
  TATUAPE: "Tatuapé",
  "AGUA RASA": "Água Rasa",
  LIBERDADE: "Liberdade",
  SAUDE: "Saúde",
  CAMBUCI: "Cambuci",
  MOEMA: "Moema",
  "ITAIM BIBI": "Itaim Bibi",
  PERDIZES: "Perdizes",
  BUTANTA: "Butantã",
};

const LOWERCASE_CONNECTORS = new Set(["da", "de", "do", "das", "dos", "e"]);

export function formatDistrictName(raw?: string | null): string {
  if (!raw) return "—";
  const cleaned = raw.trim();
  if (!cleaned) return "—";
  const upper = cleaned.toUpperCase();
  if (DISTRICT_ACCENT_MAP[upper]) {
    return DISTRICT_ACCENT_MAP[upper];
  }
  return cleaned
    .toLowerCase()
    .split(/\s+/)
    .map((word, index) => {
      if (index > 0 && LOWERCASE_CONNECTORS.has(word)) {
        return word;
      }
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

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

export function formatNumberPtBr(value: number, fractionDigits = 1): string {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("pt-BR", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value);
}

export function formatPercentPtBr(value: number, fractionDigits = 1): string {
  if (!Number.isFinite(value)) return "—";
  return `${formatNumberPtBr(value, fractionDigits)}%`;
}

export function formatRelativePct(part: number, total: number): string {
  if (!total || total <= 0 || !part || part <= 0) return "0,0%";
  const pct = Math.min(100, (part / total) * 100);
  if (pct < 0.1) return "< 0,1%";
  return formatPercentPtBr(pct, 1);
}

export function formatDurationSeconds(ms?: number | null): string {
  if (ms === undefined || ms === null || Number.isNaN(ms)) return "";
  return `${formatNumberPtBr(ms / 1000, 1)} s`;
}

export function formatDatePtBr(
  raw?: string | null,
  mode: "full" | "monthYear" | "dateTime" = "full",
): string {
  if (!raw) return "Sem registro";
  const text = raw.trim();
  if (!text) return "Sem registro";

  const ymMatch = /^(\d{4})-(\d{2})$/.exec(text);
  if (ymMatch) {
    const year = Number(ymMatch[1]);
    const month = Number(ymMatch[2]);
    const d = new Date(year, month - 1, 1);
    return d.toLocaleDateString("pt-BR", { month: "short", year: "numeric" });
  }

  const normalized = /^\d{4}-\d{2}-\d{2}Z?$/i.test(text)
    ? `${text.slice(0, 10)}T12:00:00`
    : text;
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) {
    return text;
  }

  if (mode === "monthYear") {
    return parsed.toLocaleDateString("pt-BR", { month: "short", year: "numeric" });
  }
  if (mode === "dateTime") {
    return parsed.toLocaleString("pt-BR", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  return parsed.toLocaleDateString("pt-BR", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

export function isInternalChunkId(chunkId?: string | null): boolean {
  if (!chunkId) return false;
  return /^(CHANGELOG|ARBORGRID)-/i.test(chunkId.trim());
}

export function formatRelevanceLabel(score: number): {
  label: string;
  cls: string;
} {
  if (score >= 0.35) {
    return {
      label: "Relevância alta",
      cls: "border-[#A7F3D0] bg-[#ECFDF5] text-[#065F46]",
    };
  }
  if (score >= 0.2) {
    return {
      label: "Relevância média",
      cls: "border-[#FDE68A] bg-[#FFFBEB] text-[#92400E]",
    };
  }
  return {
    label: "Relevância baixa",
    cls: "border-[#D5E0D1] bg-[#F6F8F5] text-[#4A5D4E]",
  };
}
