export type MacroTriage = "CRITICO" | "PROXIMIDADE" | "SEGURO";

export type PriorityTier = "URGENTE" | "ALTA" | "MEDIA" | "BAIXA";

export interface V3ViewAnalysis {
  triage: MacroTriage;
  color_hex?: string;
  copa_px: number;
  wire_px: number;
  pole_px: number;
  contact_direct_px: number;
  wind_buffer_px: number;
  v3_overlay_file?: string;
}

export interface TreeViewItem {
  perspective: string;
  capture_date?: string | null;
  heading?: number | null;
  distance_m?: number | null;
  google_maps_url?: string | null;
  overlay_url?: string | null;
  image_url?: string | null;
  v3_analysis: V3ViewAnalysis;
}

export interface TreeV3Triage {
  consolidated: MacroTriage;
  raw_consolidated: string;
  front_triage?: string;
  front_color?: string;
  side_triage?: string;
  side_color?: string;
  has_paralax_divergence: boolean;
  has_single_view?: boolean;
}

export interface TreeRecord {
  geosampa_id: string;
  region_key: string;
  region_label: string;
  distrito: string;
  subprefeitura: string;
  tree_lat: number;
  tree_lon: number;
  sisgau_last_inspection?: string | null;
  v3_triage: TreeV3Triage;
  views: TreeViewItem[];
}

export interface RegionCatalogEntry {
  label: string;
  bbox: [number, number, number, number];
  total_geosampa_trees?: number;
  sisgau_inspected_trees?: number;
  grid_type?: string;
  grid_desc?: string;
  context?: string;
  description?: string;
  is_negative_control?: boolean;
  underground_grid?: boolean;
  corridor_polygon?: [number, number][];
}

export interface DistrictBreakdownItem {
  region_key: string;
  region_label: string;
  grid_type: string;
  is_negative_control: boolean;
  total_geosampa_trees: number;
  total_trees: number;
  total_views: number;
  critico_total: number;
  critico_confirmado: number;
  divergencia_paralaxe: number;
  proximidade: number;
  seguro: number;
  max_contact_px: number;
  avg_contact_px: number;
  critical_rate_pct: number;
  sisgau_never_inspected_count: number;
  sisgau_outdated_count: number;
  sisgau_gap_pct: number;
}

export interface GoldRiskRecord {
  geosampa_id: string;
  distrito?: string | null;
  composite_risk_score: number;
  priority_tier: PriorityTier;
  wind_gust_kmh?: number | null;
  precipitation_mm?: number | null;
  weather_observed_at?: string | null;
  recommended_action: string;
  batch_id?: string;
  ingested_at_utc?: string;
}

export interface GoldSummary {
  source_table: string;
  total_records: number;
  last_batch_id: string | null;
  last_updated_utc: string | null;
}

export interface StatsPayload {
  project: string;
  model_version: string;
  total_trees: number;
  total_views: number;
  tree_triage_counts: Record<string, number>;
  detailed_triage_counts: Record<string, number>;
  view_triage_counts: Record<string, number>;
  sisgau_never_inspected_count: number;
  sisgau_outdated_count: number;
  regions: Record<string, Record<string, number>>;
  district_breakdown: DistrictBreakdownItem[];
  available_batch_dates: { date: string; records: number }[];
}

export interface MapBootstrapPayload {
  stats: StatsPayload;
  regions_catalog: Record<string, RegionCatalogEntry>;
  trees: TreeRecord[];
  gold_layer: GoldSummary;
  gold_scores_by_tree: Record<string, GoldRiskRecord>;
  gold_records: GoldRiskRecord[];
}

export interface VisionSampleItem {
  id: string;
  geosampa_id: string;
  region_key: string;
  region_label: string;
  distrito: string;
  subprefeitura: string;
  perspective: string;
  capture_date?: string | null;
  filename: string;
  image_url: string;
  consolidated_status: string;
  reference_v3: V3ViewAnalysis;
}

export interface VisionSamplesPayload {
  model_variant: string;
  model_architecture: string;
  model_available: boolean;
  model_size_mb: number;
  model_url: string;
  samples: VisionSampleItem[];
}

export interface AssistantStatusPayload {
  kb_loaded: boolean;
  chunks_count: number;
  llm_endpoint: string;
  llm_online: boolean;
  model_name: string;
  suggested_prompts: string[];
}

export interface CitationChunk {
  chunk_id: string;
  titulo: string;
  fonte: string;
  categoria?: string;
  score: number;
  conteudo: string;
  cited: boolean;
}

export interface AssistantStructuredOutput {
  status: "respondido" | "nao_consta";
  nivel_risco: "emergencia_imediata" | "urgente_vistoria" | "preventivo_rotina" | "fora_de_escopo";
  orgao_responsavel: string;
  telefone_ou_canal: string;
  pode_podar_por_conta: boolean | null;
  acao_imediata: string;
  fontes: string[];
}

export interface AssistantTreeContext {
  geosampa_id: string;
  distrito: string;
  subprefeitura: string;
  region_key: string;
  region_label: string;
  lat: number;
  lon: number;
  sisgau_last_inspection?: string | null;
  triage_macro: MacroTriage;
  triage_consolidated: string;
  has_parallax_divergence: boolean;
  max_contact_direct_px: number;
  max_wind_buffer_px: number;
  max_wire_px: number;
  max_copa_px?: number;
  contact_copa_pct?: number;
  wind_copa_pct?: number;
  wire_frame_pct?: number;
  gold_risk?: GoldRiskRecord | null;
}

export interface AssistantAskResponse {
  question: string;
  mode: "qwen3_gguf" | "rag_deterministic";
  structured: AssistantStructuredOutput;
  answer: string;
  citations: CitationChunk[];
  tree_context: AssistantTreeContext | null;
  latency_ms: number;
}
