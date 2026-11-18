"""Metricas Prometheus do ArborGrid (4 Golden Signals + telemetria do assistente)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


# Metricas dos 4 Golden Signals (Latency, Traffic, Errors, Saturation)
HTTP_REQUESTS_TOTAL = Counter(
    "arborgrid_http_requests_total",
    "Total de requisicoes HTTP processadas (Traffic)",
    ["method", "route", "status_code"],
)
HTTP_REQUEST_DURATION = Histogram(
    "arborgrid_http_request_duration_seconds",
    "Latencia de resposta HTTP em segundos (Latency)",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
)
HTTP_ERRORS_TOTAL = Counter(
    "arborgrid_http_errors_total",
    "Total de respostas de erro HTTP 4xx/5xx (Errors)",
    ["method", "route", "error_class"],
)
HTTP_INPROGRESS = Gauge(
    "arborgrid_inprogress_requests",
    "Requisicoes HTTP simultaneas em andamento (Saturation)",
)
TREES_LOADED_GAUGE = Gauge(
    "arborgrid_trees_loaded",
    "Quantidade de arvores georreferenciadas carregadas do manifesto",
)
GOLD_SCORES_GAUGE = Gauge(
    "arborgrid_gold_scores_loaded",
    "Quantidade de registros sincronizados da camada Gold (Big Data)",
)
ASSISTANT_RESPONSES_TOTAL = Counter(
    "arborgrid_assistant_responses_total",
    "Total de respostas do assistente tecnico por modo",
    ["mode"],
)
LLM_FALLBACK_TOTAL = Counter(
    "arborgrid_llm_fallback_total",
    "Total de fallbacks do servidor LLM local para o motor deterministico",
    ["reason"],
)
