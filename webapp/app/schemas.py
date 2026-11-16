"""Esquemas Pydantic da API ArborGrid e contratos com Big Data e Assistente IA."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


PriorityTier = Literal["URGENTE", "ALTA", "MEDIA", "BAIXA"]


class GoldRiskRecordIn(BaseModel):
    """Registro unitario vindo da tabela Gold (Databricks)."""

    geosampa_id: str = Field(..., min_length=1, max_length=32)
    distrito: str | None = Field(default=None, max_length=64)
    composite_risk_score: float = Field(..., ge=0.0, le=100.0)
    priority_tier: PriorityTier
    wind_gust_kmh: float | None = Field(default=None, ge=0.0, le=300.0)
    precipitation_mm: float | None = Field(default=None, ge=0.0, le=500.0)
    weather_observed_at: str | None = Field(default=None, max_length=64)
    recommended_action: str = Field(
        default="Monitoramento preventivo", min_length=3, max_length=240
    )
    audit_actor_hash: str | None = Field(
        default=None,
        max_length=128,
        description="Identificador anonimizado (hash SHA-256) conforme LGPD",
    )


class GoldRiskBatchIn(BaseModel):
    """Lote enviado pelo pipeline Big Data / Airflow / Databricks Gold."""

    batch_id: str = Field(..., min_length=3, max_length=80)
    records: list[GoldRiskRecordIn] = Field(..., min_length=1, max_length=5000)


class AssistantHistoryTurn(BaseModel):
    """Turno recente da conversa enviado para continuidade contextual."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=1500)


class AssistantQueryIn(BaseModel):
    """Pergunta enviada pela SPA ao Assistente Tecnico Qwen3 + RAG."""

    question: str = Field(..., min_length=2, max_length=1000)
    geosampa_id: str | None = Field(default=None, max_length=32)
    history: list[AssistantHistoryTurn] = Field(default_factory=list, max_length=12)
