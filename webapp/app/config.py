"""Configuracoes da aplicacao via variaveis de ambiente (sem segredos no codigo)."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_WEBAPP_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_manifest_path() -> Path:
    local_data = _WEBAPP_ROOT / "data" / "manifest.json"
    if local_data.exists():
        return local_data
    return (
        _REPO_ROOT
        / "arborgrid-vision"
        / "dataset"
        / "sp_geosampa_api"
        / "manifest.json"
    )


def _default_kb_path() -> Path:
    local_kb = _WEBAPP_ROOT / "data" / "knowledge_base.jsonl"
    if local_kb.exists():
        return local_kb
    return _REPO_ROOT / "llm-assistant" / "base" / "knowledge_base.jsonl"


def _default_onnx_path() -> Path:
    candidates = [
        _REPO_ROOT / "arborgrid-vision" / "models" / "segformer_v3_int8_dyn_matmul.onnx",
        _WEBAPP_ROOT / "models" / "segformer_v3_int8_dyn_matmul.onnx",
        Path("/app/models/segformer_v3_int8_dyn_matmul.onnx"),
        (
            _REPO_ROOT
            / "arborgrid-vision"
            / "runs"
            / "quantization_benchmark"
            / "models"
            / "segformer_v3_int8_dyn_matmul.onnx"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_DEFAULT_MANIFEST = _default_manifest_path()
_DEFAULT_MEDIA = _REPO_ROOT / "arborgrid-vision" / "dataset" / "sp_geosampa_api"
_DEFAULT_GOLD_STATE = _WEBAPP_ROOT / "data" / "gold_risk_scores_runtime.json"
_DEFAULT_KB = _default_kb_path()
_DEFAULT_SAMPLE_GOLD = _WEBAPP_ROOT / "data" / "sample_gold_upsert.json"
_DEFAULT_SAMPLES_DIR = _WEBAPP_ROOT / "data" / "samples"
_DEFAULT_ONNX = _default_onnx_path()


class Settings(BaseSettings):
    """Configuracoes de execucao do servico ArborGrid Web App + API."""

    model_config = SettingsConfigDict(
        env_prefix="ARBORGRID_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_title: str = "ArborGrid SP — Auditoria Urbana Preventiva"
    app_version: str = "3.1.0"
    model_version: str = "segformer-b2-1024-arborgrid-v3-gold"

    manifest_path: Path = Field(default=_DEFAULT_MANIFEST)
    media_dir: Path = Field(default=_DEFAULT_MEDIA)
    gold_state_path: Path = Field(default=_DEFAULT_GOLD_STATE)
    knowledge_base_path: Path = Field(default=_DEFAULT_KB)
    sample_gold_path: Path = Field(default=_DEFAULT_SAMPLE_GOLD)
    samples_dir: Path = Field(default=_DEFAULT_SAMPLES_DIR)
    onnx_model_path: Path = Field(default=_DEFAULT_ONNX)

    llm_api_url: str = Field(default="http://127.0.0.1:8080/v1/chat/completions")
    llm_model_name: str = Field(default="qwen3-1.7b-arborgrid")

    api_token: str | None = Field(default=None)
    require_auth: bool = Field(default=True)
    expose_media_urls: bool = Field(default=False)
    public_base_url: str = Field(default="")


@lru_cache
def get_settings() -> Settings:
    """Retorna instancia cacheada das configuracoes."""
    return Settings()
