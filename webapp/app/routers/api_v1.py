"""Endpoints REST v1 do Web App, Big Data, Assistente RAG e Visao ONNX Web."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from ..schemas import AssistantQueryIn, GoldRiskBatchIn
from ..security import require_bearer_token
from ..services.assistant_service import AssistantService
from ..services.gold_store import GoldRiskStore
from ..services.manifest_store import ManifestStore, _safe_is_dir, _safe_is_file


router = APIRouter(prefix="/api/v1", tags=["ArborGrid API v1"])
_SAFE_SAMPLE_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)


def _manifest_store(request: Request) -> ManifestStore:
    return request.app.state.manifest_store


def _gold_store(request: Request) -> GoldRiskStore:
    return request.app.state.gold_store


def _assistant_service(request: Request) -> AssistantService:
    return request.app.state.assistant_service


@router.get("/health", summary="Liveness e readiness do servico")
def health(request: Request) -> dict:
    store = _manifest_store(request)
    gold = _gold_store(request)
    assistant = _assistant_service(request)
    settings = request.app.state.settings
    media_dir = settings.media_dir
    media_mounted = _safe_is_dir(media_dir / "v3_overlays") and _safe_is_dir(
        media_dir / "images"
    )
    return {
        "status": "ok",
        "trees_loaded": len(store.trees),
        "views_loaded": len(store.inspections),
        "gold_records_loaded": len(gold.records),
        "kb_chunks_loaded": len(assistant.chunks),
        "onnx_model_ready": _safe_is_file(settings.onnx_model_path),
        "media_mounted": media_mounted,
        "model_version": store.model_version,
    }


@router.get("/stats", summary="Resumo dinamico de triagem V3 e cobertura territorial")
def stats(request: Request) -> dict:
    store = _manifest_store(request)
    gold = _gold_store(request)
    data = store.build_stats()
    data["gold_layer"] = gold.summary()
    return data


@router.get("/geojson", summary="Camada GeoJSON georreferenciada (WGS84)")
def geojson(
    request: Request,
    region_key: str | None = Query(default=None),
) -> dict:
    store = _manifest_store(request)
    gold = _gold_store(request)
    return store.build_geojson(gold_scores=gold.records, region_key=region_key)


@router.get(
    "/public/map-bootstrap",
    summary="Carga inicial publica da SPA (BI Executivo + Mapa Operacional Leaflet)",
)
def public_map_bootstrap(request: Request) -> dict:
    store = _manifest_store(request)
    gold = _gold_store(request)
    payload = store.build_map_bootstrap(
        gold_scores=gold.records,
        gold_summary=gold.summary(),
    )
    payload["gold_records"] = [
        {k: v for k, v in rec.items() if k != "audit_actor_hash"}
        for rec in gold.list_records()
    ]
    # O front so mostra o botao de cenario de demonstracao quando o servidor permite o seed
    payload["features"] = {"demo_seed_enabled": bool(request.app.state.settings.allow_demo_seed)}
    return payload


@router.get(
    "/manifest",
    summary="Manifesto ArborGrid padronizado (consumo autenticado Big Data)",
    dependencies=[Depends(require_bearer_token)],
)
def get_manifest(
    request: Request,
    region_key: str | None = Query(default=None),
    triage: str | None = Query(default=None, description="CRITICO | PROXIMIDADE | SEGURO"),
) -> dict:
    store = _manifest_store(request)
    return store.build_manifest_response(region_key=region_key, triage=triage)


@router.get(
    "/trees",
    summary="Lista paginada de arvores auditadas",
    dependencies=[Depends(require_bearer_token)],
)
def list_trees(
    request: Request,
    region_key: str | None = Query(default=None),
    triage: str | None = Query(default=None),
    distrito: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    store = _manifest_store(request)
    filtered = store.filter_trees(
        region_key=region_key, triage=triage, distrito=distrito
    )
    page = filtered[offset : offset + limit]
    return {
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
        "items": page,
    }


@router.get(
    "/trees/{geosampa_id}",
    summary="Detalhes e multiplas perspectivas de uma arvore GeoSampa",
    dependencies=[Depends(require_bearer_token)],
)
def get_tree(request: Request, geosampa_id: str) -> dict:
    store = _manifest_store(request)
    tree = store.tree_by_id.get(geosampa_id)
    if not tree:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Arvore geosampa_id='{geosampa_id}' nao encontrada.",
        )
    gold = _gold_store(request).records.get(geosampa_id)
    return {"tree": tree, "gold_risk_score": gold}


@router.get(
    "/inspections",
    summary="Registros normalizados por vista para ingestao incremental Airflow/Databricks",
    dependencies=[Depends(require_bearer_token)],
)
def list_inspections(
    request: Request,
    date: str | None = Query(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Data do lote operacional no formato AAAA-MM-DD",
    ),
    region_key: str | None = Query(default=None),
    triage: str | None = Query(default=None),
    conflict_only: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    store = _manifest_store(request)
    filtered = store.filter_inspections(
        date=date, region_key=region_key, triage=triage, conflict_only=conflict_only
    )
    page = filtered[offset : offset + limit]
    return {
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
        "date_filter": date,
        "model_version": store.model_version,
        "items": page,
    }


@router.get(
    "/inspections/available-dates",
    summary="Datas de lote disponiveis para coleta incremental diaria no Airflow",
    dependencies=[Depends(require_bearer_token)],
)
def list_inspection_dates(request: Request) -> dict:
    store = _manifest_store(request)
    return {"dates": store.available_dates()}


@router.get(
    "/gold/risk-scores",
    summary="Consulta aos scores de risco consolidados da camada Gold",
    dependencies=[Depends(require_bearer_token)],
)
def get_gold_scores(
    request: Request,
    priority_tier: str | None = Query(default=None),
) -> dict:
    gold = _gold_store(request)
    items = gold.list_records(priority_tier=priority_tier)
    return {
        "summary": gold.summary(),
        "total": len(items),
        "items": items,
    }


@router.post(
    "/gold/risk-scores",
    summary="Ingestao idempotente (MERGE) dos scores da tabela Gold do Databricks",
    dependencies=[Depends(require_bearer_token)],
)
def upsert_gold_scores(request: Request, batch: GoldRiskBatchIn) -> dict:
    store = _manifest_store(request)
    gold = _gold_store(request)
    unknown_ids = [
        r.geosampa_id for r in batch.records if r.geosampa_id not in store.tree_by_id
    ]
    if unknown_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"geosampa_id desconhecido no manifesto: {unknown_ids[:5]}",
        )
    payload = [r.model_dump() for r in batch.records]
    result = gold.upsert_batch(batch_id=batch.batch_id, items=payload)
    if hasattr(request.app.state, "metrics_gold_gauge"):
        request.app.state.metrics_gold_gauge.set(len(gold.records))
    return result


@router.post(
    "/gold/seed-demo",
    summary="Carrega lote demonstrativo da camada Gold para o painel BI e Mapa",
)
def seed_gold_demo(request: Request) -> dict:
    settings = request.app.state.settings
    if not settings.allow_demo_seed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Carga demonstrativa desabilitada neste ambiente (ARBORGRID_ALLOW_DEMO_SEED=false).",
        )
    store = _manifest_store(request)
    gold = _gold_store(request)
    sample_path = settings.sample_gold_path
    if not sample_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Arquivo sample_gold_upsert.json nao encontrado.",
        )
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    records = [
        r for r in raw.get("records", []) if str(r.get("geosampa_id")) in store.tree_by_id
    ]
    if not records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Nenhum registro do lote demo corresponde ao manifesto carregado.",
        )
    result = gold.seed_demo(
        batch_id=str(raw.get("batch_id") or "gold-demo-sp"),
        items=records,
    )
    if hasattr(request.app.state, "metrics_gold_gauge"):
        request.app.state.metrics_gold_gauge.set(len(gold.records))
    return result


@router.get(
    "/assistant/status",
    summary="Estado da base RAG e conectividade com servidor Qwen3 GGUF",
)
def assistant_status(request: Request) -> dict:
    return _assistant_service(request).status()


@router.get(
    "/assistant/knowledge-status",
    summary="Estado da sincronizacao RPA da base de normas do Assistente",
)
def assistant_knowledge_status() -> dict:
    env_path = os.getenv("ARBORGRID_KNOWLEDGE_REPORT_PATH", "").strip()
    report_path = (
        Path(env_path)
        if env_path
        else Path(__file__).resolve().parents[3] / "normas-bot" / "saida" / "relatorio_execucao.json"
    )
    if not _safe_is_file(report_path):
        return {"sincronizado": False}
    try:
        dados = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"sincronizado": False}
    if not isinstance(dados, dict):
        return {"sincronizado": False}
    return {
        "sincronizado": True,
        "ultima_sincronizacao": dados.get("fim") or dados.get("inicio"),
        "execucao_id": dados.get("execucao_id"),
        "modo": dados.get("modo"),
        "codigo_saida": dados.get("codigo_saida"),
        "total_fontes": dados.get("total_fontes"),
        "total_trechos": dados.get("total_trechos"),
        "contagem_status_fontes": dados.get("contagem_status_fontes", {}),
        "contagem_status_trechos": dados.get("contagem_status_trechos", {}),
    }


@router.post(
    "/assistant/ask",
    summary="Consulta ao Assistente Tecnico ArborGrid (Qwen3 + RAG + GeoSampa)",
)
def assistant_ask(request: Request, body: AssistantQueryIn) -> dict:
    assistant = _assistant_service(request)
    return assistant.ask(
        question=body.question,
        geosampa_id=body.geosampa_id,
        history=[{"role": h.role, "content": h.content} for h in body.history],
    )


@router.get(
    "/vision/samples",
    summary="Amostras dos 4 territorios e metadados do modelo ONNX Web int8_dyn_matmul",
)
def vision_samples(request: Request) -> dict:
    settings = request.app.state.settings
    samples_dir = settings.samples_dir
    catalog_path = samples_dir / "samples_catalog.json"
    samples = []
    if catalog_path.is_file():
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        for item in raw:
            fname = item.get("filename", "")
            enriched = dict(item)
            enriched["image_url"] = f"/api/v1/vision/samples/{fname}"
            samples.append(enriched)

    onnx_path = settings.onnx_model_path
    model_available = onnx_path.is_file()
    model_size_mb = (
        round(onnx_path.stat().st_size / (1024 * 1024), 2) if model_available else 0.0
    )
    return {
        "model_variant": "onnx_int8_dyn_matmul",
        "model_architecture": "SegFormer-B2 (1024x1024 Multi-Label)",
        "model_available": model_available,
        "model_size_mb": model_size_mb,
        "model_url": "/api/v1/vision/model.onnx",
        "samples": samples,
    }


@router.get(
    "/vision/samples/{filename}",
    summary="Serve imagem de amostra territorial para inferencia ONNX no navegador",
)
def vision_sample_image(request: Request, filename: str):
    if ".." in filename or not _SAFE_SAMPLE_RE.match(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nome de arquivo de amostra invalido.",
        )
    try:
        base_dir = request.app.state.settings.samples_dir.resolve()
        candidate = (base_dir / filename).resolve()
        is_valid = candidate.is_relative_to(base_dir) and candidate.is_file()
    except OSError:
        is_valid = False
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Imagem de amostra nao encontrada.",
        )
    return FileResponse(
        candidate,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get(
    "/vision/model.onnx",
    summary="Serve artefato segformer_v3_int8_dyn_matmul.onnx (51.8 MiB) para onnxruntime-web",
)
def vision_onnx_model(request: Request):
    onnx_path = request.app.state.settings.onnx_model_path
    if not onnx_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Modelo segformer_v3_int8_dyn_matmul.onnx nao encontrado no servidor.",
        )
    return FileResponse(
        onnx_path,
        media_type="application/octet-stream",
        filename="segformer_v3_int8_dyn_matmul.onnx",
        headers={"Cache-Control": "public, max-age=86400"},
    )
