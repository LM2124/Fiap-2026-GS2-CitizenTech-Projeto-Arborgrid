"""Aplicacao FastAPI principal do ArborGrid SP (SPA + API + Observabilidade)."""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import Settings, get_settings
from .metrics import (
    GOLD_SCORES_GAUGE,
    HTTP_ERRORS_TOTAL,
    HTTP_INPROGRESS,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    TREES_LOADED_GAUGE,
)
from .routers.api_v1 import router as api_v1_router
from .services.assistant_service import AssistantService
from .services.gold_store import GoldRiskStore
from .services.manifest_store import ManifestStore


logging.basicConfig(
    level=os.getenv("ARBORGRID_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")

ALLOWED_MEDIA_FOLDERS = frozenset(
    {"images", "v3_overlays", "v2_overlays", "gold_overlays", "gpu_overlays"}
)
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
STATIC_DIR = Path(__file__).resolve().parent / "static"

def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return str(route.path)
    return "unmatched"


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """Factory da aplicacao FastAPI para execucao e testes isolados."""
    settings = custom_settings or get_settings()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        manifest_store = ManifestStore(
            manifest_path=settings.manifest_path,
            model_version=settings.model_version,
            public_base_url=settings.public_base_url,
            expose_media_urls=settings.expose_media_urls,
            media_dir=settings.media_dir,
        )
        gold_store = GoldRiskStore(state_path=settings.gold_state_path)
        assistant_service = AssistantService(
            kb_path=settings.knowledge_base_path,
            manifest_store=manifest_store,
            gold_store=gold_store,
            llm_api_url=settings.llm_api_url,
            llm_model_name=settings.llm_model_name,
            llm_timeout_seconds=settings.llm_timeout_seconds,
        )
        app_instance.state.settings = settings
        app_instance.state.manifest_store = manifest_store
        app_instance.state.gold_store = gold_store
        app_instance.state.assistant_service = assistant_service
        app_instance.state.metrics_gold_gauge = GOLD_SCORES_GAUGE
        TREES_LOADED_GAUGE.set(len(manifest_store.trees))
        GOLD_SCORES_GAUGE.set(len(gold_store.records))
        yield

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "Plataforma CitizenTech para auditoria preventiva de conflito entre "
            "arborizacao urbana e rede aerea de distribuicao eletrica em Sao Paulo."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observability_and_security_middleware(request: Request, call_next):
        start = time.perf_counter()
        HTTP_INPROGRESS.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            route = _route_label(request)
            HTTP_ERRORS_TOTAL.labels(request.method, route, "5xx").inc()
            HTTP_REQUESTS_TOTAL.labels(request.method, route, "500").inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            HTTP_INPROGRESS.dec()
            route = _route_label(request)
            HTTP_REQUEST_DURATION.labels(request.method, route).observe(elapsed)

        HTTP_REQUESTS_TOTAL.labels(request.method, route, str(status_code)).inc()
        if status_code >= 500:
            HTTP_ERRORS_TOTAL.labels(request.method, route, "5xx").inc()
        elif status_code >= 400:
            HTTP_ERRORS_TOTAL.labels(request.method, route, "4xx").inc()

        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin"
        )
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Embedder-Policy", "credentialless")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        return response

    app.include_router(api_v1_router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/media/{folder}/{filename}", include_in_schema=False)
    def serve_media(folder: str, filename: str, request: Request):
        if folder not in ALLOWED_MEDIA_FOLDERS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Pasta de midia invalida."
            )
        if ".." in filename or not SAFE_FILENAME_RE.match(filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Nome de arquivo invalido."
            )
        try:
            base_dir = (request.app.state.settings.media_dir / folder).resolve()
            candidate = (base_dir / filename).resolve()
            is_valid = candidate.is_relative_to(base_dir) and candidate.is_file()
        except OSError as exc:
            logger.warning("Erro ao acessar midia %s/%s: %s", folder, filename, exc)
            is_valid = False
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Midia nao encontrada no volume montado.",
            )
        return FileResponse(
            candidate,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_static_or_fallback(full_path: str):
        if full_path.startswith(("api/", "media/", "metrics", "docs", "redoc", "openapi.json")):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nao encontrado.")

        if full_path in {"mapa", "visao", "copiloto", "assistente"}:
            index_file = STATIC_DIR / "index.html"
            return HTMLResponse(index_file.read_text(encoding="utf-8"))

        static_root = STATIC_DIR.resolve()
        candidate = (static_root / full_path).resolve()
        if full_path and candidate.is_relative_to(static_root) and candidate.is_file():
            suffix = candidate.suffix.lower()
            media_type = None
            if suffix == ".wasm":
                media_type = "application/wasm"
            elif suffix in {".js", ".mjs"}:
                media_type = "text/javascript"
            # Arquivos de assets/ levam hash do conteudo no nome (build do Vite): podem ficar em cache "para sempre"
            cache_control = (
                "public, max-age=31536000, immutable"
                if full_path.startswith("assets/")
                else "public, max-age=3600"
            )
            return FileResponse(
                candidate,
                media_type=media_type,
                headers={"Cache-Control": cache_control},
            )

        index_file = STATIC_DIR / "index.html"
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    return app


app = create_app()
