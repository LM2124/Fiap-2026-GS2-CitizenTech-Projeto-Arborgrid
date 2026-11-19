"""Carregamento, indexacao e normalizacao do manifesto V3 do ArborGrid."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import quote
import logging
from collections import Counter
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
_media_check_warned = False


def _warn_once_media_oserror(path: Path, exc: OSError) -> None:
    global _media_check_warned
    if not _media_check_warned:
        _media_check_warned = True
        logger.warning("Falha ao verificar midia em %s: %s", path, exc)


def _safe_is_file(path: Path) -> bool:
    """Verifica se o arquivo existe tratando OSError/PermissionError como indisponivel."""
    try:
        return path.is_file()
    except OSError as exc:
        _warn_once_media_oserror(path, exc)
        return False


def _safe_is_dir(path: Path) -> bool:
    """Verifica se o diretorio existe tratando OSError/PermissionError como indisponivel."""
    try:
        return path.is_dir()
    except OSError as exc:
        _warn_once_media_oserror(path, exc)
        return False


REGION_CAMPAIGN_DATES: dict[str, str] = {
    "pinheiros": "2026-11-10",
    "alto_da_mooca": "2026-11-11",
    "vila_mariana": "2026-11-12",
    "avenida_paulista": "2026-11-13",
}


def _short_sha256(raw: str, length: int = 16) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _deterministic_timestamp(seed: str, region_key: str) -> str:
    """Gera timestamp ISO 8601 (-03:00) deterministico para lotes diarios."""
    campaign_date = REGION_CAMPAIGN_DATES.get(region_key, "2026-11-14")
    digest = int(_short_sha256(seed, 8), 16)
    hour = 8 + (digest % 10)  # 08h..17h (janela diurna operacional)
    minute = (digest // 10) % 60
    second = (digest // 600) % 60
    return f"{campaign_date}T{hour:02d}:{minute:02d}:{second:02d}-03:00"


def compute_physical_confidence(v3: dict[str, Any]) -> float:
    """Calcula score deterministico [0, 1] a partir da evidencia geometrica V3."""
    triage = v3.get("triage", "SEGURO")
    contact_px = int(v3.get("contact_direct_px", 0) or 0)
    wind_px = int(v3.get("wind_buffer_px", 0) or 0)
    wire_px = int(v3.get("wire_px", 0) or 0)
    copa_px = int(v3.get("copa_px", 0) or 0)

    if triage == "CRITICO":
        strength = min(1.0, (contact_px / 1800.0) + (wire_px / 6000.0) * 0.25)
        score = 0.72 + 0.26 * strength
    elif triage == "PROXIMIDADE":
        strength = min(1.0, (wind_px / 1500.0) + (wire_px / 6000.0) * 0.25)
        score = 0.58 + 0.27 * strength
    else:
        copa_factor = min(1.0, copa_px / 80000.0)
        wire_penalty = min(0.20, (wire_px / 5000.0) * 0.15)
        score = 0.80 + 0.16 * copa_factor - wire_penalty

    return round(max(0.05, min(0.99, score)), 4)


def _normalize_macro_triage(raw_status: str | None) -> str:
    """Normaliza sub-status detalhados V3 para macro-categorias (CRITICO, PROXIMIDADE, SEGURO)."""
    norm = str(raw_status or "").strip().upper()
    if norm.startswith("CRITICO") or norm == "DIVERGENCIA_PARALAXE":
        return "CRITICO"
    if norm.startswith("PROXIMIDADE"):
        return "PROXIMIDADE"
    if norm.startswith("SEGURO"):
        return "SEGURO"
    return norm or "SEGURO"


def _tree_raw_consolidated_triage(tree: dict[str, Any]) -> str:
    return str((tree.get("v3_triage") or {}).get("consolidated") or "SEGURO").upper()


def _tree_consolidated_triage(tree: dict[str, Any]) -> str:
    return _normalize_macro_triage(_tree_raw_consolidated_triage(tree))


def _is_sisgau_outdated(sisgau_raw: Any) -> tuple[bool, bool]:
    """Retorna (nunca_inspecionada, defasada_ou_sem_vistoria)."""
    if sisgau_raw is None:
        return True, True
    text = str(sisgau_raw).strip()
    if not text or text.lower() in {"null", "none", "n/a", "sem registro"}:
        return True, True
    try:
        year = int(text[:4])
    except ValueError:
        return False, True
    return False, year < 2024


class ManifestStore:
    """Repositorio em memoria dos dados geoespaciais e de visao computacional V3."""

    def __init__(
        self,
        manifest_path: Path,
        model_version: str,
        public_base_url: str = "",
        expose_media_urls: bool = False,
        media_dir: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.model_version = model_version
        self.public_base_url = public_base_url.rstrip("/")
        self.expose_media_urls = expose_media_urls
        self.media_dir = Path(media_dir) if media_dir is not None else self.manifest_path.parent
        self.raw_manifest: dict[str, Any] = {}
        self.trees: list[dict[str, Any]] = []
        self.tree_by_id: dict[str, dict[str, Any]] = {}
        self.regions_catalog: dict[str, Any] = {}
        self.inspections: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        if not self.manifest_path.exists():
            self.raw_manifest = {}
            self.trees = []
            self.tree_by_id = {}
            self.regions_catalog = {}
            self.inspections = []
            return

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.raw_manifest = payload
        self.trees = payload.get("trees", [])
        self.regions_catalog = payload.get("regions_catalog", {})
        self.tree_by_id = {str(t.get("geosampa_id")): t for t in self.trees}
        self.inspections = self._build_inspections()

    def _build_inspections(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for tree in self.trees:
            geosampa_id = str(tree.get("geosampa_id", ""))
            region_key = str(tree.get("region_key", "desconhecida"))
            tree_lat = round(float(tree.get("tree_lat", 0.0)), 5)
            tree_lon = round(float(tree.get("tree_lon", 0.0)), 5)
            raw_tree_triage = _tree_raw_consolidated_triage(tree)
            macro_tree_triage = _normalize_macro_triage(raw_tree_triage)

            for view in tree.get("views", []):
                v3 = view.get("v3_analysis") or {}
                file_name = str(view.get("file_name", ""))
                perspective = str(view.get("perspective", ""))
                overlay_file = str(v3.get("v3_overlay_file", ""))
                triage_view = str(v3.get("triage", "SEGURO"))
                seed = f"{geosampa_id}:{perspective}:{file_name}"
                overlay_url = None
                if self.expose_media_urls and overlay_file:
                    prefix = self.public_base_url
                    overlay_url = f"{prefix}/media/v3_overlays/{overlay_file}{self._overlay_version_suffix()}"

                rows.append(
                    {
                        "detection_id": f"arb-{_short_sha256(seed)}",
                        "image_id": Path(file_name).stem if file_name else seed,
                        "geosampa_id": geosampa_id,
                        "region_key": region_key,
                        "region_label": tree.get("region_label", region_key),
                        "distrito": tree.get("distrito"),
                        "subprefeitura": tree.get("subprefeitura"),
                        "sisgau_last_inspection": tree.get("sisgau_last_inspection"),
                        "perspective": perspective,
                        "streetview_capture_month": view.get("capture_date"),
                        "captured_at": _deterministic_timestamp(seed, region_key),
                        "latitude": tree_lat,
                        "longitude": tree_lon,
                        "triage_class": triage_view,
                        "tree_consolidated_triage": raw_tree_triage,
                        "tree_macro_triage": macro_tree_triage,
                        "conflict_category": (
                            "tree_wire_conflict"
                            if triage_view in ("CRITICO", "PROXIMIDADE")
                            else "no_conflict"
                        ),
                        "confidence": compute_physical_confidence(v3),
                        "model_version": self.model_version,
                        "copa_px": int(v3.get("copa_px", 0) or 0),
                        "wire_px": int(v3.get("wire_px", 0) or 0),
                        "pole_px": int(v3.get("pole_px", 0) or 0),
                        "contact_direct_px": int(v3.get("contact_direct_px", 0) or 0),
                        "wind_buffer_px": int(v3.get("wind_buffer_px", 0) or 0),
                        "pano_id_hash": _short_sha256(str(view.get("pano_id", "")), 12),
                        "overlay_file": overlay_file or None,
                        "overlay_url": overlay_url,
                        "google_maps_url": view.get("google_maps_url"),
                    }
                )
        return rows

    def filter_trees(
        self,
        region_key: str | None = None,
        triage: str | None = None,
        distrito: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self.trees
        if region_key:
            results = [t for t in results if t.get("region_key") == region_key]
        if triage:
            norm = triage.upper()
            results = [
                t
                for t in results
                if _tree_consolidated_triage(t) == norm
                or _tree_raw_consolidated_triage(t) == norm
            ]
        if distrito:
            norm_d = distrito.upper()
            results = [
                t for t in results if str(t.get("distrito", "")).upper() == norm_d
            ]
        return results

    def filter_inspections(
        self,
        date: str | None = None,
        region_key: str | None = None,
        triage: str | None = None,
        conflict_only: bool = False,
    ) -> list[dict[str, Any]]:
        results = self.inspections
        if date:
            results = [r for r in results if str(r["captured_at"]).startswith(date)]
        if region_key:
            results = [r for r in results if r["region_key"] == region_key]
        if triage:
            norm = triage.upper()
            results = [r for r in results if r["triage_class"] == norm]
        if conflict_only:
            results = [
                r for r in results if r["conflict_category"] == "tree_wire_conflict"
            ]
        return results

    def available_dates(self) -> list[dict[str, Any]]:
        counts = Counter(str(r["captured_at"])[:10] for r in self.inspections)
        return [{"date": d, "records": counts[d]} for d in sorted(counts)]

    def _overlay_version_suffix(self) -> str:
        """Sufixo ?v= dos overlays: muda quando as imagens sao regeradas (render_web_overlays.py),
        para o navegador nao servir a versao antiga do cache (Cache-Control de 1 h na midia)."""
        version = str(self.raw_manifest.get("overlays_render_version") or "").strip()
        return f"?v={quote(version)}" if version else ""

    def build_stats(self) -> dict[str, Any]:
        tree_triages = Counter(_tree_consolidated_triage(t) for t in self.trees)
        detailed_triages = Counter(_tree_raw_consolidated_triage(t) for t in self.trees)
        view_triages = Counter(r["triage_class"] for r in self.inspections)

        by_region: dict[str, dict[str, int]] = {}
        breakdown_acc: dict[str, dict[str, Any]] = {}
        sisgau_never = 0
        sisgau_outdated = 0

        for tree in self.trees:
            rk = str(tree.get("region_key", "desconhecida"))
            macro = _tree_consolidated_triage(tree)
            raw = _tree_raw_consolidated_triage(tree)
            never_insp, outdated = _is_sisgau_outdated(tree.get("sisgau_last_inspection"))
            if never_insp:
                sisgau_never += 1
            if outdated:
                sisgau_outdated += 1

            bucket = by_region.setdefault(
                rk, {"trees": 0, "CRITICO": 0, "PROXIMIDADE": 0, "SEGURO": 0}
            )
            bucket["trees"] += 1
            bucket[macro] = bucket.get(macro, 0) + 1

            cat = self.regions_catalog.get(rk, {})
            bd = breakdown_acc.setdefault(
                rk,
                {
                    "region_key": rk,
                    "region_label": cat.get("label") or tree.get("region_label") or rk,
                    "grid_type": cat.get("grid_type", "AÉREA CONVENCIONAL"),
                    "is_negative_control": bool(cat.get("is_negative_control", False)),
                    "total_geosampa_trees": int(cat.get("total_geosampa_trees") or 0),
                    "total_trees": 0,
                    "total_views": 0,
                    "critico_total": 0,
                    "critico_confirmado": 0,
                    "divergencia_paralaxe": 0,
                    "proximidade": 0,
                    "seguro": 0,
                    "sum_contact_px": 0,
                    "max_contact_px": 0,
                    "sisgau_never_inspected_count": 0,
                    "sisgau_outdated_count": 0,
                },
            )
            views = tree.get("views", [])
            max_contact = max(
                (int((v.get("v3_analysis") or {}).get("contact_direct_px", 0) or 0) for v in views),
                default=0,
            )
            bd["total_trees"] += 1
            bd["total_views"] += len(views)
            if macro == "CRITICO":
                bd["critico_total"] += 1
            elif macro == "PROXIMIDADE":
                bd["proximidade"] += 1
            else:
                bd["seguro"] += 1

            if raw in ("CRITICO_CONFIRMADO", "CRITICO_MONO_VISADA", "CRITICO"):
                bd["critico_confirmado"] += 1
            elif raw == "DIVERGENCIA_PARALAXE":
                bd["divergencia_paralaxe"] += 1

            bd["sum_contact_px"] += max_contact
            bd["max_contact_px"] = max(bd["max_contact_px"], max_contact)
            if never_insp:
                bd["sisgau_never_inspected_count"] += 1
            if outdated:
                bd["sisgau_outdated_count"] += 1

        district_breakdown: list[dict[str, Any]] = []
        for bd in breakdown_acc.values():
            n = max(bd["total_trees"], 1)
            row = {k: v for k, v in bd.items() if k != "sum_contact_px"}
            row["avg_contact_px"] = round(bd["sum_contact_px"] / n, 1)
            row["critical_rate_pct"] = round(100.0 * bd["critico_total"] / n, 1)
            row["sisgau_gap_pct"] = round(100.0 * bd["sisgau_outdated_count"] / n, 1)
            district_breakdown.append(row)

        return {
            "project": self.raw_manifest.get(
                "project", "ArborGrid - FIAP Global Solution CitizenTech (4ES)"
            ),
            "model_version": self.model_version,
            "total_trees": len(self.trees),
            "total_views": len(self.inspections),
            "tree_triage_counts": dict(tree_triages),
            "detailed_triage_counts": dict(detailed_triages),
            "view_triage_counts": dict(view_triages),
            "sisgau_never_inspected_count": sisgau_never,
            "sisgau_outdated_count": sisgau_outdated,
            "regions": by_region,
            "district_breakdown": district_breakdown,
            "available_batch_dates": self.available_dates(),
        }

    def build_map_bootstrap(
        self,
        gold_scores: dict[str, dict[str, Any]],
        gold_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Monta payload enxuto e enriquecido para a SPA React + Leaflet."""
        overlays_dir = self.media_dir / "v3_overlays"
        images_dir = self.media_dir / "images"
        has_media_root = _safe_is_dir(overlays_dir) or _safe_is_dir(images_dir)

        public_trees: list[dict[str, Any]] = []
        for tree in self.trees:
            v3_raw = dict(tree.get("v3_triage") or {})
            raw_status = _tree_raw_consolidated_triage(tree)
            macro_status = _normalize_macro_triage(raw_status)
            v3_enriched = {
                "consolidated": macro_status,
                "raw_consolidated": raw_status,
                "front_triage": v3_raw.get("front_triage", macro_status),
                "front_color": v3_raw.get("front_color"),
                "side_triage": v3_raw.get("side_triage"),
                "side_color": v3_raw.get("side_color"),
                "has_paralax_divergence": bool(v3_raw.get("has_paralax_divergence", False)),
                "has_single_view": bool(v3_raw.get("has_single_view", False)),
            }

            public_views: list[dict[str, Any]] = []
            for view in tree.get("views", []):
                v3_view = view.get("v3_analysis") or {}
                ov_file = str(v3_view.get("v3_overlay_file") or "")
                img_file = str(view.get("file_name") or "")
                ov_exists = bool(
                    ov_file
                    and (self.expose_media_urls or (has_media_root and _safe_is_file(overlays_dir / ov_file)))
                )
                img_exists = bool(
                    img_file
                    and (self.expose_media_urls or (has_media_root and _safe_is_file(images_dir / img_file)))
                )
                prefix = self.public_base_url
                public_views.append(
                    {
                        "perspective": view.get("perspective"),
                        "capture_date": view.get("capture_date"),
                        "heading": view.get("heading"),
                        "distance_m": view.get("distance_m"),
                        "google_maps_url": view.get("google_maps_url"),
                        "overlay_url": (
                            f"{prefix}/media/v3_overlays/{ov_file}{self._overlay_version_suffix()}" if ov_exists else None
                        ),
                        "image_url": f"{prefix}/media/images/{img_file}" if img_exists else None,
                        "v3_analysis": v3_view,
                    }
                )

            public_trees.append(
                {
                    "geosampa_id": str(tree.get("geosampa_id", "")),
                    "region_key": tree.get("region_key"),
                    "region_label": tree.get("region_label"),
                    "distrito": tree.get("distrito"),
                    "subprefeitura": tree.get("subprefeitura"),
                    "tree_lat": tree.get("tree_lat"),
                    "tree_lon": tree.get("tree_lon"),
                    "sisgau_last_inspection": tree.get("sisgau_last_inspection"),
                    "v3_triage": v3_enriched,
                    "views": public_views,
                }
            )

        return {
            "stats": self.build_stats(),
            "regions_catalog": self.regions_catalog,
            "trees": public_trees,
            "gold_layer": gold_summary,
            "gold_scores_by_tree": gold_scores,
        }

    def build_manifest_response(
        self,
        region_key: str | None = None,
        triage: str | None = None,
    ) -> dict[str, Any]:
        filtered = self.filter_trees(region_key=region_key, triage=triage)
        view_count = sum(len(t.get("views", [])) for t in filtered)
        return {
            "project": self.raw_manifest.get("project"),
            "generated_at_utc": self.raw_manifest.get("generated_at_utc"),
            "updated_at_utc": self.raw_manifest.get("updated_at_utc"),
            "model_version": self.model_version,
            "total_trees": len(filtered),
            "total_views": view_count,
            "data_sources_attribution": self.raw_manifest.get(
                "data_sources_attribution", {}
            ),
            "regions_catalog": self.regions_catalog,
            "trees": filtered,
        }

    def build_geojson(
        self,
        gold_scores: dict[str, dict[str, Any]] | None = None,
        region_key: str | None = None,
    ) -> dict[str, Any]:
        gold_map = gold_scores or {}
        features: list[dict[str, Any]] = []
        for tree in self.filter_trees(region_key=region_key):
            gid = str(tree.get("geosampa_id"))
            v3 = tree.get("v3_triage") or {}
            raw_consolidated = _tree_raw_consolidated_triage(tree)
            macro_triage = _normalize_macro_triage(raw_consolidated)
            gold = gold_map.get(gid)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            round(float(tree.get("tree_lon", 0.0)), 6),
                            round(float(tree.get("tree_lat", 0.0)), 6),
                        ],
                    },
                    "properties": {
                        "geosampa_id": gid,
                        "region_key": tree.get("region_key"),
                        "region_label": tree.get("region_label"),
                        "distrito": tree.get("distrito"),
                        "subprefeitura": tree.get("subprefeitura"),
                        "sisgau_last_inspection": tree.get("sisgau_last_inspection"),
                        "v3_consolidated_triage": raw_consolidated,
                        "v3_macro_triage": macro_triage,
                        "has_paralax_divergence": v3.get(
                            "has_paralax_divergence", False
                        ),
                        "views_count": len(tree.get("views", [])),
                        "gold_composite_risk_score": (
                            gold.get("composite_risk_score") if gold else None
                        ),
                        "gold_priority_tier": (
                            gold.get("priority_tier") if gold else None
                        ),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}
