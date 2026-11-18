"""Ferramentas determinísticas do Agente ArborGrid (consulta ao manifest GeoSampa)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.common import DEFAULT_MANIFEST_PATH

_TREE_ID_RE = re.compile(
    r"(?:(?:árvore|arvore|geosampa|id)[_\s:#-]*)(\d{1,7})\b|\b(\d{5,7})\b",
    re.IGNORECASE,
)


class ArborGridInspector:
    """Consulta determinística ao manifest.json para evitar alucinações sobre inspeções."""

    def __init__(self, manifest_path: Path | None = None) -> None:
        self.manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
        self.by_id: dict[str, dict[str, Any]] = {}
        self.district_counts: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        for tree in payload.get("trees", []):
            gid = str(tree.get("geosampa_id", "")).strip()
            if gid:
                self.by_id[gid] = tree
                district = str(tree.get("distrito") or "DESCONHECIDO").upper()
                bucket = self.district_counts.setdefault(
                    district, {"total": 0, "CRITICO": 0, "PROXIMIDADE": 0, "SEGURO": 0}
                )
                bucket["total"] += 1
                triage = self._aggregate_triage(tree)
                macro = (
                    "CRITICO"
                    if triage in {"CRITICO", "CRITICO_CONFIRMADO", "CRITICO_MONO_VISADA", "DIVERGENCIA_PARALAXE"}
                    else triage
                )
                if macro in bucket:
                    bucket[macro] += 1

    @staticmethod
    def _aggregate_triage(tree: dict[str, Any]) -> str:
        triage = tree.get("v3_triage")
        if isinstance(triage, dict) and triage.get("consolidated"):
            return str(triage["consolidated"]).upper()
        if isinstance(triage, str) and triage.strip():
            return triage.strip().upper()
        priority = {"CRITICO": 3, "PROXIMIDADE": 2, "SEGURO": 1}
        best_rank = 0
        best_label = "DESCONHECIDO"
        for view in tree.get("views", []):
            v3 = view.get("v3_analysis") or {}
            label = str(v3.get("triage") or "").upper()
            if priority.get(label, 0) > best_rank:
                best_rank = priority[label]
                best_label = label
        return best_label

    def lookup_tree(self, geosampa_id: str) -> dict[str, Any] | None:
        """Retorna resumo técnico determinístico de uma árvore pelo GeoSampa ID."""
        tree = self.by_id.get(str(geosampa_id).strip())
        if not tree:
            return None

        views_summary = []
        max_contact = 0
        max_wind = 0
        for view in tree.get("views", []):
            v3 = view.get("v3_analysis") or {}
            c_px = int(v3.get("contact_direct_px") or 0)
            w_px = int(v3.get("wind_buffer_px") or 0)
            max_contact = max(max_contact, c_px)
            max_wind = max(max_wind, w_px)
            views_summary.append(
                {
                    "perspective": view.get("perspective"),
                    "capture_date": view.get("capture_date"),
                    "triage": v3.get("triage"),
                    "contact_direct_px": c_px,
                    "wind_buffer_px": w_px,
                }
            )

        return {
            "geosampa_id": str(tree.get("geosampa_id")),
            "distrito": tree.get("distrito"),
            "subprefeitura": tree.get("subprefeitura"),
            "lat": round(float(tree.get("tree_lat", 0.0)), 5),
            "lon": round(float(tree.get("tree_lon", 0.0)), 5),
            "sisgau_last_inspection": tree.get("sisgau_last_inspection"),
            "triage_v3": self._aggregate_triage(tree),
            "max_contact_direct_px": max_contact,
            "max_wind_buffer_px": max_wind,
            "views_count": len(views_summary),
            "views": views_summary,
        }

    def maybe_build_tool_context(self, query: str) -> tuple[str | None, dict[str, Any] | None]:
        """Detecta pedido com GeoSampa ID ou distrito e executa a ferramenta determinística."""
        for match in _TREE_ID_RE.finditer(query):
            candidate = match.group(1) or match.group(2)
            if candidate and candidate in self.by_id:
                record = self.lookup_tree(candidate)
                if record is not None:
                    text = (
                        f"Consulta GeoSampa ID {record['geosampa_id']} | "
                        f"Distrito: {record['distrito']} | Subprefeitura: {record['subprefeitura']} | "
                        f"Coordenadas: ({record['lat']}, {record['lon']}) | "
                        f"Última vistoria SISGAU: {record['sisgau_last_inspection']} | "
                        f"Triagem ArborGrid v3: {record['triage_v3']} | "
                        f"max_contact_direct_px={record['max_contact_direct_px']} | "
                        f"max_wind_buffer_px={record['max_wind_buffer_px']}."
                    )
                    return text, record
        return None, None
