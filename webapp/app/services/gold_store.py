"""Armazenamento idempotente dos scores consolidados da camada Gold (Big Data)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class GoldRiskStore:
    """Persiste e consulta registros da tabela Gold consumidos pelo MVP."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = Path(state_path)
        self._lock = threading.Lock()
        self.records: dict[str, dict[str, Any]] = {}
        self.last_batch_id: str | None = None
        self.last_updated_utc: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Estado Gold invalido: esperado objeto JSON.")
            self.last_batch_id = payload.get("last_batch_id")
            self.last_updated_utc = payload.get("last_updated_utc")
            self.records = {
                str(item["geosampa_id"]): item
                for item in payload.get("records", [])
                if "geosampa_id" in item
            }
        except (OSError, ValueError, KeyError) as exc:
            logger.error(
                "Erro ao carregar estado Gold em %s (%s); iniciando store vazio.",
                self.state_path,
                exc,
            )
            self.records = {}
            self.last_batch_id = None
            self.last_updated_utc = None
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            corrupt_path = self.state_path.with_name(
                f"{self.state_path.name}.corrupt-{stamp}"
            )
            try:
                self.state_path.replace(corrupt_path)
            except OSError:
                pass

    def _write_state_locked(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_table": "workspace.gs.gold_fato_risco_arboreo",
            "last_batch_id": self.last_batch_id,
            "last_updated_utc": self.last_updated_utc,
            "total_records": len(self.records),
            "records": list(self.records.values()),
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_path.parent,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(serialized)
                tmp_file.flush()
            os.replace(tmp_path, self.state_path)
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _save(self) -> None:
        with self._lock:
            self._write_state_locked()

    def upsert_batch(
        self, batch_id: str, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Executa MERGE idempotente por geosampa_id e salva estado em disco."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            inserted = 0
            updated = 0
            for item in items:
                gid = str(item["geosampa_id"])
                if gid in self.records:
                    updated += 1
                else:
                    inserted += 1
                enriched = dict(item)
                enriched["geosampa_id"] = gid
                enriched["batch_id"] = batch_id
                enriched["ingested_at_utc"] = now_utc
                enriched["is_demo"] = False
                self.records[gid] = enriched

            self.last_batch_id = batch_id
            self.last_updated_utc = now_utc
            self._write_state_locked()
            total_records = len(self.records)
        return {
            "batch_id": batch_id,
            "inserted": inserted,
            "updated": updated,
            "total_gold_records": total_records,
            "ingested_at_utc": now_utc,
        }

    def seed_demo(
        self, batch_id: str, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Carrega registros de exemplo sem sobrescrever scores reais da camada Gold."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            inserted = 0
            updated = 0
            skipped_real = 0
            for item in items:
                gid = str(item["geosampa_id"])
                existing = self.records.get(gid)
                if existing is None:
                    inserted += 1
                elif existing.get("is_demo") is True:
                    updated += 1
                else:
                    skipped_real += 1
                    continue
                enriched = dict(item)
                enriched["geosampa_id"] = gid
                enriched["batch_id"] = batch_id
                enriched["ingested_at_utc"] = now_utc
                enriched["is_demo"] = True
                self.records[gid] = enriched

            if inserted > 0 or updated > 0:
                self.last_batch_id = batch_id
                self.last_updated_utc = now_utc
                self._write_state_locked()
            total_records = len(self.records)
        return {
            "batch_id": batch_id,
            "inserted": inserted,
            "updated": updated,
            "skipped_real": skipped_real,
            "total_gold_records": total_records,
            "ingested_at_utc": now_utc,
        }

    def list_records(self, priority_tier: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = [dict(r) for r in self.records.values()]
        if priority_tier:
            norm = priority_tier.upper()
            rows = [r for r in rows if str(r.get("priority_tier", "")).upper() == norm]
        rows.sort(
            key=lambda r: float(r.get("composite_risk_score", 0.0)), reverse=True
        )
        return rows

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source_table": "workspace.gs.gold_fato_risco_arboreo",
                "total_records": len(self.records),
                "last_batch_id": self.last_batch_id,
                "last_updated_utc": self.last_updated_utc,
            }
