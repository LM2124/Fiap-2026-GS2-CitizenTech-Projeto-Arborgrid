"""Testes automatizados da API ArborGrid, contrato Big Data, Copiloto RAG, ONNX e SPA."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_ROOT = REPO_ROOT / "webapp"


SAMPLE_MANIFEST = {
    "project": "ArborGrid Test Suite",
    "generated_at_utc": "2026-11-09T05:00:00+00:00",
    "updated_at_utc": "2026-11-09T23:07:09+00:00",
    "total_trees_collected": 999,  # propositalmente obsoleto para validar calculo dinamico
    "total_static_api_images": 999,
    "data_sources_attribution": {"tree_coordinates": {"provider": "GeoSampa"}},
    "regions_catalog": {
        "pinheiros": {"label": "Pinheiros (Zona Oeste)", "bbox": [-46.69, -23.56, -46.68, -23.55]}
    },
    "trees": [
        {
            "geosampa_id": "612162",
            "region_key": "pinheiros",
            "region_label": "Pinheiros (Zona Oeste)",
            "distrito": "PINHEIROS",
            "subprefeitura": "PINHEIROS",
            "tree_lat": -23.56808648,
            "tree_lon": -46.68432084,
            "sisgau_last_inspection": "2013-06-12Z",
            "v3_triage": {"consolidated": "CRITICO", "has_paralax_divergence": False},
            "views": [
                {
                    "perspective": "center_front",
                    "pano_id": "pano_secret_id_123",
                    "capture_date": "2024-04",
                    "file_name": "tree_612162_center_front_h34.jpg",
                    "google_maps_url": "https://maps.google.com/?test=1",
                    "v3_analysis": {
                        "triage": "CRITICO",
                        "color_hex": "#ef4444",
                        "copa_px": 250000,
                        "wire_px": 3200,
                        "pole_px": 1800,
                        "contact_direct_px": 950,
                        "wind_buffer_px": 1400,
                        "v3_overlay_file": "ov_v3_tree_612162_center_front_h34.jpg",
                    },
                },
                {
                    "perspective": "side_angle_A",
                    "pano_id": "pano_secret_id_456",
                    "capture_date": "2024-04",
                    "file_name": "tree_612162_side_A.jpg",
                    "google_maps_url": "https://maps.google.com/?test=2",
                    "v3_analysis": {
                        "triage": "SEGURO",
                        "color_hex": "#10b981",
                        "copa_px": 180000,
                        "wire_px": 0,
                        "pole_px": 0,
                        "contact_direct_px": 0,
                        "wind_buffer_px": 0,
                        "v3_overlay_file": "ov_v3_tree_612162_side_A.jpg",
                    },
                },
            ],
        }
    ],
}


def _make_client(tmp_path: Path, api_token: str | None = "test-secret-token") -> TestClient:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(SAMPLE_MANIFEST), encoding="utf-8")
    media_dir = tmp_path / "media"
    (media_dir / "v3_overlays").mkdir(parents=True, exist_ok=True)
    (media_dir / "images").mkdir(parents=True, exist_ok=True)

    kb_file = WEBAPP_ROOT / "data" / "knowledge_base.jsonl"
    if not kb_file.exists():
        kb_file = REPO_ROOT / "llm-assistant" / "base" / "knowledge_base.jsonl"

    sample_gold_file = tmp_path / "sample_gold.json"
    sample_gold_file.write_text(
        json.dumps(
            {
                "batch_id": "demo-batch-test",
                "records": [
                    {
                        "geosampa_id": "612162",
                        "distrito": "PINHEIROS",
                        "composite_risk_score": 88.0,
                        "priority_tier": "URGENTE",
                        "wind_gust_kmh": 70.0,
                        "precipitation_mm": 30.0,
                        "recommended_action": "Poda preventiva prioritaria Enel + Subprefeitura",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    samples_dir = WEBAPP_ROOT / "data" / "samples"
    onnx_file = REPO_ROOT / "arborgrid-vision" / "models" / "segformer_v3_int8_dyn_matmul.onnx"

    settings = Settings(
        manifest_path=manifest_file,
        media_dir=media_dir,
        gold_state_path=tmp_path / "gold_state.json",
        knowledge_base_path=kb_file,
        sample_gold_path=sample_gold_file,
        samples_dir=samples_dir,
        onnx_model_path=onnx_file,
        api_token=api_token,
        require_auth=True,
        expose_media_urls=False,
    )
    app = create_app(custom_settings=settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_public_endpoints_and_dynamic_stats(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        r_health = client.get("/api/v1/health")
        assert r_health.status_code == 200
        assert r_health.json()["trees_loaded"] == 1
        assert r_health.json()["views_loaded"] == 2
        assert r_health.json()["kb_chunks_loaded"] >= 16

        r_stats = client.get("/api/v1/stats")
        assert r_stats.status_code == 200
        assert r_stats.json()["total_trees"] == 1
        assert r_stats.json()["total_views"] == 2
        assert r_stats.json()["tree_triage_counts"]["CRITICO"] == 1

        r_boot = client.get("/api/v1/public/map-bootstrap")
        assert r_boot.status_code == 200
        assert len(r_boot.json()["trees"]) == 1

        for spa_route in ("/", "/mapa", "/visao", "/assistente", "/copiloto"):
            r_spa = client.get(spa_route)
            assert r_spa.status_code == 200
            assert "ArborGrid SP" in r_spa.text


def test_bearer_auth_fail_closed_and_success(tmp_path: Path) -> None:
    with _make_client(tmp_path, api_token=None) as client_no_cfg:
        r_missing_cfg = client_no_cfg.get(
            "/api/v1/inspections", headers={"Authorization": "Bearer any"}
        )
        assert r_missing_cfg.status_code == 503

    with _make_client(tmp_path, api_token="expected-token") as client:
        assert client.get("/api/v1/inspections").status_code == 401
        assert (
            client.get(
                "/api/v1/inspections",
                headers={"Authorization": "Bearer wrong-token"},
            ).status_code
            == 401
        )
        r_ok = client.get(
            "/api/v1/inspections",
            headers={"Authorization": "Bearer expected-token"},
        )
        assert r_ok.status_code == 200
        body = r_ok.json()
        assert body["total"] == 2
        first = body["items"][0]
        assert first["latitude"] == -23.56809
        assert first["longitude"] == -46.68432
        assert first["overlay_url"] is None
        assert "pano_secret_id_123" not in json.dumps(body)


def test_inspections_date_filter_determinism_and_gold_upsert(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer expected-token"}
    with _make_client(tmp_path, api_token="expected-token") as client:
        dates = client.get("/api/v1/inspections/available-dates", headers=headers).json()["dates"]
        assert dates == [{"date": "2026-11-10", "records": 2}]

        r_day1 = client.get("/api/v1/inspections?date=2026-11-10", headers=headers)
        r_day1_again = client.get("/api/v1/inspections?date=2026-11-10", headers=headers)
        assert r_day1.status_code == 200
        assert r_day1.json()["items"][0]["detection_id"] == r_day1_again.json()["items"][0]["detection_id"]

        r_empty = client.get("/api/v1/inspections?date=2026-10-01", headers=headers)
        assert r_empty.status_code == 200
        assert r_empty.json()["items"] == []

        r_conflict = client.get("/api/v1/inspections?conflict_only=true", headers=headers)
        assert r_conflict.json()["total"] == 1

        batch_payload = {
            "batch_id": "gold-2026-11-14-lote1",
            "records": [
                {
                    "geosampa_id": "612162",
                    "distrito": "PINHEIROS",
                    "composite_risk_score": 91.5,
                    "priority_tier": "URGENTE",
                    "wind_gust_kmh": 62.4,
                    "precipitation_mm": 18.0,
                    "recommended_action": "Acionar poda preventiva Enel/Subprefeitura",
                }
            ],
        }
        r_post_1 = client.post("/api/v1/gold/risk-scores", json=batch_payload, headers=headers)
        assert r_post_1.status_code == 200
        assert r_post_1.json()["inserted"] == 1
        assert r_post_1.json()["updated"] == 0

        # Idempotencia no reenvio (MERGE)
        r_post_2 = client.post("/api/v1/gold/risk-scores", json=batch_payload, headers=headers)
        assert r_post_2.status_code == 200
        assert r_post_2.json()["inserted"] == 0
        assert r_post_2.json()["updated"] == 1

        r_geo = client.get("/api/v1/geojson")
        assert r_geo.status_code == 200
        props = r_geo.json()["features"][0]["properties"]
        assert props["gold_priority_tier"] == "URGENTE"
        assert props["gold_composite_risk_score"] == 91.5


def test_seed_demo_assistant_and_vision_endpoints(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        r_seed = client.post("/api/v1/gold/seed-demo")
        assert r_seed.status_code == 200
        assert r_seed.json()["total_gold_records"] == 1

        r_status = client.get("/api/v1/assistant/status")
        assert r_status.status_code == 200
        assert r_status.json()["kb_loaded"] is True
        assert r_status.json()["chunks_count"] >= 16

        r_ask_tree = client.post(
            "/api/v1/assistant/ask",
            json={
                "question": "Qual a situacao da arvore #612162 encostando em fios sem faisca?",
                "geosampa_id": "612162",
            },
        )
        assert r_ask_tree.status_code == 200
        body_tree = r_ask_tree.json()
        assert body_tree["structured"]["status"] == "respondido"
        assert "ARBORGRID-TRIAGEM" in body_tree["structured"]["fontes"]
        assert body_tree["tree_context"]["geosampa_id"] == "612162"
        assert "CRITICO_CONFIRMADO" not in body_tree["answer"]
        assert "DIVERGENCIA_PARALAXE" not in body_tree["answer"]

        r_ask_prune = client.post(
            "/api/v1/assistant/ask",
            json={"question": "Posso podar?"},
        )
        assert r_ask_prune.status_code == 200
        body_prune = r_ask_prune.json()
        assert body_prune["structured"]["status"] == "respondido"
        assert body_prune["structured"]["pode_podar_por_conta"] is False
        assert "LEI-17794-ART05" in body_prune["structured"]["fontes"]

        r_ask_followup = client.post(
            "/api/v1/assistant/ask",
            json={
                "question": "Posso podar?",
                "history": [
                    {
                        "role": "user",
                        "content": "Analise a arvore #612162 encostando na fiacao.",
                    },
                    {
                        "role": "assistant",
                        "content": "Solicite vistoria preventiva pelo SP156.",
                    },
                ],
            },
        )
        assert r_ask_followup.status_code == 200
        body_followup = r_ask_followup.json()
        assert body_followup["tree_context"]["geosampa_id"] == "612162"
        assert body_followup["structured"]["pode_podar_por_conta"] is False

        r_ask_oos = client.post(
            "/api/v1/assistant/ask",
            json={"question": "Qual a receita de bolo de cenoura com chocolate?"},
        )
        assert r_ask_oos.status_code == 200
        assert r_ask_oos.json()["structured"]["status"] == "nao_consta"
        assert r_ask_oos.json()["structured"]["nivel_risco"] == "fora_de_escopo"

        r_samples = client.get("/api/v1/vision/samples")
        assert r_samples.status_code == 200
        assert len(r_samples.json()["samples"]) == 4

        first_file = r_samples.json()["samples"][0]["filename"]
        r_img = client.get(f"/api/v1/vision/samples/{first_file}")
        assert r_img.status_code == 200
        assert client.get("/api/v1/vision/samples/..secret.jpg").status_code == 400


def test_prometheus_metrics_and_media_safety(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        client.get("/api/v1/health")
        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert "arborgrid_http_requests_total" in r_metrics.text
        assert "arborgrid_http_request_duration_seconds" in r_metrics.text
        assert "arborgrid_inprogress_requests" in r_metrics.text

        assert client.get("/media/invalid_dir/test.jpg").status_code == 404
        assert client.get("/media/v3_overlays/..secret.txt").status_code == 400
        assert client.get("/api/v1/does-not-exist").status_code == 404


def test_real_manifest_counts_and_v3_triage_normalization(tmp_path: Path) -> None:
    repo_manifest = WEBAPP_ROOT / "data" / "manifest.json"
    if not repo_manifest.exists():
        repo_manifest = (
            REPO_ROOT
            / "arborgrid-vision"
            / "dataset"
            / "sp_geosampa_api"
            / "manifest.json"
        )
    if not repo_manifest.exists():
        return
    settings = Settings(
        manifest_path=repo_manifest,
        media_dir=REPO_ROOT / "arborgrid-vision" / "dataset" / "sp_geosampa_api",
        gold_state_path=tmp_path / "gold_real.json",
        api_token="real-token",
        require_auth=True,
    )
    app = create_app(custom_settings=settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        stats = client.get("/api/v1/stats").json()
        assert stats["total_trees"] == 271
        assert stats["total_views"] == 536
        assert stats["tree_triage_counts"]["CRITICO"] == 106
        assert stats["tree_triage_counts"]["PROXIMIDADE"] == 42
        assert stats["tree_triage_counts"]["SEGURO"] == 123
        assert stats["detailed_triage_counts"]["DIVERGENCIA_PARALAXE"] == 45
        assert stats["detailed_triage_counts"]["CRITICO_CONFIRMADO"] == 59

        headers = {"Authorization": "Bearer real-token"}
        dates = client.get("/api/v1/inspections/available-dates", headers=headers).json()["dates"]
        assert sum(d["records"] for d in dates) == 536

        r_crit = client.get("/api/v1/trees?triage=CRITICO&limit=500", headers=headers).json()
        assert r_crit["total"] == 106

        r_para = client.get("/api/v1/trees?triage=DIVERGENCIA_PARALAXE&limit=500", headers=headers).json()
        assert r_para["total"] == 45

        r_onnx = client.head("/api/v1/vision/model.onnx")
        if settings.onnx_model_path.is_file():
            assert r_onnx.status_code in (200, 405)
            r_get_onnx = client.get("/api/v1/vision/model.onnx")
            assert r_get_onnx.status_code == 200
            assert len(r_get_onnx.content) > 50_000_000
