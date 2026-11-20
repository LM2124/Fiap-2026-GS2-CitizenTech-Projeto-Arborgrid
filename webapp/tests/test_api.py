"""Testes automatizados da API ArborGrid, contrato Big Data, Copiloto RAG, ONNX e SPA."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.privacy import mask_personal_data


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


def _make_client(
    tmp_path: Path,
    api_token: str | None = "test-secret-token",
    allow_demo_seed: bool = True,
) -> TestClient:
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
        allow_demo_seed=allow_demo_seed,
        llm_api_url="http://127.0.0.1:9/v1/chat/completions",
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
        assert "llm_endpoint" not in r_status.json()

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


def test_media_permission_error_degrades_gracefully(tmp_path: Path, monkeypatch) -> None:
    orig_is_file = Path.is_file

    def _fake_is_file(path_obj: Path) -> bool:
        if "v3_overlays" in str(path_obj):
            raise PermissionError(13, "Permission denied", str(path_obj))
        return orig_is_file(path_obj)

    monkeypatch.setattr(Path, "is_file", _fake_is_file)
    with _make_client(tmp_path) as client:
        r_boot = client.get("/api/v1/public/map-bootstrap")
        assert r_boot.status_code == 200
        trees = r_boot.json()["trees"]
        assert len(trees) == 1
        assert all(view["overlay_url"] is None for view in trees[0]["views"])

        r_health = client.get("/api/v1/health")
        assert r_health.status_code == 200


def test_gold_seed_demo_non_destructive_and_switchable(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer test-secret-token"}
    with _make_client(tmp_path) as client:
        real_batch = {
            "batch_id": "databricks-real-batch",
            "records": [
                {
                    "geosampa_id": "612162",
                    "distrito": "PINHEIROS",
                    "composite_risk_score": 12.5,
                    "priority_tier": "BAIXA",
                    "recommended_action": "Monitoramento de rotina",
                }
            ],
        }
        r_real = client.post("/api/v1/gold/risk-scores", json=real_batch, headers=headers)
        assert r_real.status_code == 200

        r_seed = client.post("/api/v1/gold/seed-demo")
        assert r_seed.status_code == 200
        assert r_seed.json()["skipped_real"] == 1

        r_scores = client.get("/api/v1/gold/risk-scores", headers=headers)
        assert r_scores.status_code == 200
        assert r_scores.json()["items"][0]["composite_risk_score"] == 12.5

    demo_twice_dir = tmp_path / "demo_twice"
    demo_twice_dir.mkdir()
    with _make_client(demo_twice_dir) as client_demo:
        r_first = client_demo.post("/api/v1/gold/seed-demo")
        assert r_first.status_code == 200
        assert r_first.json()["inserted"] == 1
        assert r_first.json()["updated"] == 0

        r_second = client_demo.post("/api/v1/gold/seed-demo")
        assert r_second.status_code == 200
        assert r_second.json()["inserted"] == 0
        assert r_second.json()["updated"] == 1

    demo_disabled_dir = tmp_path / "demo_disabled"
    demo_disabled_dir.mkdir()
    with _make_client(demo_disabled_dir, allow_demo_seed=False) as client_off:
        r_disabled = client_off.post("/api/v1/gold/seed-demo")
        assert r_disabled.status_code == 403
        features = client_off.get("/api/v1/public/map-bootstrap").json()["features"]
        assert features == {"demo_seed_enabled": False}


def test_real_gold_batch_purges_demo_records(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer test-secret-token"}
    with _make_client(tmp_path) as client:
        assert client.get("/api/v1/public/map-bootstrap").json()["features"] == {"demo_seed_enabled": True}
        assert client.post("/api/v1/gold/seed-demo").status_code == 200

        real_batch = {
            "batch_id": "databricks-real-batch",
            "records": [
                {
                    "geosampa_id": "612162",
                    "distrito": "PINHEIROS",
                    "composite_risk_score": 40.0,
                    "priority_tier": "MEDIA",
                    "recommended_action": "Vistoria programada",
                }
            ],
        }
        r_real = client.post("/api/v1/gold/risk-scores", json=real_batch, headers=headers)
        assert r_real.status_code == 200
        assert r_real.json()["removed_demo"] == 1
        assert r_real.json()["total_gold_records"] == 1

        items = client.get("/api/v1/gold/risk-scores", headers=headers).json()["items"]
        assert [i["batch_id"] for i in items] == ["databricks-real-batch"]
        assert items[0]["is_demo"] is False


def test_corrupt_gold_state_quarantined_on_startup(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold_state.json"
    gold_path.write_text("{corrupt-json", encoding="utf-8")
    with _make_client(tmp_path) as client:
        r_stats = client.get("/api/v1/stats")
        assert r_stats.status_code == 200
        assert r_stats.json()["gold_layer"]["total_records"] == 0
        corrupt_files = list(tmp_path.glob("gold_state.json.corrupt-*"))
        assert len(corrupt_files) == 1


def test_privacy_masking_unit_and_assistant_api(tmp_path: Path) -> None:
    masked_cpf, cats_cpf = mask_personal_data("Meu CPF e 123.456.789-00.")
    assert masked_cpf == "Meu CPF e [CPF omitido]."
    assert cats_cpf == ["cpf"]

    masked_cnpj, cats_cnpj = mask_personal_data("Empresa CNPJ 12.345.678/0001-90")
    assert masked_cnpj == "Empresa CNPJ [CNPJ omitido]"
    assert cats_cnpj == ["cnpj"]

    masked_email, cats_email = mask_personal_data("Contato: cidadao@exemplo.sp.gov.br")
    assert masked_email == "Contato: [e-mail omitido]"
    assert cats_email == ["email"]

    masked_placa, cats_placa = mask_personal_data("Carro placa ABC1D23 e antiga XYZ-1234")
    assert masked_placa == "Carro placa [placa omitida] e antiga [placa omitida]"
    assert cats_placa == ["placa"]

    masked_tel, cats_tel = mask_personal_data(
        "Ligar (11) 91234-5678, 11 91234 5678, 11912345678 ou (11) 3456-7890"
    )
    assert masked_tel == (
        "Ligar [telefone omitido], [telefone omitido], [telefone omitido] ou [telefone omitido]"
    )
    assert cats_tel == ["telefone"]

    unchanged_service, cats_service = mask_personal_data("Ligue 0800 72 72 196 ou 193")
    assert unchanged_service == "Ligue 0800 72 72 196 ou 193"
    assert cats_service == []

    unchanged_tree, cats_tree = mask_personal_data("arvore #612162")
    assert unchanged_tree == "arvore #612162"
    assert cats_tree == []

    with _make_client(tmp_path) as client:
        r_ask = client.post(
            "/api/v1/assistant/ask",
            json={
                "question": (
                    "Sou morador, CPF 123.456.789-00, placa ABC1D23, "
                    "telefone (11) 91234-5678. Como pedir poda da arvore na calcada?"
                )
            },
        )
        assert r_ask.status_code == 200
        body = r_ask.json()
        dumped = json.dumps(body)
        assert "123.456.789-00" not in dumped
        assert "ABC1D23" not in dumped
        assert "91234-5678" not in dumped
        assert body["privacy"]["masked_categories"] == ["cpf", "placa", "telefone"]


def test_llm_never_fabricates_citations(tmp_path: Path, monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._data = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    with _make_client(tmp_path) as client:
        service = client.app.state.assistant_service
        two_chunks = [dict(c) for c in service.chunks[:2]]

        fake_ungrounded = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "status": "respondido",
                                "nivel_risco": "preventivo_rotina",
                                "orgao_responsavel": "Prefeitura de Sao Paulo",
                                "telefone_ou_canal": "Portal SP156 (156)",
                                "pode_podar_por_conta": False,
                                "acao_imediata": "Solicite vistoria pelo Portal SP156.",
                                "fontes": ["INVENTADO-99"],
                            }
                        )
                    }
                }
            ]
        }
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=25.0: _FakeResponse(fake_ungrounded),
        )
        assert service._call_local_qwen("Como pedir poda?", two_chunks, None) is None

        real_id = two_chunks[0]["chunk_id"]
        fake_grounded = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "status": "respondido",
                                "nivel_risco": "preventivo_rotina",
                                "orgao_responsavel": "Prefeitura de Sao Paulo",
                                "telefone_ou_canal": "Portal SP156 (156)",
                                "pode_podar_por_conta": False,
                                "acao_imediata": "Solicite vistoria pelo Portal SP156.",
                                "fontes": [real_id],
                            }
                        )
                    }
                }
            ]
        }
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=25.0: _FakeResponse(fake_grounded),
        )
        ok_res = service._call_local_qwen("Como pedir poda?", two_chunks, None)
        assert ok_res is not None
        assert real_id in ok_res["fontes"]


def test_metrics_assistant_and_low_cardinality_route_labels(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        r_ask = client.post(
            "/api/v1/assistant/ask",
            json={"question": "Posso podar arvore na calcada?"},
        )
        assert r_ask.status_code == 200
        client.get("/static/nao-existe-xyz.js")

        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert "arborgrid_assistant_responses_total" in r_metrics.text
        assert "nao-existe-xyz" not in r_metrics.text


def test_public_map_bootstrap_omits_audit_actor_hash(tmp_path: Path) -> None:
    headers = {"Authorization": "Bearer test-secret-token"}
    with _make_client(tmp_path) as client:
        batch = {
            "batch_id": "batch-lgpd-check",
            "records": [
                {
                    "geosampa_id": "612162",
                    "distrito": "PINHEIROS",
                    "composite_risk_score": 82.0,
                    "priority_tier": "ALTA",
                    "recommended_action": "Vistoria preventiva",
                    "audit_actor_hash": "sha256-operador-privado",
                }
            ],
        }
        assert client.post("/api/v1/gold/risk-scores", json=batch, headers=headers).status_code == 200

        r_pub = client.get("/api/v1/public/map-bootstrap")
        assert r_pub.status_code == 200
        gold_public = r_pub.json()["gold_records"]
        assert len(gold_public) == 1
        assert "audit_actor_hash" not in gold_public[0]

        r_priv = client.get("/api/v1/gold/risk-scores", headers=headers)
        assert r_priv.status_code == 200
        assert r_priv.json()["items"][0]["audit_actor_hash"] == "sha256-operador-privado"


def test_negated_emergency_terms_do_not_force_emergency(tmp_path: Path, monkeypatch) -> None:
    """Negacoes como 'nao tem faisca' nao contam como emergencia no modo de regras, e a resposta do LLM nunca e reescrita."""
    from app.services.assistant_service import _EMERGENCY_TERMS, _NEGATED_EMERGENCY_RE, _normalize_text

    def sinal(pergunta: str) -> list[str]:
        limpa = _NEGATED_EMERGENCY_RE.sub(" ", _normalize_text(pergunta))
        return [t for t in _EMERGENCY_TERMS if t in limpa]

    preventivas = [
        "A árvore em frente de casa está encostando nos fios, mas não tem faísca. Como peço a poda?",
        "A árvore cresceu perto dos fios, mas não há faíscas nem curto-circuito. Como pedir poda preventiva?",
        "Galho encostado na fiação, sem faísca e sem nenhum fio partido.",
        "Não tem ninguém ferido, só um galho grande perto do fio.",
    ]
    emergencias = [
        "Tem uma árvore soltando faíscas nos fios agora!",
        "Não tem faísca, mas começou a sair fogo do poste.",
        "Não sei se é faísca, mas tem um barulho estranho no fio.",
        "A árvore caiu em cima do fio na tempestade.",
    ]
    for pergunta in preventivas:
        assert sinal(pergunta) == [], pergunta
    for pergunta in emergencias:
        assert sinal(pergunta), pergunta

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._data = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    with _make_client(tmp_path) as client:
        service = client.app.state.assistant_service
        chunks = [dict(c) for c in service.chunks[:2]]
        preventiva = {
            "status": "respondido",
            "nivel_risco": "preventivo_rotina",
            "orgao_responsavel": "Prefeitura de Sao Paulo",
            "telefone_ou_canal": "Portal SP156 (156)",
            "pode_podar_por_conta": False,
            "acao_imediata": "Solicite a poda preventiva pelo Portal SP156 (156).",
            "fontes": [chunks[0]["chunk_id"]],
        }
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=25.0: _FakeResponse({"choices": [{"message": {"content": json.dumps(preventiva)}}]}),
        )
        res = service._call_local_qwen(preventivas[0], chunks, None)
        assert res is not None
        assert res["nivel_risco"] == "preventivo_rotina"
        assert "156" in res["acao_imediata"]


def test_assistant_knowledge_status_and_synced_kb_compatibility(tmp_path: Path, monkeypatch) -> None:
    """Endpoint de status da base RPA responde sem falhar quando ausente e le relatorio quando presente."""
    missing_report = tmp_path / "relatorio_inexistente.json"
    monkeypatch.setenv("ARBORGRID_KNOWLEDGE_REPORT_PATH", str(missing_report))

    with _make_client(tmp_path) as client:
        r_missing = client.get("/api/v1/assistant/knowledge-status")
        assert r_missing.status_code == 200
        assert r_missing.json() == {"sincronizado": False}

        present_report = tmp_path / "relatorio_execucao.json"
        present_report.write_text(
            json.dumps(
                {
                    "execucao_id": "abc12345",
                    "inicio": "2026-11-19T21:30:00-03:00",
                    "fim": "2026-11-19T21:30:02-03:00",
                    "modo": "offline",
                    "codigo_saida": 0,
                    "total_fontes": 11,
                    "total_trechos": 16,
                    "contagem_status_fontes": {"CONFIRMADA": 9, "FONTE_FECHADA": 1, "FONTE_INTERNA": 1},
                    "contagem_status_trechos": {"CONFIRMADA": 10, "FONTE_FECHADA": 3, "FONTE_INTERNA": 3},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("ARBORGRID_KNOWLEDGE_REPORT_PATH", str(present_report))

        r_ok = client.get("/api/v1/assistant/knowledge-status")
        assert r_ok.status_code == 200
        body = r_ok.json()
        assert body["sincronizado"] is True
        assert body["ultima_sincronizacao"] == "2026-11-19T21:30:02-03:00"
        assert body["contagem_status_fontes"]["CONFIRMADA"] == 9
        assert body["contagem_status_trechos"]["CONFIRMADA"] == 10

        # Valida que o AssistantService le normalmente uma base JSONL com o campo extra verificacao
        kb_orig = WEBAPP_ROOT / "data" / "knowledge_base.jsonl"
        synced_kb = tmp_path / "knowledge_base_synced.jsonl"
        linhas_sync = []
        for linha in kb_orig.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                obj = json.loads(linha)
                obj["verificacao"] = {
                    "status": "CONFIRMADA",
                    "fonte_id": "teste",
                    "url": "https://sp156.prefeitura.sp.gov.br/portal/servicos",
                    "sha256": "abc123",
                    "verificado_em": "2026-11-19T21:30:02-03:00",
                    "fatos_ausentes": [],
                }
                linhas_sync.append(json.dumps(obj, ensure_ascii=False))
        synced_kb.write_text("\n".join(linhas_sync) + "\n", encoding="utf-8")

        from app.services.assistant_service import AssistantService

        synced_service = AssistantService(
            kb_path=synced_kb,
            manifest_store=client.app.state.manifest_store,
            gold_store=client.app.state.gold_store,
            llm_api_url="http://127.0.0.1:9/v1/chat/completions",
        )
        assert len(synced_service.chunks) == 16
        resp_ask = synced_service.ask("Posso podar arvore na calcada?")
        assert resp_ask["structured"]["status"] == "respondido"
        assert resp_ask["citations"]
