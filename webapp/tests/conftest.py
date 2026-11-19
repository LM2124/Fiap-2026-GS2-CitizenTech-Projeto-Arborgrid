from __future__ import annotations

import pytest

import app.main as main_module


@pytest.fixture(autouse=True)
def stub_frontend_build(tmp_path, monkeypatch):
    """Os testes da API nao dependem do build do frontend (gerado pelo Dockerfile e pelo CI).

    Aponta o diretorio estatico para uma pasta temporaria com um index.html minimo, para que
    as rotas do SPA respondam 200 mesmo num clone limpo, sem `npm run build`.
    """
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        "<!doctype html><title>ArborGrid SP</title><div id=\"root\"></div>", encoding="utf-8"
    )
    monkeypatch.setattr(main_module, "STATIC_DIR", static_dir)
