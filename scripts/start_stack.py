"""Inicializador multiplataforma (Windows / macOS / Linux) do ArborGrid SP.

Prepara `infra/.env` automaticamente em clones novos, detecta se o modelo GGUF
opcional do Qwen3 esta presente para ativar o perfil `llm` e sobe toda a stack
Docker (ou a API FastAPI local sem Docker).
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
INFRA_DIR = ROOT_DIR / "infra"
WEBAPP_DIR = ROOT_DIR / "webapp"
ENV_FILE = INFRA_DIR / ".env"
ENV_EXAMPLE = INFRA_DIR / ".env.example"
GGUF_MODEL_PATH = ROOT_DIR / "llm-assistant" / "models" / "arborgrid-qwen3-1.7b-q4_k_m.gguf"


def _ensure_infra_env() -> dict[str, str]:
    """Garante infra/.env valido sem depender de edicao manual em clone novo."""
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            ENV_FILE.write_text("", encoding="utf-8")
        print("[setup] Criado infra/.env a partir de infra/.env.example.")

    raw_lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    values: dict[str, str] = {}
    changed = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue
        key, val = line.split("=", 1)
        key_clean = key.strip()
        val_clean = val.strip()

        if key_clean == "ARBORGRID_API_TOKEN" and (
            not val_clean or val_clean.startswith("substitua_")
        ):
            val_clean = f"arborgrid-dev-{secrets.token_urlsafe(18)}"
            line = f"{key_clean}={val_clean}"
            changed = True
        elif key_clean == "GRAFANA_ADMIN_PASSWORD" and (
            not val_clean or val_clean.startswith("substitua_")
        ):
            val_clean = secrets.token_urlsafe(12)
            line = f"{key_clean}={val_clean}"
            changed = True

        values[key_clean] = val_clean
        updated_lines.append(line)

    if "ARBORGRID_API_TOKEN" not in values:
        tok = f"arborgrid-dev-{secrets.token_urlsafe(18)}"
        values["ARBORGRID_API_TOKEN"] = tok
        updated_lines.append(f"ARBORGRID_API_TOKEN={tok}")
        changed = True
    if "GRAFANA_ADMIN_PASSWORD" not in values:
        pwd = secrets.token_urlsafe(12)
        values["GRAFANA_ADMIN_PASSWORD"] = pwd
        updated_lines.append(f"GRAFANA_ADMIN_PASSWORD={pwd}")
        changed = True

    if changed:
        ENV_FILE.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        print("[setup] Segredos locais gerados automaticamente em infra/.env (nao versionado).")

    return values


def _check_docker_ready() -> None:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        print(
            "[erro] Docker CLI nao encontrado no PATH. Instale o Docker Desktop ou use `--local-api`.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    probe = subprocess.run(
        [docker_bin, "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        print(
            "[erro] O daemon Docker nao esta respondendo. Abra o Docker Desktop e tente novamente (ou rode com `--local-api`).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _wait_health_docker(timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1/api/v1/health", timeout=2.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1.5)
    return False


def _run_docker_up(skip_build: bool = False) -> None:
    _check_docker_ready()
    env_values = _ensure_infra_env()

    has_gguf = GGUF_MODEL_PATH.is_file()
    cmd = ["docker", "compose", "--env-file", ".env"]
    if has_gguf:
        cmd.extend(["--profile", "llm"])
        print(f"[llm] Modelo Qwen3 GGUF detectado em {GGUF_MODEL_PATH.name}; ativando container llama-server.")
    else:
        print(
            "[llm] Arquivo GGUF opcional ausente em llm-assistant/models/; "
            "subindo stack padrao (assistente em modo normativo deterministico)."
        )

    cmd.extend(["up", "-d"])
    if not skip_build:
        cmd.append("--build")

    print(f"[docker] Executando: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=INFRA_DIR, check=True)

    print("[health] Aguardando aplicacao ficar pronta em http://localhost ...")
    ok = _wait_health_docker()
    if not ok:
        print(
            "[aviso] Health check nao respondeu a tempo; verifique logs com `docker compose logs -f` em infra/.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    token = env_values.get("ARBORGRID_API_TOKEN", "")
    print("\n" + "=" * 68)
    print("ArborGrid SP pronto!")
    print("=" * 68)
    print("  Web App:            http://localhost")
    print("  Swagger OpenAPI:    http://localhost/docs")
    print("  Prometheus:         http://127.0.0.1:9090")
    print("  Grafana:            http://127.0.0.1:3000")
    print("  API Big Data:")
    print("    GET  http://localhost/api/v1/inspections")
    print("    GET  http://localhost/api/v1/inspections/available-dates")
    print("    POST http://localhost/api/v1/gold/risk-scores")
    print(f"  Authorization:      Bearer {token}")
    print("=" * 68)


def _run_docker_down() -> None:
    _check_docker_ready()
    cmd = ["docker", "compose", "--profile", "llm", "down"]
    print(f"[docker] Executando: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=INFRA_DIR, check=True)


def _resolve_webapp_python() -> str:
    venv_dir = WEBAPP_DIR / ".venv"
    py_win = venv_dir / "Scripts" / "python.exe"
    py_nix = venv_dir / "bin" / "python"
    if py_win.exists():
        return str(py_win)
    if py_nix.exists():
        return str(py_nix)

    print("[setup] Criando ambiente virtual Python em webapp/.venv ...")
    uv_bin = shutil.which("uv")
    if uv_bin:
        subprocess.run([uv_bin, "venv", ".venv"], cwd=WEBAPP_DIR, check=True)
        py_exec = str(py_win if os.name == "nt" else py_nix)
        subprocess.run(
            [uv_bin, "pip", "install", "--python", py_exec, "-r", "requirements-dev.txt"],
            cwd=WEBAPP_DIR,
            check=True,
        )
        return py_exec

    subprocess.run([sys.executable, "-m", "venv", ".venv"], cwd=WEBAPP_DIR, check=True)
    py_exec = str(py_win if os.name == "nt" else py_nix)
    subprocess.run([py_exec, "-m", "pip", "install", "-r", "requirements-dev.txt"], cwd=WEBAPP_DIR, check=True)
    return py_exec


def _run_local_api() -> None:
    py_exec = _resolve_webapp_python()
    env = dict(os.environ)
    env.setdefault("ARBORGRID_REQUIRE_AUTH", "false")
    print("\n[local] Iniciando Web App + API FastAPI em http://127.0.0.1:8000 (auth local desabilitada).")
    print("[local] Swagger OpenAPI: http://127.0.0.1:8000/docs\n")
    subprocess.run(
        [py_exec, "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
        cwd=WEBAPP_DIR,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicializador ArborGrid SP")
    parser.add_argument("--local-api", action="store_true", help="Roda apenas o FastAPI local sem Docker (porta 8000)")
    parser.add_argument("--down", action="store_true", help="Encerra os containers Docker Compose")
    parser.add_argument("--no-build", action="store_true", help="Sobe Docker Compose sem rebuild da imagem")
    args = parser.parse_args()

    if args.down:
        _run_docker_down()
    elif args.local_api:
        _run_local_api()
    else:
        _run_docker_up(skip_build=args.no_build)


if __name__ == "__main__":
    main()
