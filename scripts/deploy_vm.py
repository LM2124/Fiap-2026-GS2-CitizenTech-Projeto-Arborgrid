"""Utilitario de deploy direto e tunel SSH para a VM ArborGrid SP.

Permite enviar o codigo do repositorio local diretamente para a VM via SSH
(sem exigir `rsync` no Windows nem repositorio publico no GitHub), gerar
`infra/.env` na primeira execucao, reconstruir os containers Docker e abrir o
tunel seguro do Grafana/Prometheus sem conflitar com portas locais do Windows.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TERRAFORM_DIR = ROOT_DIR / "infra" / "terraform"
DEFAULT_KEY_PATH = Path.home() / ".ssh" / "arborgrid_gcp"
SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _resolve_host(explicit_host: str | None) -> str:
    if explicit_host:
        host = explicit_host.strip()
        if not SAFE_HOST_RE.match(host):
            raise SystemExit(f"[erro] Host invalido: {host}")
        return host

    tf_bin = shutil.which("terraform")
    if tf_bin and (TERRAFORM_DIR / "terraform.tfstate").is_file():
        res = subprocess.run(
            [tf_bin, f"-chdir={TERRAFORM_DIR}", "output", "-raw", "public_ip"],
            capture_output=True,
            text=True,
            check=False,
        )
        ip = res.stdout.strip()
        if res.returncode == 0 and ip and SAFE_HOST_RE.match(ip):
            print(f"[terraform] IP publico detectado: {ip}")
            return ip

    raise SystemExit(
        "[erro] Informe --host <PUBLIC_IP> ou garanta infra/terraform/terraform.tfstate gerado."
    )


def _resolve_ssh_key(explicit_key: str | None) -> Path:
    key_path = Path(explicit_key).expanduser() if explicit_key else DEFAULT_KEY_PATH
    if not key_path.is_file():
        raise SystemExit(f"[erro] Chave SSH nao encontrada em: {key_path}")
    return key_path


def _build_archive_bytes(use_worktree: bool) -> bytes:
    if not use_worktree:
        dirty = subprocess.run(
            ["git", "-C", str(ROOT_DIR), "status", "--short", "-uno"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if dirty:
            print(
                "[aviso] Voce tem alteracoes locais nao commitadas. "
                "O modo padrao envia o ultimo commit (HEAD); use `--worktree` para enviar tambem arquivos modificados."
            )
        res = subprocess.run(
            ["git", "-C", str(ROOT_DIR), "archive", "--format=tar.gz", "HEAD"],
            capture_output=True,
            check=True,
        )
        return res.stdout

    ls_res = subprocess.run(
        ["git", "-C", str(ROOT_DIR), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    rel_paths = [p.decode("utf-8") for p in ls_res.stdout.split(b"\x00") if p]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in rel_paths:
            full = ROOT_DIR / rel
            if full.is_file():
                tf.add(full, arcname=rel)
    return buf.getvalue()


def _ssh_base_args(key_path: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]


def _open_tunnel(host: str, user: str, key_path: Path, grafana_port: int, prom_port: int) -> None:
    print(f"[tunel] Abrindo tunel SSH para {user}@{host}...")
    print(f"  - Grafana   -> http://localhost:{grafana_port}")
    print(f"  - Prometheus-> http://localhost:{prom_port}")
    print("  Pressione Ctrl+C para encerrar o tunel.")
    cmd = _ssh_base_args(key_path) + [
        "-N",
        "-L",
        f"{grafana_port}:127.0.0.1:3000",
        "-L",
        f"{prom_port}:127.0.0.1:9090",
        f"{user}@{host}",
    ]
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n[tunel] Encerrado.")


def _deploy(
    host: str,
    user: str,
    key_path: Path,
    use_worktree: bool,
    expose_media: bool,
    reset_env: bool,
) -> None:
    if not SAFE_HOST_RE.match(user):
        raise SystemExit(f"[erro] Usuario SSH invalido: {user}")

    archive_bytes = _build_archive_bytes(use_worktree=use_worktree)
    size_mib = len(archive_bytes) / (1024 * 1024)
    print(f"[sync] Pacote preparado ({size_mib:.1f} MiB). Enviando para {user}@{host}:/opt/arborgrid/app ...")

    subprocess.run(
        _ssh_base_args(key_path)
        + [
            f"{user}@{host}",
            "mkdir -p /opt/arborgrid/app /opt/arborgrid/media /opt/arborgrid/llm_models && tar -xzf - -C /opt/arborgrid/app",
        ],
        input=archive_bytes,
        check=True,
    )

    api_token = f"arborgrid-{secrets.token_urlsafe(24)}"
    grafana_pass = secrets.token_urlsafe(14)
    expose_str = "true" if expose_media else "false"
    reset_flag = "1" if reset_env else "0"

    remote_script = f"""set -euo pipefail
ENV_PATH="/opt/arborgrid/app/infra/.env"
if [ ! -f "$ENV_PATH" ] || [ "{reset_flag}" = "1" ]; then
  cat > "$ENV_PATH" <<'ENVEOF'
ARBORGRID_API_TOKEN={api_token}
ARBORGRID_EXPOSE_MEDIA_URLS={expose_str}
ARBORGRID_MEDIA_HOST_DIR=/opt/arborgrid/media
ARBORGRID_LLM_MODELS_DIR=/opt/arborgrid/llm_models
COMPOSE_PROFILES=llm
LLAMA_THREADS=2
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD={grafana_pass}
ENVEOF
  chmod 600 "$ENV_PATH"
  echo "[env] Novo .env criado em $ENV_PATH"
  echo "  ARBORGRID_API_TOKEN={api_token}"
  echo "  GRAFANA_ADMIN_USER=admin"
  echo "  GRAFANA_ADMIN_PASSWORD={grafana_pass}"
else
  echo "[env] Mantendo $ENV_PATH existente (use --reset-env para regerar)."
fi

cd /opt/arborgrid/app/infra
if [ ! -f /opt/arborgrid/llm_models/arborgrid-qwen3-1.7b-q4_k_m.gguf ]; then
  echo "[aviso] Modelo GGUF nao encontrado em /opt/arborgrid/llm_models/; subindo sem perfil llm (fallback RAG)."
  sed -i 's/^COMPOSE_PROFILES=.*/COMPOSE_PROFILES=/' "$ENV_PATH"
fi
# O container roda como UID 10001: midias enviadas do Windows chegam sem leitura/travessia para "outros"
sudo chmod -R a+rX /opt/arborgrid/media || true
docker compose --env-file "$ENV_PATH" up -d --build --remove-orphans
docker compose --env-file "$ENV_PATH" restart nginx prometheus grafana  # bind mounts de arquivo presos ao inode antigo
docker compose --env-file "$ENV_PATH" ps
"""

    print("[deploy] Subindo containers com Docker Compose na VM...")
    subprocess.run(
        _ssh_base_args(key_path) + [f"{user}@{host}", "bash -s"],
        input=remote_script.encode("utf-8"),
        check=True,
    )
    print(f"\n[ok] MVP publicado em: http://{host}")
    print(
        f"[dica] Para abrir Grafana e Prometheus sem conflito de porta no Windows, rode:\n"
        f"  python scripts/deploy_vm.py --host {host} --tunnel"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Deploy direto e tunel SSH da VM ArborGrid SP.")
    parser.add_argument("--host", help="IP publico da VM (se omitido, le de infra/terraform/terraform.tfstate)")
    parser.add_argument("--user", default="ubuntu", help="Usuario SSH da VM (padrao: ubuntu)")
    parser.add_argument("--key", help="Caminho da chave privada SSH (padrao: ~/.ssh/arborgrid_gcp)")
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="Envia arquivos rastreados do working tree atual (inclui edicoes locais antes do commit)",
    )
    parser.add_argument(
        "--expose-media",
        action="store_true",
        help="Define ARBORGRID_EXPOSE_MEDIA_URLS=true ao criar um novo .env",
    )
    parser.add_argument(
        "--reset-env",
        action="store_true",
        help="Recria /opt/arborgrid/app/infra/.env na VM com novos tokens",
    )
    parser.add_argument("--tunnel", action="store_true", help="Abre tunel SSH para Grafana e Prometheus")
    parser.add_argument("--grafana-port", type=int, default=13000, help="Porta local para Grafana (padrao: 13000)")
    parser.add_argument("--prom-port", type=int, default=19090, help="Porta local para Prometheus (padrao: 19090)")
    args = parser.parse_args()

    host = _resolve_host(args.host)
    key_path = _resolve_ssh_key(args.key)

    if args.tunnel:
        _open_tunnel(host, args.user, key_path, args.grafana_port, args.prom_port)
        return

    _deploy(
        host=host,
        user=args.user,
        key_path=key_path,
        use_worktree=args.worktree,
        expose_media=args.expose_media,
        reset_env=args.reset_env,
    )


if __name__ == "__main__":
    main()
