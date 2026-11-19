"""Bootstrap e acesso administrativo da VM ArborGrid SP (IAP + OS Login).

O deploy continuo e do GitHub Actions (imagem publicada no Artifact Registry). Este
script cobre o que o pipeline nao faz, sempre pelo tunel do Identity-Aware Proxy, com
a sua identidade do Google (OS Login), sem chave SSH fixa nem porta 22 publica:

  bootstrap  cria /opt/arborgrid/*, gera o infra/.env na VM (so na primeira vez ou
             com --reset-env) e envia, se pedido, as midias e o modelo GGUF do LLM.
  tunnel     abre Grafana (3000) e Prometheus (9090) no seu computador.
  shell      abre um terminal na VM.

Requer o gcloud autenticado (gcloud auth login) e o papel de administrador da VM
concedido pelo Terraform (variavel admin_principals).
"""

from __future__ import annotations

import argparse
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TFVARS_PATH = ROOT_DIR / "infra" / "terraform" / "terraform.tfvars"
SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
LLM_MODEL_NAME = "arborgrid-qwen3-1.7b-q4_k_m.gguf"


def _tfvar(name: str) -> str | None:
    """Le um valor simples do terraform.tfvars local (fora do Git), se existir."""
    if not TFVARS_PATH.is_file():
        return None
    match = re.search(rf'^\s*{name}\s*=\s*"([^"]*)"', TFVARS_PATH.read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else None


def _safe(value: str, label: str) -> str:
    if not SAFE_VALUE_RE.match(value):
        sys.exit(f"[erro] Valor invalido para {label}: {value!r}")
    return value


class Vm:
    def __init__(self, project: str, zone: str, name: str) -> None:
        gcloud = shutil.which("gcloud")
        if not gcloud:
            sys.exit("[erro] gcloud nao encontrado no PATH. Instale o Google Cloud SDK e rode gcloud auth login.")
        self.gcloud = gcloud
        self.project = _safe(project, "--project")
        self.zone = _safe(zone, "--zone")
        self.name = _safe(name, "--vm")

    def _base(self, command: str) -> list[str]:
        return [self.gcloud, "compute", command, "--project", self.project, "--zone", self.zone, "--tunnel-through-iap"]

    def ssh(self, remote_command: str | None = None, extra: list[str] | None = None) -> None:
        args = self._base("ssh") + [self.name]
        if remote_command:
            args += ["--command", remote_command]
        if extra:
            args += ["--", *extra]
        subprocess.run(args, check=True)

    def scp(self, local: Path, remote: str) -> None:
        subprocess.run(self._base("scp") + [str(local), f"{self.name}:{remote}"], check=True)

    def run_script(self, script: str) -> None:
        """Envia o script como arquivo e executa (evita depender de stdin pelo PuTTY no Windows)."""
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8", newline="\n") as tmp:
            tmp.write(script)
            local = Path(tmp.name)
        try:
            self.scp(local, "arborgrid_bootstrap.sh")
            self.ssh("bash ~/arborgrid_bootstrap.sh; status=$?; rm -f ~/arborgrid_bootstrap.sh; exit $status")
        finally:
            local.unlink(missing_ok=True)

    def upload_tree(self, local_dir: Path, remote_dir: str) -> None:
        """Compacta a pasta localmente, envia e extrai com sudo no destino."""
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            archive = Path(tmp.name)
        try:
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(local_dir, arcname=".")
            print(f"[upload] {local_dir} -> {remote_dir} ({archive.stat().st_size / 1e6:.1f} MB)")
            self.scp(archive, "arborgrid_upload.tgz")
            self.ssh(
                f"sudo install -d -m 755 {remote_dir} && sudo tar xzf ~/arborgrid_upload.tgz -C {remote_dir} "
                f"--no-same-owner && rm -f ~/arborgrid_upload.tgz && sudo chmod -R a+rX {remote_dir}"
            )
        finally:
            archive.unlink(missing_ok=True)


def _bootstrap(vm: Vm, domain: str, reset_env: bool, media: Path | None, llm_model: Path | None) -> None:
    api_token = f"arborgrid-{secrets.token_urlsafe(24)}"
    grafana_pass = secrets.token_urlsafe(14)
    reset_flag = "1" if reset_env else "0"
    script = f"""set -euo pipefail
sudo install -d -m 755 /opt/arborgrid/app/infra /opt/arborgrid/media /opt/arborgrid/llm_models /opt/arborgrid/vision_models
ENV_PATH=/opt/arborgrid/app/infra/.env
if ! sudo test -f "$ENV_PATH" || [ "{reset_flag}" = "1" ]; then
  sudo tee "$ENV_PATH" > /dev/null <<'ENVEOF'
ARBORGRID_API_TOKEN={api_token}
ARBORGRID_EXPOSE_MEDIA_URLS=false
ARBORGRID_ALLOW_DEMO_SEED=false
ARBORGRID_DOMAIN={domain}
ARBORGRID_LLM_TIMEOUT_SECONDS=110
ARBORGRID_MEDIA_HOST_DIR=/opt/arborgrid/media
ARBORGRID_LLM_MODELS_DIR=/opt/arborgrid/llm_models
ARBORGRID_VISION_MODELS_DIR=/opt/arborgrid/vision_models
COMPOSE_PROFILES=llm
LLAMA_THREADS=4
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD={grafana_pass}
ENVEOF
  sudo chmod 600 "$ENV_PATH"
  echo "[env] Novo .env criado. Guarde estes valores (aparecem so agora):"
  echo "  ARBORGRID_API_TOKEN={api_token}"
  echo "  GRAFANA_ADMIN_PASSWORD={grafana_pass}"
else
  echo "[env] Mantendo o .env existente (use --reset-env para regerar)."
fi
# O container roda como UID 10001 e monta as midias somente leitura
sudo chmod -R a+rX /opt/arborgrid/media /opt/arborgrid/llm_models /opt/arborgrid/vision_models
"""
    print(f"[bootstrap] Preparando {vm.name} pelo tunel do IAP...")
    vm.run_script(script)

    if media:
        for sub in ("images", "v3_overlays"):
            source = media / sub
            if not source.is_dir():
                sys.exit(f"[erro] Pasta de midia nao encontrada: {source}")
            vm.upload_tree(source, f"/opt/arborgrid/media/{sub}")

    if llm_model:
        if not llm_model.is_file():
            sys.exit(f"[erro] Modelo GGUF nao encontrado: {llm_model}")
        print(f"[upload] {llm_model.name} ({llm_model.stat().st_size / 1e9:.2f} GB)")
        vm.scp(llm_model, LLM_MODEL_NAME)
        vm.ssh(f"sudo mv ~/{LLM_MODEL_NAME} /opt/arborgrid/llm_models/ && sudo chmod a+r /opt/arborgrid/llm_models/{LLM_MODEL_NAME}")

    print("\n[ok] VM pronta. O deploy da aplicacao e feito pelo GitHub Actions: faca push na main")
    print("     (ou rode o workflow de novo) para publicar a imagem e subir os containers.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=_tfvar("gcp_project_id"), help="Projeto GCP (padrao: terraform.tfvars)")
    parser.add_argument("--zone", default=_tfvar("gcp_zone") or "us-central1-a", help="Zona da VM")
    parser.add_argument(
        "--vm",
        default=f"{_tfvar('project_name') or 'arborgrid-mvp'}-vm",
        help="Nome da VM (padrao: <project_name>-vm do terraform.tfvars)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="Prepara pastas, .env, midias e modelo GGUF na VM")
    boot.add_argument("--domain", default=_tfvar("domain") or "arborgrid.duckdns.org")
    boot.add_argument("--reset-env", action="store_true", help="Regera o .env (troca token da API e senha do Grafana)")
    boot.add_argument("--media", type=Path, help="Pasta com images/ e v3_overlays/ (ex.: arborgrid-vision/dataset/sp_geosampa_api)")
    boot.add_argument("--llm-model", type=Path, help=f"Caminho local do {LLM_MODEL_NAME}")

    tun = sub.add_parser("tunnel", help="Abre Grafana e Prometheus no seu computador")
    tun.add_argument("--grafana-port", type=int, default=13000, help="Porta local do Grafana (padrao: 13000)")
    tun.add_argument("--prom-port", type=int, default=19090, help="Porta local do Prometheus (padrao: 19090)")

    sub.add_parser("shell", help="Abre um terminal na VM")

    args = parser.parse_args()
    if not args.project:
        sys.exit("[erro] Informe --project (ou defina gcp_project_id no infra/terraform/terraform.tfvars).")
    vm = Vm(args.project, args.zone, args.vm)

    if args.command == "bootstrap":
        _bootstrap(vm, _safe(args.domain, "--domain"), args.reset_env, args.media, args.llm_model)
    elif args.command == "tunnel":
        print(f"[tunel] Grafana:    http://127.0.0.1:{args.grafana_port}")
        print(f"[tunel] Prometheus: http://127.0.0.1:{args.prom_port}  (Ctrl+C encerra)")
        vm.ssh(extra=["-N", "-L", f"{args.grafana_port}:127.0.0.1:3000", "-L", f"{args.prom_port}:127.0.0.1:9090"])
    else:
        vm.ssh()


if __name__ == "__main__":
    main()
