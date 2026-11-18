# Deploy automatico do GitHub Actions sem abrir a porta 22 para a internet:
# o runner se autentica por Workload Identity Federation (sem chave JSON de longa
# duracao) e chega na VM pelo tunel TCP do Identity-Aware Proxy (IAP).
#
# Pre-requisito (bootstrap manual, uma vez por projeto): as APIs
# cloudresourcemanager.googleapis.com e serviceusage.googleapis.com precisam estar
# ativas ANTES do apply. O provider le o projeto pela Cloud Resource Manager ao
# gerenciar google_project_service e google_project_iam_member, entao nao da para
# ativa-la pelo proprio Terraform (problema do ovo e da galinha).

locals {
  cicd_apis = toset([
    "iap.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ])
}

resource "google_project_service" "cicd" {
  for_each           = local.cicd_apis
  service            = each.value
  disable_on_destroy = false
}

# SSH apenas a partir da faixa fixa do IAP (o acesso administrativo direto segue restrito a allowed_ssh_cidr)
resource "google_compute_firewall" "arborgrid_ssh_iap" {
  name        = "${var.project_name}-allow-ssh-iap"
  network     = "default"
  description = "SSH (22) somente via tunel TCP do Identity-Aware Proxy (35.235.240.0/20) para o deploy do CI"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["${var.project_name}-ssh"]
}

# Identidade do pipeline: so consegue ler instancias e abrir tunel IAP para ESTA VM
resource "google_service_account" "cicd" {
  account_id   = "${var.project_name}-cicd"
  display_name = "ArborGrid CI/CD (GitHub Actions via IAP)"
}

resource "google_project_iam_member" "cicd_compute_viewer" {
  project = var.gcp_project_id
  role    = "roles/compute.viewer"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

resource "google_iap_tunnel_instance_iam_member" "cicd_tunnel" {
  project  = var.gcp_project_id
  zone     = var.gcp_zone
  instance = google_compute_instance.arborgrid_vm.name
  role     = "roles/iap.tunnelResourceAccessor"
  member   = "serviceAccount:${google_service_account.cicd.email}"

  depends_on = [google_project_service.cicd]
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "${var.project_name}-github"
  display_name              = "GitHub Actions"
  description               = "Tokens OIDC do GitHub Actions para o deploy do ArborGrid"

  depends_on = [google_project_service.cicd]
}

# Aceita apenas tokens do repositorio configurado, na branch main
resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  attribute_condition = "assertion.repository == \"${var.github_repository}\" && assertion.ref == \"refs/heads/main\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "cicd_wif" {
  service_account_id = google_service_account.cicd.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}
