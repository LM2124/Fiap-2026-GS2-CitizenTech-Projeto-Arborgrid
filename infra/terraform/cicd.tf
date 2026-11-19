# Deploy automatico do GitHub Actions sem nenhum segredo de longa duracao:
# o runner se autentica por Workload Identity Federation (token OIDC trocado por um
# token GCP de curta duracao), publica a imagem no Artifact Registry e entra na VM
# por OS Login, pelo tunel TCP do Identity-Aware Proxy (IAP). A porta 22 so aceita
# a faixa fixa do IAP; nao ha IP administrativo liberado.
#
# Pre-requisito (bootstrap manual, uma vez por projeto): as APIs
# cloudresourcemanager.googleapis.com e serviceusage.googleapis.com precisam estar
# ativas ANTES do apply. O provider le o projeto pela Cloud Resource Manager ao
# gerenciar google_project_service e google_project_iam_member, entao nao da para
# ativa-la pelo proprio Terraform (problema do ovo e da galinha).

locals {
  cicd_apis = toset([
    "artifactregistry.googleapis.com",
    "iap.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "oslogin.googleapis.com",
    "sts.googleapis.com",
  ])
}

resource "google_project_service" "cicd" {
  for_each           = local.cicd_apis
  service            = each.value
  disable_on_destroy = false
}

# Uma API recem-ativada pode negar acesso por 1 a 2 minutos ate propagar. Os recursos que
# dependem dela esperam aqui, para o apply passar de primeira num projeto novo. Ativar uma
# API nova na lista recria a espera.
resource "time_sleep" "apis_propagation" {
  create_duration = "90s"
  triggers = {
    apis = join(",", sort(tolist(local.cicd_apis)))
  }
  depends_on = [google_project_service.cicd]
}

# Unica entrada SSH: a faixa fixa do IAP (usada pelo CI e pelo acesso administrativo)
resource "google_compute_firewall" "arborgrid_ssh_iap" {
  name        = "${var.project_name}-allow-ssh-iap"
  network     = "default"
  description = "SSH (22) somente via tunel TCP do Identity-Aware Proxy (35.235.240.0/20): deploy do CI e acesso administrativo"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["${var.project_name}-ssh"]
}

# ---------------------------------------------------------------------------------------
# Identidade do pipeline: publica imagens, le a VM e entra nela por IAP + OS Login
# ---------------------------------------------------------------------------------------
resource "google_service_account" "cicd" {
  account_id   = "${var.project_name}-cicd"
  display_name = "ArborGrid CI/CD (GitHub Actions via IAP)"
}

# Ler a instancia e as chaves de host publicadas em guest attributes
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

  depends_on = [time_sleep.apis_propagation]
}

# OS Login com sudo apenas NESTA VM (docker e rsync para /opt exigem root)
resource "google_compute_instance_iam_member" "cicd_os_admin_login" {
  project       = var.gcp_project_id
  zone          = var.gcp_zone
  instance_name = google_compute_instance.arborgrid_vm.name
  role          = "roles/compute.osAdminLogin"
  member        = "serviceAccount:${google_service_account.cicd.email}"
}

# Exigido pelo OS Login em VMs com service account anexada
resource "google_service_account_iam_member" "cicd_act_as_vm_runtime" {
  service_account_id = google_service_account.vm_runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cicd.email}"
}

# Publicar imagens apenas neste repositorio do Artifact Registry
resource "google_artifact_registry_repository_iam_member" "cicd_writer" {
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.arborgrid.location
  repository = google_artifact_registry_repository.arborgrid.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cicd.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "${var.project_name}-github"
  display_name              = "GitHub Actions"
  description               = "Tokens OIDC do GitHub Actions para o deploy do ArborGrid"

  depends_on = [time_sleep.apis_propagation]
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

# ---------------------------------------------------------------------------------------
# Acesso administrativo humano: mesmo caminho do CI (IAP + OS Login), sem chave SSH fixa
# ---------------------------------------------------------------------------------------
resource "google_iap_tunnel_instance_iam_member" "admin_tunnel" {
  for_each = toset(var.admin_principals)
  project  = var.gcp_project_id
  zone     = var.gcp_zone
  instance = google_compute_instance.arborgrid_vm.name
  role     = "roles/iap.tunnelResourceAccessor"
  member   = each.value

  depends_on = [time_sleep.apis_propagation]
}

resource "google_compute_instance_iam_member" "admin_os_admin_login" {
  for_each      = toset(var.admin_principals)
  project       = var.gcp_project_id
  zone          = var.gcp_zone
  instance_name = google_compute_instance.arborgrid_vm.name
  role          = "roles/compute.osAdminLogin"
  member        = each.value
}

resource "google_service_account_iam_member" "admin_act_as_vm_runtime" {
  for_each           = toset(var.admin_principals)
  service_account_id = google_service_account.vm_runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = each.value
}
