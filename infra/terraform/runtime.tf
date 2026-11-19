# Registro de imagens e identidade de execucao da VM.
# O CI constroi e escaneia a imagem uma unica vez e a publica aqui; a VM baixa exatamente
# aquele digest. O que foi escaneado e o que roda em producao.

resource "google_artifact_registry_repository" "arborgrid" {
  repository_id = var.artifact_registry_repository
  location      = var.gcp_region
  format        = "DOCKER"
  description   = "Imagens do MVP ArborGrid SP publicadas pelo GitHub Actions"

  # Guarda as 10 versoes mais recentes; o resto e apagado automaticamente
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "manter-10-recentes"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }
  cleanup_policies {
    id     = "apagar-antigas"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "604800s" # 7 dias
    }
  }

  depends_on = [time_sleep.apis_propagation]
}

# Service account da VM: no lugar da conta padrao do Compute Engine (que tem papel de Editor)
resource "google_service_account" "vm_runtime" {
  account_id   = "${var.project_name}-vm-runtime"
  display_name = "ArborGrid VM runtime (pull de imagens)"
}

resource "google_artifact_registry_repository_iam_member" "vm_reader" {
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.arborgrid.location
  repository = google_artifact_registry_repository.arborgrid.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.vm_runtime.email}"
}
