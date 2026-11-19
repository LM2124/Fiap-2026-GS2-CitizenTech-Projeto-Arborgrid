terraform {
  required_version = ">= 1.6.0"

  # Estado remoto em bucket GCS versionado (criado uma vez no bootstrap, fora deste estado).
  # O nome do bucket vai na inicializacao: terraform init -backend-config="bucket=<projeto>-tfstate"
  backend "gcs" {
    prefix = "arborgrid/production"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

# Credenciais: o operador assina com a propria conta (gcloud auth application-default login)
# e impersona a service account terraform-deployer via GOOGLE_IMPERSONATE_SERVICE_ACCOUNT.
# Nenhuma chave JSON de longa duracao.
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  zone    = var.gcp_zone
}

# HTTP/HTTPS publico apenas para o Caddy (TLS) na frente do Nginx do MVP
#trivy:ignore:AVD-GCP-0051
resource "google_compute_firewall" "arborgrid_web" {
  name        = "${var.project_name}-allow-web"
  network     = "default"
  description = "Permite trafego HTTP/HTTPS publico (80/443) ao Caddy do MVP ArborGrid"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["${var.project_name}-web"]
}

# IP externo estatico: o registro DNS (DuckDNS) aponta para ele, entao nao pode mudar.
resource "google_compute_address" "arborgrid_ip" {
  name         = "${var.project_name}-ip"
  region       = var.gcp_region
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
  address      = var.static_ip_address

  lifecycle {
    prevent_destroy = true
  }
}

# VM Ubuntu 24.04 LTS com Shielded VM. Sem chaves SSH no metadata: o acesso e por OS Login
# (identidade do Google + IAM), sempre pelo tunel do Identity-Aware Proxy.
#trivy:ignore:AVD-GCP-0030
#trivy:ignore:AVD-GCP-0031
resource "google_compute_instance" "arborgrid_vm" { # nosemgrep
  name         = "${var.project_name}-vm"
  machine_type = var.machine_type
  zone         = var.gcp_zone

  # Trocar a service account ou o tipo da maquina exige parar a VM (~1 min)
  allow_stopping_for_update = true

  tags = [
    "${var.project_name}-web",
    "${var.project_name}-ssh",
  ]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 30
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.arborgrid_ip.address
    }
  }

  metadata = {
    enable-oslogin          = "TRUE"
    enable-guest-attributes = "TRUE" # publica as chaves de host para o CI fixa-las sem TOFU
    block-project-ssh-keys  = "true"
  }

  metadata_startup_script = templatefile("${path.module}/user_data.sh.tpl", {
    project_name = var.project_name
  })

  # Identidade propria e minima: so le imagens do Artifact Registry do projeto
  service_account {
    email  = google_service_account.vm_runtime.email
    scopes = ["https://www.googleapis.com/auth/devstorage.read_only"]
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  labels = {
    project     = "arborgrid-citizentech"
    environment = "production"
    managed_by  = "terraform"
  }

  depends_on = [google_artifact_registry_repository_iam_member.vm_reader]
}
