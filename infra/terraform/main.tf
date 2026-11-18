terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  zone    = var.gcp_zone
}

# Regra de Firewall HTTP/HTTPS publica apenas para o proxy reverso Nginx do MVP
#trivy:ignore:AVD-GCP-0051
resource "google_compute_firewall" "arborgrid_web" {
  name        = "${var.project_name}-allow-web"
  network     = "default"
  description = "Permite trafego HTTP/HTTPS publico (80/443) ao Nginx do MVP ArborGrid"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["${var.project_name}-web"]
}

# Regra de Firewall SSH estritamente restrita ao CIDR administrativo (nunca 0.0.0.0/0)
resource "google_compute_firewall" "arborgrid_ssh" {
  name        = "${var.project_name}-allow-ssh-restricted"
  network     = "default"
  description = "Restringe SSH (22) exclusivamente ao IP/CIDR administrativo definido em allowed_ssh_cidr"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = [var.allowed_ssh_cidr]
  target_tags   = ["${var.project_name}-ssh"]
}

# Instancia Compute Engine Ubuntu 24.04 LTS com Shielded VM e chave SSH isolada por instancia
#trivy:ignore:AVD-GCP-0030
#trivy:ignore:AVD-GCP-0031
resource "google_compute_instance" "arborgrid_vm" { # nosemgrep
  name         = "${var.project_name}-vm"
  machine_type = var.machine_type
  zone         = var.gcp_zone

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
      # Aloca IPv4 publico efemero para acesso ao MVP Web
    }
  }

  metadata = {
    block-project-ssh-keys = "true"
    enable-oslogin         = "false"
    ssh-keys               = "${var.ssh_user}:${trimspace(var.ssh_public_key)}"
  }

  metadata_startup_script = templatefile("${path.module}/user_data.sh.tpl", {
    project_name = var.project_name
  })

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
}
