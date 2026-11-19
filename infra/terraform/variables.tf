variable "gcp_project_id" {
  description = "ID do Projeto no Google Cloud Platform (ex.: meu-projeto-123456)"
  type        = string
}

variable "gcp_region" {
  description = "Regiao GCP de implantacao (us-central1: menor custo e e2 disponivel)"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "Zona GCP da instancia Compute Engine"
  type        = string
  default     = "us-central1-a"
}

variable "project_name" {
  description = "Prefixo de identificacao dos recursos"
  type        = string
  default     = "arborgrid-mvp"
}

variable "machine_type" {
  description = "Tipo da VM (e2-standard-4 = 4 vCPU, 16 GB: llama.cpp com 4 threads responde ~2x mais rapido; e2-medium e o piso economico, com o assistente mais lento)"
  type        = string
  default     = "e2-standard-4"
}

variable "domain" {
  description = "Dominio publico do MVP (registro DNS A apontando para o IP estatico)"
  type        = string
  default     = "arborgrid.duckdns.org"
}

variable "static_ip_address" {
  description = "IP externo estatico ja reservado (o DNS aponta para ele). null aloca um novo endereco na primeira criacao."
  type        = string
  default     = null
}

variable "admin_principals" {
  description = "Quem administra a VM por IAP + OS Login (ex.: [\"user:fulano@gmail.com\"]). Sem chave SSH fixa."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for p in var.admin_principals : can(regex("^(user|group|serviceAccount):", p))])
    error_message = "Cada item de admin_principals deve comecar com user:, group: ou serviceAccount:."
  }
}

variable "artifact_registry_repository" {
  description = "Nome do repositorio Docker no Artifact Registry"
  type        = string
  default     = "arborgrid"
}

variable "github_repository" {
  description = "Repositorio GitHub (owner/nome) autorizado a assumir a Service Account de deploy via Workload Identity Federation"
  type        = string
  default     = "LM2124/Fiap-2026-GS2-CitizenTech-Projeto-Arborgrid"
}
