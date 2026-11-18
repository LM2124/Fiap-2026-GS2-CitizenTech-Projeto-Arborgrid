variable "gcp_project_id" {
  description = "ID do Projeto no Google Cloud Platform (ex.: meu-projeto-123456)"
  type        = string
}

variable "gcp_region" {
  description = "Regiao GCP de implantacao (us-central1 padrao de menor custo)"
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
  description = "Tipo da VM no Compute Engine (e2-medium = 2 vCPU, 4 GB RAM para FastAPI + Nginx + llama.cpp + Prometheus + Grafana)"
  type        = string
  default     = "e2-medium"
}

variable "ssh_user" {
  description = "Usuario administrativo Linux configurado na VM"
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key" {
  description = "Conteudo da chave publica SSH (ex.: ssh-ed25519 AAAA... gerada com ssh-keygen)"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "Bloco CIDR estrito autorizado para SSH (porta 22). NUNCA usar 0.0.0.0/0."
  type        = string

  validation {
    condition     = var.allowed_ssh_cidr != "0.0.0.0/0" && can(cidrhost(var.allowed_ssh_cidr, 0))
    error_message = "Por seguranca (DevSecOps), allowed_ssh_cidr deve ser um IP/CIDR valido e jamais 0.0.0.0/0."
  }
}
