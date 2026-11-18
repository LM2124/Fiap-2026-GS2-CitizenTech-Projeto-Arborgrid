variable "aws_region" {
  description = "Regiao AWS de implantacao (us-east-1 padrao Learner Lab / Free Tier)"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefixo de identificacao dos recursos"
  type        = string
  default     = "arborgrid-mvp"
}

variable "instance_type" {
  description = "Tipo da instancia EC2 para hospedar o Docker Compose (FastAPI + Nginx + llama.cpp + Prometheus + Grafana)"
  type        = string
  default     = "t3.medium"
}

variable "key_name" {
  description = "Nome do Key Pair EC2 ja existente na AWS para acesso administrativo SSH"
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
