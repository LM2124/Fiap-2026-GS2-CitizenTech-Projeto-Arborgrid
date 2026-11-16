terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.75"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu_24_04" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "arborgrid_sg" {
  name        = "${var.project_name}-sg"
  description = "Permite trafego HTTP/HTTPS publico ao MVP e restringe SSH a CIDR administrativo"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP publico (Web App + API ArborGrid via Nginx)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS publico (TLS)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH restrito ao CIDR administrativo (nunca 0.0.0.0/0)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Egress estritamente necessario para Docker Hub / repositorios APT / API Open-Meteo (HTTPS e HTTP)
  egress {
    description = "Saida HTTPS para pulls de imagem Docker e API Open-Meteo"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] #trivy:ignore:AVD-AWS-0104
  }

  egress {
    description = "Saida HTTP para repositorios de pacotes Ubuntu durante provisionamento"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] #trivy:ignore:AVD-AWS-0104
  }

  tags = {
    Name        = "${var.project_name}-sg"
    Project     = "ArborGrid-CitizenTech"
    Environment = "production"
  }
}

# Instancia publica controlada com SSH restrito a var.allowed_ssh_cidr e IMDSv2 obrigatorio
resource "aws_instance" "arborgrid_vm" { # nosemgrep: terraform.aws.security.aws-ec2-has-public-ip.aws-ec2-has-public-ip
  ami                         = data.aws_ami.ubuntu_24_04.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.arborgrid_sg.id]
  key_name                    = var.key_name
  associate_public_ip_address = true

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required" # IMDSv2 obrigatorio
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    project_name = var.project_name
  })

  tags = {
    Name        = "${var.project_name}-ec2"
    Project     = "ArborGrid-CitizenTech"
    ManagedBy   = "Terraform"
  }
}
