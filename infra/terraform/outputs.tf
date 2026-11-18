output "instance_id" {
  description = "ID da instancia EC2 provisionada"
  value       = aws_instance.arborgrid_vm.id
}

output "public_ip" {
  description = "IP publico da VM do MVP"
  value       = aws_instance.arborgrid_vm.public_ip
}

output "mvp_public_url" {
  description = "URL publica por IP do Web App ArborGrid"
  value       = "http://${aws_instance.arborgrid_vm.public_ip}"
}

output "mvp_aws_dns_url" {
  description = "URL publica com DNS automatico gerado pela AWS EC2"
  value       = "http://${aws_instance.arborgrid_vm.public_dns}"
}

output "grafana_ssh_tunnel_command" {
  description = "Comando seguro via SSH tunnel para abrir Grafana (3000) e Prometheus (9090) sem expor portas admin"
  value       = "ssh -i <SUA_CHAVE.pem> -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 ubuntu@${aws_instance.arborgrid_vm.public_ip}"
}
