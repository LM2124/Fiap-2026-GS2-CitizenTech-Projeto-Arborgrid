output "instance_name" {
  description = "Nome da instancia Compute Engine provisionada no GCP"
  value       = google_compute_instance.arborgrid_vm.name
}

output "public_ip" {
  description = "IP publico externo da VM do MVP"
  value       = google_compute_instance.arborgrid_vm.network_interface[0].access_config[0].nat_ip
}

output "mvp_public_url" {
  description = "URL publica por IP do Web App ArborGrid"
  value       = "http://${google_compute_instance.arborgrid_vm.network_interface[0].access_config[0].nat_ip}"
}

output "grafana_ssh_tunnel_command" {
  description = "Comando seguro via SSH tunnel para abrir Grafana (3000) e Prometheus (9090) sem expor portas admin"
  value       = "ssh -i ~/.ssh/arborgrid_gcp -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 ${var.ssh_user}@${google_compute_instance.arborgrid_vm.network_interface[0].access_config[0].nat_ip}"
}
