output "instance_name" {
  description = "Nome da instancia Compute Engine provisionada no GCP"
  value       = google_compute_instance.arborgrid_vm.name
}

output "public_ip" {
  description = "IP publico estatico da VM do MVP (destino do registro DNS)"
  value       = google_compute_address.arborgrid_ip.address
}

output "mvp_public_url" {
  description = "URL publica do Web App ArborGrid (HTTPS via Caddy + Let's Encrypt)"
  value       = "https://${var.domain}"
}

output "admin_ssh_command" {
  description = "Acesso administrativo: OS Login pelo tunel do IAP (sem porta 22 publica nem chave fixa)"
  value       = "gcloud compute ssh ${google_compute_instance.arborgrid_vm.name} --zone ${var.gcp_zone} --project ${var.gcp_project_id} --tunnel-through-iap"
}

output "grafana_tunnel_command" {
  description = "Abre Grafana (3000) e Prometheus (9090) no seu computador pelo tunel do IAP, sem expor portas admin"
  value       = "gcloud compute ssh ${google_compute_instance.arborgrid_vm.name} --zone ${var.gcp_zone} --project ${var.gcp_project_id} --tunnel-through-iap -- -N -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090"
}

output "artifact_registry_image" {
  description = "Caminho da imagem do webapp no Artifact Registry"
  value       = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.arborgrid.repository_id}/webapp"
}

output "workload_identity_provider" {
  description = "Valor do secret GCP_WIF_PROVIDER do GitHub Actions"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deploy_service_account" {
  description = "Valor do secret GCP_DEPLOY_SA do GitHub Actions"
  value       = google_service_account.cicd.email
}
