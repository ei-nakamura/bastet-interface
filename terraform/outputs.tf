output "function_url" {
  description = "Cloud FunctionsのエンドポイントURL"
  value       = google_cloudfunctions2_function.function.service_config[0].uri
}

output "service_account_email" {
  description = "Cloud Functions用サービスアカウントのメールアドレス"
  value       = google_service_account.function_sa.email
}

output "cloudbuild_trigger_id" {
  description = "Cloud BuildトリガーID"
  value       = google_cloudbuild_trigger.deploy.trigger_id
}
