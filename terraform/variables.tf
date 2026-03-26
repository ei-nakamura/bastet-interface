variable "project_id" {
  description = "GCPプロジェクトID（必須）"
  type        = string
}

variable "region" {
  description = "デプロイリージョン"
  type        = string
  default     = "us-central1"
}

variable "function_name" {
  description = "Cloud Functions の関数名"
  type        = string
  default     = "bastet-interface"
}
