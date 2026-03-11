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

variable "github_owner" {
  description = "GitHubリポジトリのオーナー（ユーザー名 or 組織名）"
  type        = string
}

variable "github_repo" {
  description = "GitHubリポジトリ名"
  type        = string
  default     = "bastet-interface"
}

variable "github_branch" {
  description = "Cloud Buildトリガーの対象ブランチ（正規表現）"
  type        = string
  default     = "^main$"
}
