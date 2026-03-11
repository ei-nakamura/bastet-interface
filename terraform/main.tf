# ---------- API 有効化 ----------

resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------- ソースコードアーカイブ ----------

resource "google_storage_bucket" "source_bucket" {
  name                        = "${var.project_id}-${var.function_name}-source"
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true

  depends_on = [google_project_service.apis]
}

data "archive_file" "source" {
  type        = "zip"
  source_dir  = "${path.module}/.."
  output_path = "${path.module}/.terraform/source.zip"
  excludes    = ["terraform", ".terraform", "__pycache__", ".git"]
}

resource "google_storage_bucket_object" "source_archive" {
  name   = "source-${data.archive_file.source.output_md5}.zip"
  bucket = google_storage_bucket.source_bucket.name
  source = data.archive_file.source.output_path
}

# ---------- サービスアカウント ----------

resource "google_service_account" "function_sa" {
  account_id   = "${var.function_name}-sa"
  display_name = "Bastet Interface Service Account"
  project      = var.project_id
}

# ---------- IAM バインディング ----------

resource "google_project_iam_member" "aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.function_sa.email}"
}

# ---------- Cloud Functions Gen2 ----------

resource "google_cloudfunctions2_function" "function" {
  name     = var.function_name
  location = var.region
  project  = var.project_id

  build_config {
    runtime     = "python312"
    entry_point = "inference"
    source {
      storage_source {
        bucket = google_storage_bucket.source_bucket.name
        object = google_storage_bucket_object.source_archive.name
      }
    }
  }

  service_config {
    max_instance_count    = 10
    available_memory      = "512M"
    timeout_seconds       = 120
    service_account_email = google_service_account.function_sa.email
    environment_variables = {
      GOOGLE_CLOUD_PROJECT = var.project_id
    }
  }

  depends_on = [
    google_project_service.apis,
    google_storage_bucket_object.source_archive,
  ]
}

# ---------- Cloud Build トリガー ----------

resource "google_cloudbuild_trigger" "deploy" {
  name     = "${var.function_name}-deploy"
  project  = var.project_id
  location = var.region

  github {
    owner = var.github_owner
    name  = var.github_repo

    push {
      branch = var.github_branch
    }
  }

  filename = "cloudbuild.yaml"

  substitutions = {
    _REGION        = var.region
    _FUNCTION_NAME = var.function_name
  }

  depends_on = [google_project_service.apis]
}

# ---------- Cloud Run IAM: 未認証アクセス許可 ----------

resource "google_cloud_run_service_iam_member" "invoker" {
  project  = var.project_id
  service  = google_cloudfunctions2_function.function.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
