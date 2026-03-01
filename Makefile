.PHONY: init plan apply deploy dev test clean

# Terraform: 初期化
init:
	terraform -chdir=terraform init

# Terraform: 変更プレビュー
plan:
	terraform -chdir=terraform plan

# Terraform: インフラ適用
apply:
	terraform -chdir=terraform apply

# gcloud: Cloud Functions 直接デプロイ（Terraform管理外の代替手段）
deploy:
	gcloud functions deploy bastet-interface \
		--gen2 \
		--runtime=python311 \
		--region=us-central1 \
		--source=. \
		--entry-point=inference \
		--trigger-http \
		--allow-unauthenticated

# ローカル実行（functions-framework）
dev:
	functions-framework --target=inference

# テスト実行
test:
	pytest tests/

# 一時ファイル削除
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
