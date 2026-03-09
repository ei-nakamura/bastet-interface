# bastet-interface

Vertex AI + Document AI 統合推論サーバー（Google Cloud Functions）

Bastetフロントエンドからの推論リクエストを受け取り、Vertex AI（Claude / Gemini）および
Document AI OCRへのアクセスを一元的に提供するHTTPエンドポイント。

---

## 技術スタック

| カテゴリ | 技術 |
|----------|------|
| ランタイム | Python 3.11+ |
| サーバーレス | Google Cloud Functions (Gen 2) |
| LLM | Vertex AI — Claude（rawPredict）/ Gemini（generateContent） |
| OCR | Google Cloud Document AI |
| 認証 | Application Default Credentials (ADC) + IAM OIDC |
| フレームワーク | functions-framework 3.x |

---

## 前提条件

- Google Cloud プロジェクトが作成済みであること
- Vertex AI API が有効化されていること
- Document AI API が有効化されていること
- Document AI OCR プロセッサが作成済みであること
- Cloud Functions のサービスアカウントに以下のロールが付与されていること:
  - `roles/aiplatform.user`
  - `roles/documentai.apiUser`
- ローカル開発時は `gcloud auth application-default login` でADCが設定されていること

---

## 環境変数

| 変数名 | 必須 | デフォルト | 説明 |
|--------|------|-----------|------|
| `GOOGLE_CLOUD_PROJECT` | ✅ | — | GCPプロジェクトID |
| `GOOGLE_CLOUD_LOCATION` | | `us-central1` | Vertex AIのリージョン |
| `DOCUMENT_AI_PROCESSOR_ID` | ✅ | — | Document AIプロセッサID |
| `DOCUMENT_AI_LOCATION` | | `us` | Document AIのロケーション（`us` または `eu`） |

---

## ローカル開発

```bash
# 仮想環境の作成と依存パッケージのインストール
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 環境変数の設定
export GOOGLE_CLOUD_PROJECT=your-project-id
export DOCUMENT_AI_PROCESSOR_ID=your-processor-id

# ADC認証（初回のみ）
gcloud auth application-default login

# ローカルサーバー起動（ポート8080）
functions-framework --target=inference --port=8080
```

---

## デプロイ

```bash
gcloud functions deploy bastet-interface \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=inference \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-project-id,DOCUMENT_AI_PROCESSOR_ID=your-processor-id
```

> **注意**: `--allow-unauthenticated` を使用する場合はCloud ArmorやIAMポリシーで
> フロントエンドIPからのアクセスのみ許可することを推奨する。

---

## API仕様

### エンドポイント

```
POST /
Content-Type: application/json
```

### リクエスト形式

リクエストは2種類。`image` または `text_blocks` のいずれか一方を必ず含める。

#### (A) 画像解析+翻訳リクエスト（全プロバイダ対応）

```json
{
  "provider": "vertex-claude" | "vertex-gemini" | "document-ai",
  "model": "claude-sonnet-4-6-20250514",
  "image": {
    "base64Data": "<base64エンコードされた画像>",
    "mediaType": "image/png"
  },
  "targetLang": "日本語",
  "imageWidth": 1536,
  "imageHeight": 2048
}
```

#### (B) テキスト翻訳リクエスト（vertex-claude / vertex-gemini のみ）

```json
{
  "provider": "vertex-claude" | "vertex-gemini",
  "model": "claude-sonnet-4-6-20250514",
  "text_blocks": [
    { "id": 1, "content": "原文テキスト" }
  ],
  "targetLang": "日本語"
}
```

> **注意**: `document-ai` プロバイダは `text_blocks` リクエスト非対応（400エラー）。

### プロバイダ・モデル一覧

| プロバイダ | 対応リクエスト | 動作 |
|-----------|-------------|------|
| `vertex-claude` | 画像・テキスト両対応 | Vertex AI rawPredict経由でClaudeを呼び出す |
| `vertex-gemini` | 画像・テキスト両対応 | Vertex AI generateContent経由でGeminiを呼び出す |
| `document-ai` | 画像のみ | Document AI OCR → LLM翻訳の2段階処理 |

#### Claude モデル（vertex-claude / document-ai）

| モデルID | デフォルト |
|---------|-----------|
| `claude-sonnet-4-6-20250514` | ✅ |
| `claude-sonnet-4-20250514` | |
| `claude-opus-4-6-20250822` | |
| `claude-opus-4-20250514` | |
| `claude-sonnet-4-5-20250514` | |
| `claude-haiku-4-5-20251001` | |

#### Gemini モデル（vertex-gemini）

| モデルID | デフォルト |
|---------|-----------|
| `gemini-2.5-flash` | ✅ |
| `gemini-2.5-pro` | |
| `gemini-2.5-flash-lite` | |

モデル未指定または未知のモデル名はデフォルトモデルにフォールバックする。

### レスポンス形式（両パターン共通）

```json
{
  "text_blocks": [
    {
      "id": 1,
      "type": "title",
      "content": "原文テキスト",
      "translation": "翻訳されたテキスト",
      "bbox": { "x": 0.05, "y": 0.1, "w": 0.9, "h": 0.08 },
      "confidence": 0.95
    }
  ]
}
```

> **注意**: `text_blocks` リクエスト（B）の場合、`bbox` / `confidence` は元データに含まれる場合のみ出力される。

#### text_block フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `id` | int | ブロックの連番ID（1始まり） |
| `type` | string | テキスト種別（`title` / `heading` / `paragraph` / `list` / `table` / `caption` / `header` / `footer` / `other`） |
| `content` | string | OCRで抽出した原文テキスト |
| `translation` | string | LLMによる翻訳テキスト |
| `bbox` | object | 正規化バウンディングボックス `{x, y, w, h}`（0.0〜1.0） |
| `confidence` | float | OCR信頼度スコア |

### エラーレスポンス

```json
{
  "error": "エラーメッセージ"
}
```

| HTTPステータス | 原因 |
|--------------|------|
| 400 | リクエストボディが不正 / 必須フィールド不足 / document-ai に text_blocks を送信 |
| 405 | POST以外のメソッド |
| 500 | 内部エラー（Vertex AI / Document AI APIエラー含む） |

---

## テスト実行

```bash
# 仮想環境をアクティブにした状態で実行
pip install pytest
pytest tests/ -v
```

テストはすべて外部APIをMockで代替するため、GCPの認証情報なしで実行可能。
