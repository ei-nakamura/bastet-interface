# bastet-interface

Vertex AI 推論サーバー（Google Cloud Functions）

Bastetフロントエンドからの推論リクエストを受け取り、Vertex AI（Claude / Gemini）への
アクセスを一元的に提供するHTTPエンドポイント。

---

## 技術スタック

| カテゴリ | 技術 |
|----------|------|
| ランタイム | Python 3.11+ |
| サーバーレス | Google Cloud Functions (Gen 2) |
| LLM | Vertex AI — Claude（rawPredict）/ Gemini（generateContent） |
| 認証 | Application Default Credentials (ADC) + IAM OIDC |
| フレームワーク | functions-framework 3.x |

---

## 前提条件

- Google Cloud プロジェクトが作成済みであること
- Vertex AI API が有効化されていること
- Cloud Functions のサービスアカウントに以下のロールが付与されていること:
  - `roles/aiplatform.user`
- ローカル開発時は `gcloud auth application-default login` でADCが設定されていること

---

## 環境変数

| 変数名 | 必須 | デフォルト | 説明 |
|--------|------|-----------|------|
| `GOOGLE_CLOUD_PROJECT` | ✅ | — | GCPプロジェクトID |
| `GOOGLE_CLOUD_LOCATION` | | `us-central1` | Vertex AIのリージョン |

---

## ローカル開発

```bash
# 仮想環境の作成と依存パッケージのインストール
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 環境変数の設定
export GOOGLE_CLOUD_PROJECT=your-project-id

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
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-project-id
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

#### (A) 画像解析+翻訳リクエスト

```json
{
  "provider": "vertex-claude" | "vertex-gemini",
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

### プロバイダ・モデル一覧

| プロバイダ | 対応リクエスト | 動作 |
|-----------|-------------|------|
| `vertex-claude` | 画像・テキスト両対応 | Vertex AI rawPredict経由でClaudeを呼び出す |
| `vertex-gemini` | 画像・テキスト両対応 | Vertex AI generateContent経由でGeminiを呼び出す |

#### Claude モデル（vertex-claude）

Vertex AIのモデルIDは `@日付` 形式（Claude API公式ID `-日付` 形式とは異なる）。
最新世代（Opus 4.7 / Sonnet 4.6 / Opus 4.6）は日付サフィックス不要。

| モデルID | 段階 | デフォルト |
|---------|------|-----------|
| `claude-opus-4-7` | GA（最新） | |
| `claude-sonnet-4-6` | GA（最新） | ✅ |
| `claude-haiku-4-5@20251001` | GA（最新） | |
| `claude-opus-4-6` | Legacy | |
| `claude-sonnet-4-5@20250929` | Legacy | |
| `claude-opus-4-5@20251101` | Legacy | |
| `claude-opus-4-1@20250805` | Legacy | |
| `claude-sonnet-4@20250514` | Deprecated（2026-06-15 引退） | |
| `claude-opus-4@20250514` | Deprecated（2026-06-15 引退） | |

#### Gemini モデル（vertex-gemini）

| モデルID | 段階 | デフォルト |
|---------|------|-----------|
| `gemini-2.5-flash` | GA | ✅ |
| `gemini-2.5-pro` | GA | |
| `gemini-2.5-flash-lite` | GA | |
| `gemini-3.1-pro-preview` | Preview | |
| `gemini-3-flash-preview` | Preview | |
| `gemini-3.1-flash-lite-preview` | Preview | |

> **注意**: Gemini 3.x系のPreviewモデルはグローバルエンドポイントでのみ提供される。
> これらのモデルを利用する場合は `GOOGLE_CLOUD_LOCATION=global` を設定すること。
> また `gemini-3-pro-preview` は2026年3月26日に廃止済み（`gemini-3.1-pro-preview` へ移行）。

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
| 400 | リクエストボディが不正 / 必須フィールド不足 |
| 405 | POST以外のメソッド |
| 500 | 内部エラー（Vertex AI APIエラー含む） |

---

## テスト実行

```bash
# 仮想環境をアクティブにした状態で実行
pip install pytest
pytest tests/ -v
```

テストはすべて外部APIをMockで代替するため、GCPの認証情報なしで実行可能。
