"""Bastet Interface — Vertex AI 統合推論サーバー（Cloud Function）

vertex-claude（rawPredict）、vertex-gemini（generateContent）、
document-ai（Document AI OCR + LLMによる翻訳）の3プロバイダを統合処理する。
認証方式: Vertex AIへのアクセスにはCloud FunctionsサービスアカウントのADCを使用し、
受信リクエストの検証にはNext.jsアプリからのIAM（OIDC）認証を使用する。
"""

import base64
import json
import os

import functions_framework
import google.auth
import google.auth.transport.requests
from flask import Request, jsonify

# ---------- 設定 ----------

# GCPプロジェクトID（環境変数から取得。未設定の場合は空文字）
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
# Vertex AIのリージョン（デフォルト: us-central1）
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
# Document AIプロセッサID（環境変数から取得）
DOCUMENT_AI_PROCESSOR_ID = os.environ.get("DOCUMENT_AI_PROCESSOR_ID", "")
# Document AIのロケーション（マルチリージョン: us または eu）
DOCUMENT_AI_LOCATION = os.environ.get("DOCUMENT_AI_LOCATION", "us")

# 許可するClaudeモデルの一覧（Vertex AI経由でアクセス可能なモデルのみ）
ALLOWED_CLAUDE_MODELS = {
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-6-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-6-20250822",
    "claude-sonnet-4-5-20250514",
    "claude-haiku-4-5-20251001",
}
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6-20250514"

# 許可するGeminiモデルの一覧（Vertex AI経由でアクセス可能なモデルのみ）
ALLOWED_GEMINI_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
}
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Vertex AI APIアクセスに必要なOAuthスコープ
VERTEX_AI_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# ---------- 認証 ----------


def _get_access_token() -> str:
    """ADCを使用してVertex AI用のアクセストークンを取得する。

    Application Default Credentials（ADC）を用いてサービスアカウント認証を行い、
    Vertex AI APIへのリクエストに使用するBearerトークンを返す。
    """
    credentials, _ = google.auth.default(scopes=[VERTEX_AI_SCOPE])
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token


# ---------- メッセージ変換 ----------


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """NormalizedMessage[] をAnthropicAPIのメッセージ形式に変換する。

    フロントエンドの統一メッセージ形式（text / image ブロック）を
    Anthropic APIが要求するフォーマットに変換する。
    """
    result = []
    for msg in messages:
        blocks = []
        for block in msg.get("content", []):
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                # base64エンコードされた画像データをAnthropicのsource形式に変換
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block["mediaType"],
                        "data": block["base64Data"],
                    },
                })
        result.append({"role": msg["role"], "content": blocks})
    return result


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """NormalizedMessage[] をGemini APIのcontents形式に変換する。

    フロントエンドの統一メッセージ形式（text / image ブロック）を
    Gemini APIが要求するcontents/parts形式に変換する。
    Gemini APIでは"assistant"ロールを"model"として送信する必要がある点に注意。
    """
    result = []
    for msg in messages:
        parts = []
        for block in msg.get("content", []):
            if block["type"] == "text":
                parts.append({"text": block["text"]})
            elif block["type"] == "image":
                # base64画像データをGeminiのinlineData形式に変換
                parts.append({
                    "inlineData": {
                        "mimeType": block["mediaType"],
                        "data": block["base64Data"],
                    }
                })
        # Gemini APIはassistantロールを"model"として扱う
        role = "model" if msg["role"] == "assistant" else msg["role"]
        result.append({"role": role, "parts": parts})
    return result


# ---------- プロバイダハンドラ ----------


def _call_vertex_claude(messages: list[dict], model: str) -> dict:
    """Vertex AI経由でClaude（rawPredict）を呼び出す。

    モデル名が許可リストにない場合はデフォルトモデルにフォールバックする。
    レスポンスのcontentブロックからテキストを結合して返す。
    """
    import requests

    # 未知のモデル名はデフォルトモデルに差し替える
    resolved_model = model if model in ALLOWED_CLAUDE_MODELS else DEFAULT_CLAUDE_MODEL
    token = _get_access_token()
    anthropic_messages = _to_anthropic_messages(messages)

    # Vertex AI rawPredict エンドポイントを動的に構築
    endpoint = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"publishers/anthropic/models/{resolved_model}:rawPredict"
    )

    resp = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 40960,
            "messages": anthropic_messages,
        },
        timeout=120,
    )

    if not resp.ok:
        # エラー詳細をJSONから取り出せない場合はHTTPステータスコードを返す
        try:
            err_msg = resp.json().get("error", {}).get("message", f"Vertex AI Error {resp.status_code}")
        except Exception:
            err_msg = f"Vertex AI Error {resp.status_code}"
        return {"error": err_msg}, resp.status_code

    data = resp.json()
    # レスポンスのcontentブロックからtextタイプのみを結合
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            text += block["text"]

    return {"text": text, "stop_reason": data.get("stop_reason", "unknown")}, 200


def _call_vertex_gemini(messages: list[dict], model: str) -> dict:
    """Vertex AI経由でGemini（generateContent）を呼び出す。

    モデル名が許可リストにない場合はデフォルトモデルにフォールバックする。
    GeminiのfinishReasonをAnthropicと同じstop_reason形式に正規化して返す。
    """
    import requests

    # 未知のモデル名はデフォルトモデルに差し替える
    resolved_model = model if model in ALLOWED_GEMINI_MODELS else DEFAULT_GEMINI_MODEL
    token = _get_access_token()
    contents = _to_gemini_contents(messages)

    # Vertex AI generateContent エンドポイントを動的に構築
    endpoint = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"publishers/google/models/{resolved_model}:generateContent"
    )

    resp = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 40960},
        },
        timeout=120,
    )

    if not resp.ok:
        try:
            err_msg = resp.json().get("error", {}).get("message", f"Gemini API Error {resp.status_code}")
        except Exception:
            err_msg = f"Gemini API Error {resp.status_code}"
        return {"error": err_msg}, resp.status_code

    data = resp.json()
    # 最初のcandidateからテキストpartsを結合する
    text = ""
    candidate = (data.get("candidates") or [{}])[0]
    for part in candidate.get("content", {}).get("parts", []):
        if part.get("text"):
            text += part["text"]

    # GeminiのfinishReasonをフロントエンド共通のstop_reason形式に変換
    finish_reason = candidate.get("finishReason", "unknown")
    if finish_reason == "MAX_TOKENS":
        stop_reason = "max_tokens"
    elif finish_reason == "STOP":
        stop_reason = "end_turn"
    else:
        stop_reason = finish_reason.lower()

    return {"text": text, "stop_reason": stop_reason}, 200


# ---------- Document AI 処理 ----------


def _extract_image_from_messages(messages: list[dict]) -> tuple[bytes, str]:
    """NormalizedMessage[] から最初の画像を（バイト列, MIMEタイプ）として取り出す。

    メッセージリストを順に走査し、最初に見つかったimageブロックの
    base64データをデコードして返す。画像が存在しない場合はValueErrorを送出する。
    """
    for msg in messages:
        for block in msg.get("content", []):
            if block.get("type") == "image":
                image_bytes = base64.b64decode(block["base64Data"])
                mime_type = block.get("mediaType", "image/png")
                return image_bytes, mime_type
    raise ValueError("No image found in messages")


def _ocr_with_document_ai(image_bytes: bytes, mime_type: str) -> "google.cloud.documentai.Document":
    """Document AI OCRプロセッサを呼び出し、Documentオブジェクトを返す。

    環境変数で指定されたプロセッサリソースに対してOCR処理を実行し、
    段落・バウンディングボックス・信頼度スコアを含むDocumentオブジェクトを返す。
    """
    from google.cloud import documentai

    # Document AIクライアントをロケーション固有のエンドポイントで初期化
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{DOCUMENT_AI_LOCATION}-documentai.googleapis.com"}
    )
    resource_name = client.processor_path(PROJECT_ID, DOCUMENT_AI_LOCATION, DOCUMENT_AI_PROCESSOR_ID)

    raw_document = documentai.RawDocument(content=image_bytes, mime_type=mime_type)
    request = documentai.ProcessRequest(name=resource_name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document


def _build_text_blocks_from_ocr(document) -> list[dict]:
    """Document AI OCR結果をtext_blocks形式に変換する。

    各ブロックは以下のフィールドを持つ:
    - id: 連番ID（1始まり）
    - content: OCRで抽出したテキスト
    - bbox: 正規化バウンディングボックス（0.0〜1.0）{x, y, w, h}
    - confidence: OCR信頼度スコア
    """
    blocks = []
    block_id = 0

    for page in document.pages:
        for paragraph in page.paragraphs:
            # text_anchorのセグメント情報を使いdocument.textから文字列を取り出す
            text = ""
            for segment in paragraph.layout.text_anchor.text_segments:
                start = int(segment.start_index)
                end = int(segment.end_index)
                text += document.text[start:end]
            text = text.strip().replace("\n", " ")
            if not text:
                # 空白のみの段落はスキップ
                continue

            # normalized_verticesから軸平行バウンディングボックス {x, y, w, h} を算出
            vertices = paragraph.layout.bounding_poly.normalized_vertices
            if not vertices:
                continue
            xs = [v.x for v in vertices]
            ys = [v.y for v in vertices]
            bbox = {
                "x": round(min(xs), 6),
                "y": round(min(ys), 6),
                "w": round(max(xs) - min(xs), 6),
                "h": round(max(ys) - min(ys), 6),
            }

            confidence = round(paragraph.layout.confidence, 4) if paragraph.layout.confidence else 0.9
            block_id += 1
            blocks.append({
                "id": block_id,
                "content": text,
                "bbox": bbox,
                "confidence": confidence,
            })

    return blocks


def _translate_blocks(blocks: list[dict], model: str, target_lang: str) -> list[dict]:
    """LLMを使ってブロックを翻訳し、テキストタイプを分類する。

    モデル名に応じてVertex AI Claude または Gemini を呼び出し、
    翻訳結果とタイプ分類を各ブロックにマージして返す。
    LLM呼び出しが失敗した場合は元テキストをそのままtranslationに設定する。
    """
    if not blocks:
        return blocks

    # LLMへのプロンプトを構築（ブロックID付きで翻訳とタイプ分類を依頼）
    blocks_text = "\n".join(f'[{b["id"]}] "{b["content"]}"' for b in blocks)
    prompt = f"""以下のテキストブロック（OCRで検出済み）を{target_lang}に翻訳し、各ブロックのタイプを分類してください。

テキストブロック:
{blocks_text}

各ブロックについてJSON形式で出力:
{{
  "results": [
    {{"id": 1, "type": "title"|"heading"|"paragraph"|"list"|"table"|"caption"|"header"|"footer"|"other", "translation": "翻訳テキスト"}},
    ...
  ]
}}

注意:
- 翻訳は自然で流暢な{target_lang}にすること
- typeはテキストの内容・位置・フォントサイズの推測から適切に分類
- JSONのみ出力、マークダウンの囲みや説明文は不要"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    # モデル名のプレフィックスでGemini/Claudeを振り分け
    if model.startswith("gemini"):
        result, status = _call_vertex_gemini(messages, model)
    else:
        result, status = _call_vertex_claude(messages, model)

    if status != 200:
        # LLM呼び出し失敗時は元テキストをtranslationとしてフォールバック
        for b in blocks:
            b["type"] = "paragraph"
            b["translation"] = b["content"]
        return blocks

    # LLMレスポンスのJSONをパース
    try:
        text = result["text"].strip()
        # マークダウンコードフェンスが含まれている場合は除去
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        parsed = json.loads(text)
        results = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
    except (json.JSONDecodeError, KeyError):
        # JSONパース失敗時は元テキストをそのまま返す
        for b in blocks:
            b["type"] = "paragraph"
            b["translation"] = b["content"]
        return blocks

    # ブロックIDをキーにして翻訳結果をマージ
    translation_map = {r["id"]: r for r in results}
    for b in blocks:
        tr = translation_map.get(b["id"], {})
        b["type"] = tr.get("type", "paragraph")
        b["translation"] = tr.get("translation", b["content"])

    return blocks


def _call_document_ai(messages: list[dict], model: str, target_lang: str) -> tuple[dict, int]:
    """メインハンドラ: Document AI OCR → LLM翻訳 → JSON形式でレスポンスを構築する。

    画像抽出 → OCR → テキストブロック変換 → LLM翻訳 の4ステップで処理し、
    Claude Vision互換のtext_blocks形式でレスポンスを返す。
    """
    try:
        image_bytes, mime_type = _extract_image_from_messages(messages)
    except ValueError as e:
        return {"error": str(e)}, 400

    # Step 1: Document AIでOCR処理を実行
    document = _ocr_with_document_ai(image_bytes, mime_type)

    # Step 2: OCR結果をtext_blocks形式に変換
    blocks = _build_text_blocks_from_ocr(document)

    if not blocks:
        # OCR結果が空の場合は空のtext_blocksを返す
        return {
            "text": json.dumps({"text_blocks": []}, ensure_ascii=False),
            "stop_reason": "end_turn",
        }, 200

    # Step 3: LLMで翻訳とタイプ分類を実行
    blocks = _translate_blocks(blocks, model, target_lang)

    # Step 4: Claude Vision互換のtext_blocks形式でレスポンスを組み立てる
    text_blocks = []
    for b in blocks:
        text_blocks.append({
            "id": b["id"],
            "type": b.get("type", "paragraph"),
            "content": b["content"],
            "translation": b.get("translation", ""),
            "bbox": b["bbox"],
            "confidence": b["confidence"],
        })

    response_json = json.dumps({"text_blocks": text_blocks}, ensure_ascii=False)
    return {"text": response_json, "stop_reason": "end_turn"}, 200


# ---------- 新リクエスト形式ハンドラ ----------


def _build_detect_translate_prompt(target_lang: str, img_w: int, img_h: int) -> str:
    return f"""この書類画像を解析し、文章のかたまり（テキストブロック）を検出し、同時に{target_lang}に翻訳してください。

    画像サイズ: 幅 {img_w}px × 高さ {img_h}px

    各テキストブロックについて以下のJSON形式で出力:
    - id: 連番
    - type: "title"|"heading"|"paragraph"|"list"|"table"|"caption"|"header"|"footer"|"other"
    - content: 原文テキスト内容（改行は半角スペースに置換。省略せず全文を含めること）
    - translation: {target_lang}に翻訳したテキスト
    - bbox: 正規化された境界ボックス座標（画像の幅・高さを1.0とした0.0〜1.0の小数）
      - x: 左端の位置（0.0=左端、1.0=右端）
      - y: 上端の位置（0.0=上端、1.0=下端）
      - w: 幅（0.0〜1.0）
      - h: 高さ（0.0〜1.0）
    - confidence: 0.0〜1.0

    注意:
    1. 座標は必ず0.0〜1.0の範囲内にすること
    2. x+w<=1.0, y+h<=1.0であること
    3. すべてのテキストを漏れなく検出
    4. 翻訳は自然で流暢な{target_lang}にすること
    5. JSONのみ出力、マークダウンの囲みや説明文は不要

    出力: {{ "text_blocks": [ ... ] }}"""


def _build_translate_prompt(blocks: list[dict], target_lang: str) -> str:
    blocks_text = "\n".join(f'[{b["id"]}] {b["content"]}' for b in blocks)
    return f"以下のテキストブロックを{target_lang}に翻訳してください。\n\n{blocks_text}\n\nJSON配列のみ出力（マークダウンの囲みや説明文は不要）:\n[ {{ \"id\": 1, \"translation\": \"翻訳文\" }}, ... ]"


def _parse_json_response(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def _handle_image_request(image: dict, target_lang: str, img_w: int, img_h: int, model: str, provider: str) -> tuple[dict, int]:
    base64_data = image.get("base64Data", "")
    media_type = image.get("mediaType", "image/png")

    if provider == "document-ai":
        image_bytes = base64.b64decode(base64_data)
        document = _ocr_with_document_ai(image_bytes, media_type)
        blocks = _build_text_blocks_from_ocr(document)
        if not blocks:
            return {"text_blocks": []}, 200
        blocks = _translate_blocks(blocks, model, target_lang)
        text_blocks = [
            {
                "id": b["id"],
                "type": b.get("type", "paragraph"),
                "content": b["content"],
                "translation": b.get("translation", ""),
                "bbox": b["bbox"],
                "confidence": b["confidence"],
            }
            for b in blocks
        ]
        return {"text_blocks": text_blocks}, 200
    else:
        prompt = _build_detect_translate_prompt(target_lang, img_w, img_h)
        messages = [{"role": "user", "content": [
            {"type": "image", "mediaType": media_type, "base64Data": base64_data},
            {"type": "text", "text": prompt},
        ]}]
        if provider == "vertex-gemini":
            result, status = _call_vertex_gemini(messages, model)
        else:
            result, status = _call_vertex_claude(messages, model)
        if status != 200:
            return result, status
        try:
            parsed = _parse_json_response(result["text"])
            text_blocks = parsed.get("text_blocks", parsed) if isinstance(parsed, dict) else parsed
            return {"text_blocks": text_blocks}, 200
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"error": "Failed to parse LLM response", "raw": result.get("text", "")}, 500


def _handle_text_translation_request(text_blocks: list[dict], target_lang: str, model: str, provider: str) -> tuple[dict, int]:
    prompt = _build_translate_prompt(text_blocks, target_lang)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if provider == "vertex-gemini":
        result, status = _call_vertex_gemini(messages, model)
    else:
        result, status = _call_vertex_claude(messages, model)
    if status != 200:
        return result, status
    try:
        parsed = _parse_json_response(result["text"])
        translations = parsed if isinstance(parsed, list) else parsed.get("translations", [])
        trans_map = {t["id"]: t["translation"] for t in translations}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"error": "Failed to parse translation response"}, 500
    output_blocks = []
    for b in text_blocks:
        block = {
            "id": b["id"],
            "type": b.get("type", "paragraph"),
            "content": b["content"],
            "translation": trans_map.get(b["id"], b["content"]),
        }
        if "bbox" in b:
            block["bbox"] = b["bbox"]
        if "confidence" in b:
            block["confidence"] = b["confidence"]
        output_blocks.append(block)
    return {"text_blocks": output_blocks}, 200


# ---------- エントリポイント ----------


@functions_framework.http
def inference(request: Request):
    """Cloud Functionの推論リクエスト受付エントリポイント。

    POSTリクエストのbodyからprovider・model・image or text_blocksを取得し、
    各プロバイダハンドラに処理を委譲する。
    CORSプリフライト（OPTIONS）も処理する。
    """
    # CORSプリフライトリクエストへの応答
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    try:
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "Invalid JSON body"}), 400

        provider = body.get("provider")
        model = body.get("model", "")
        target_lang = body.get("targetLang") or body.get("target_lang", "日本語")

        if not provider:
            return jsonify({"error": "provider is required"}), 400

        # リクエスト種別で分岐
        if "image" in body:
            image = body.get("image")
            img_w = body.get("imageWidth", 0)
            img_h = body.get("imageHeight", 0)
            result, status = _handle_image_request(image, target_lang, img_w, img_h, model, provider)
        elif "text_blocks" in body:
            text_blocks = body.get("text_blocks")
            if provider == "document-ai":
                return jsonify({"error": "document-ai does not support text_blocks request"}), 400
            result, status = _handle_text_translation_request(text_blocks, target_lang, model, provider)
        else:
            return jsonify({"error": "Either 'image' or 'text_blocks' is required"}), 400

        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500
