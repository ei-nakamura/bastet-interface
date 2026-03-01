"""Bastet Inference — Unified Cloud Function for Vertex AI inference.

Handles vertex-claude (rawPredict), vertex-gemini (generateContent),
and document-ai (Document AI OCR + LLM translation) providers.
Authentication: Cloud Functions service account (ADC) for Vertex AI,
IAM (OIDC) for incoming requests from the Next.js app.
"""

import base64
import json
import os

import functions_framework
import google.auth
import google.auth.transport.requests
from flask import Request, jsonify

# ---------- Configuration ----------

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DOCUMENT_AI_PROCESSOR_ID = os.environ.get("DOCUMENT_AI_PROCESSOR_ID", "")
DOCUMENT_AI_LOCATION = os.environ.get("DOCUMENT_AI_LOCATION", "us")

ALLOWED_CLAUDE_MODELS = {
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-6-20250514",
}
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"

ALLOWED_GEMINI_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
}
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

VERTEX_AI_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# ---------- Auth ----------


def _get_access_token() -> str:
    """Get an access token for Vertex AI using ADC."""
    credentials, _ = google.auth.default(scopes=[VERTEX_AI_SCOPE])
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token


# ---------- Message formatting ----------


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert NormalizedMessage[] to Anthropic API message format."""
    result = []
    for msg in messages:
        blocks = []
        for block in msg.get("content", []):
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
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
    """Convert NormalizedMessage[] to Gemini API contents format."""
    result = []
    for msg in messages:
        parts = []
        for block in msg.get("content", []):
            if block["type"] == "text":
                parts.append({"text": block["text"]})
            elif block["type"] == "image":
                parts.append({
                    "inlineData": {
                        "mimeType": block["mediaType"],
                        "data": block["base64Data"],
                    }
                })
        role = "model" if msg["role"] == "assistant" else msg["role"]
        result.append({"role": role, "parts": parts})
    return result


# ---------- Provider handlers ----------


def _call_vertex_claude(messages: list[dict], model: str) -> dict:
    """Call Vertex AI Claude via rawPredict."""
    import requests

    resolved_model = model if model in ALLOWED_CLAUDE_MODELS else DEFAULT_CLAUDE_MODEL
    token = _get_access_token()
    anthropic_messages = _to_anthropic_messages(messages)

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
        try:
            err_msg = resp.json().get("error", {}).get("message", f"Vertex AI Error {resp.status_code}")
        except Exception:
            err_msg = f"Vertex AI Error {resp.status_code}"
        return {"error": err_msg}, resp.status_code

    data = resp.json()
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            text += block["text"]

    return {"text": text, "stop_reason": data.get("stop_reason", "unknown")}, 200


def _call_vertex_gemini(messages: list[dict], model: str) -> dict:
    """Call Vertex AI Gemini via generateContent."""
    import requests

    resolved_model = model if model in ALLOWED_GEMINI_MODELS else DEFAULT_GEMINI_MODEL
    token = _get_access_token()
    contents = _to_gemini_contents(messages)

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
    text = ""
    candidate = (data.get("candidates") or [{}])[0]
    for part in candidate.get("content", {}).get("parts", []):
        if part.get("text"):
            text += part["text"]

    finish_reason = candidate.get("finishReason", "unknown")
    if finish_reason == "MAX_TOKENS":
        stop_reason = "max_tokens"
    elif finish_reason == "STOP":
        stop_reason = "end_turn"
    else:
        stop_reason = finish_reason.lower()

    return {"text": text, "stop_reason": stop_reason}, 200


# ---------- Document AI ----------


def _extract_image_from_messages(messages: list[dict]) -> tuple[bytes, str]:
    """Extract the first image from NormalizedMessage[] as (bytes, mime_type)."""
    for msg in messages:
        for block in msg.get("content", []):
            if block.get("type") == "image":
                image_bytes = base64.b64decode(block["base64Data"])
                mime_type = block.get("mediaType", "image/png")
                return image_bytes, mime_type
    raise ValueError("No image found in messages")


def _ocr_with_document_ai(image_bytes: bytes, mime_type: str) -> "google.cloud.documentai.Document":
    """Call Document AI OCR processor and return the Document object."""
    from google.cloud import documentai

    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{DOCUMENT_AI_LOCATION}-documentai.googleapis.com"}
    )
    resource_name = client.processor_path(PROJECT_ID, DOCUMENT_AI_LOCATION, DOCUMENT_AI_PROCESSOR_ID)

    raw_document = documentai.RawDocument(content=image_bytes, mime_type=mime_type)
    request = documentai.ProcessRequest(name=resource_name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document


def _build_text_blocks_from_ocr(document) -> list[dict]:
    """Convert Document AI OCR result to text_blocks format.

    Each block has: id, content, bbox (normalized 0.0-1.0), confidence.
    """
    blocks = []
    block_id = 0

    for page in document.pages:
        for paragraph in page.paragraphs:
            # Extract text from text_anchor segments
            text = ""
            for segment in paragraph.layout.text_anchor.text_segments:
                start = int(segment.start_index)
                end = int(segment.end_index)
                text += document.text[start:end]
            text = text.strip().replace("\n", " ")
            if not text:
                continue

            # Convert normalized_vertices to bbox {x, y, w, h}
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
    """Use LLM to translate blocks and classify their types.

    Calls the appropriate Vertex AI model (Claude or Gemini) and merges
    translation + type back into each block.
    """
    if not blocks:
        return blocks

    # Build prompt
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

    # Route to appropriate LLM
    if model.startswith("gemini"):
        result, status = _call_vertex_gemini(messages, model)
    else:
        result, status = _call_vertex_claude(messages, model)

    if status != 200:
        # LLM call failed — return blocks without translation
        for b in blocks:
            b["type"] = "paragraph"
            b["translation"] = b["content"]
        return blocks

    # Parse LLM response
    try:
        text = result["text"].strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        parsed = json.loads(text)
        results = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
    except (json.JSONDecodeError, KeyError):
        # Fallback: no translation
        for b in blocks:
            b["type"] = "paragraph"
            b["translation"] = b["content"]
        return blocks

    # Merge translations into blocks
    translation_map = {r["id"]: r for r in results}
    for b in blocks:
        tr = translation_map.get(b["id"], {})
        b["type"] = tr.get("type", "paragraph")
        b["translation"] = tr.get("translation", b["content"])

    return blocks


def _call_document_ai(messages: list[dict], model: str, target_lang: str) -> tuple[dict, int]:
    """Main handler: Document AI OCR → LLM translation → merged JSON response."""
    try:
        image_bytes, mime_type = _extract_image_from_messages(messages)
    except ValueError as e:
        return {"error": str(e)}, 400

    # Step 1: OCR with Document AI
    document = _ocr_with_document_ai(image_bytes, mime_type)

    # Step 2: Build text blocks from OCR result
    blocks = _build_text_blocks_from_ocr(document)

    if not blocks:
        return {
            "text": json.dumps({"text_blocks": []}, ensure_ascii=False),
            "stop_reason": "end_turn",
        }, 200

    # Step 3: Translate + classify with LLM
    blocks = _translate_blocks(blocks, model, target_lang)

    # Step 4: Build response in same format as Claude Vision
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


# ---------- Entry point ----------


@functions_framework.http
def inference(request: Request):
    """Cloud Function entry point for inference requests."""
    # CORS preflight
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
        messages = body.get("messages")

        if not provider or not messages:
            return jsonify({"error": "provider and messages are required"}), 400

        if provider == "vertex-claude":
            result, status = _call_vertex_claude(messages, model)
        elif provider == "vertex-gemini":
            result, status = _call_vertex_gemini(messages, model)
        elif provider == "document-ai":
            target_lang = body.get("target_lang", "日本語")
            result, status = _call_document_ai(messages, model, target_lang)
        else:
            return jsonify({"error": f"Unknown provider: {provider}"}), 400

        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500
