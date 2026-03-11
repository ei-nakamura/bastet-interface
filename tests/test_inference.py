"""bastet-inferenceの推論ハンドラ関数に関するテスト群。

_build_detect_translate_prompt, _build_translate_prompt, _parse_json_response,
_handle_image_request, _handle_text_translation_request, inference()ルーティングを検証する。
"""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------- _build_detect_translate_prompt ----------

class TestBuildDetectTranslatePrompt:
    """_build_detect_translate_prompt 関数のテスト。"""

    def test_contains_target_lang(self):
        """プロンプトに target_lang が含まれること。"""
        from main import _build_detect_translate_prompt

        prompt = _build_detect_translate_prompt("English", 800, 600)
        assert "English" in prompt

    def test_contains_image_dimensions(self):
        """プロンプトに img_w と img_h が含まれること。"""
        from main import _build_detect_translate_prompt

        prompt = _build_detect_translate_prompt("日本語", 1024, 768)
        assert "1024" in prompt
        assert "768" in prompt

    def test_contains_text_blocks_keyword(self):
        """プロンプトに "text_blocks" キーワードが含まれること。"""
        from main import _build_detect_translate_prompt

        prompt = _build_detect_translate_prompt("English", 800, 600)
        assert "text_blocks" in prompt

    def test_contains_bbox_keyword(self):
        """プロンプトに "bbox" キーワードが含まれること。"""
        from main import _build_detect_translate_prompt

        prompt = _build_detect_translate_prompt("English", 800, 600)
        assert "bbox" in prompt


# ---------- _build_translate_prompt ----------

class TestBuildTranslatePrompt:
    """_build_translate_prompt 関数のテスト。"""

    def test_contains_block_id_and_content(self):
        """ブロックのidとcontentがプロンプトに含まれること。"""
        from main import _build_translate_prompt

        blocks = [
            {"id": 1, "content": "Hello World"},
            {"id": 2, "content": "Foo Bar"},
        ]
        prompt = _build_translate_prompt(blocks, "日本語")
        assert "1" in prompt
        assert "Hello World" in prompt
        assert "2" in prompt
        assert "Foo Bar" in prompt

    def test_contains_json_array_output_instruction(self):
        """JSON配列形式の出力指示が含まれること。"""
        from main import _build_translate_prompt

        blocks = [{"id": 1, "content": "Test"}]
        prompt = _build_translate_prompt(blocks, "English")
        # JSON配列出力の指示（"["や"id"などのキーワードを確認）
        assert "[" in prompt
        assert "id" in prompt
        assert "translation" in prompt


# ---------- _parse_json_response ----------

class TestParseJsonResponse:
    """_parse_json_response 関数のテスト。"""

    def test_parses_plain_json(self):
        """通常JSONが正しくパースされること。"""
        from main import _parse_json_response

        text = '{"text_blocks": [{"id": 1}]}'
        result = _parse_json_response(text)
        assert result["text_blocks"][0]["id"] == 1

    def test_parses_code_fence_json(self):
        """コードフェンス付きJSONが正しくパースされること。"""
        from main import _parse_json_response

        text = "```\n{\"text_blocks\": []}\n```"
        result = _parse_json_response(text)
        assert "text_blocks" in result

    def test_parses_json_fence(self):
        """```json フェンス付きJSONが正しくパースされること。"""
        from main import _parse_json_response

        text = "```json\n[{\"id\": 1, \"translation\": \"テスト\"}]\n```"
        result = _parse_json_response(text)
        assert result[0]["id"] == 1
        assert result[0]["translation"] == "テスト"


# ---------- _handle_image_request ----------

class TestHandleImageRequest:
    """_handle_image_request 関数のテスト（vertex-claude / vertex-gemini）。"""

    @patch("main._call_vertex_claude")
    def test_image_request_vertex_claude(self, mock_claude):
        """vertex-claudeプロバイダでの画像リクエストが正しく処理されること。"""
        mock_claude.return_value = (
            {"text": '{"text_blocks": [{"id": 1, "type": "paragraph", "content": "Hello", "translation": "こんにちは", "bbox": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.1}, "confidence": 0.95}]}'},
            200,
        )
        img_data = base64.b64encode(b"fake-image").decode()
        from main import _handle_image_request
        result, status = _handle_image_request(
            {"base64Data": img_data, "mediaType": "image/png"},
            "日本語", 800, 600, "claude-sonnet-4-20250514", "vertex-claude"
        )
        assert status == 200
        assert "text_blocks" in result
        assert result["text_blocks"][0]["translation"] == "こんにちは"

    @patch("main._call_vertex_gemini")
    def test_image_request_vertex_gemini(self, mock_gemini):
        """vertex-geminiプロバイダでの画像リクエストが正しく処理されること。"""
        mock_gemini.return_value = (
            {"text": '{"text_blocks": [{"id": 1, "type": "paragraph", "content": "Hello", "translation": "こんにちは", "bbox": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.1}, "confidence": 0.9}]}'},
            200,
        )
        img_data = base64.b64encode(b"fake-image").decode()
        from main import _handle_image_request
        result, status = _handle_image_request(
            {"base64Data": img_data, "mediaType": "image/png"},
            "日本語", 800, 600, "gemini-2.5-flash", "vertex-gemini"
        )
        assert status == 200
        assert "text_blocks" in result
        assert result["text_blocks"][0]["translation"] == "こんにちは"


# ---------- _handle_text_translation_request ----------

class TestHandleTextTranslationRequest:
    """_handle_text_translation_request 関数のテスト。"""

    @patch("main._call_vertex_claude")
    def test_text_translation_vertex_claude(self, mock_claude):
        """vertex-claudeを使ったテキスト翻訳リクエストが正しく処理されること。"""
        mock_claude.return_value = (
            {"text": '[{"id": 1, "translation": "翻訳テスト"}]'},
            200,
        )
        from main import _handle_text_translation_request
        blocks = [{"id": 1, "content": "Hello", "bbox": {"x": 0, "y": 0, "w": 0.5, "h": 0.1}}]
        result, status = _handle_text_translation_request(blocks, "日本語", "claude-sonnet-4-20250514", "vertex-claude")
        assert status == 200
        assert result["text_blocks"][0]["translation"] == "翻訳テスト"
        assert result["text_blocks"][0]["bbox"] == {"x": 0, "y": 0, "w": 0.5, "h": 0.1}


# ---------- inference() エントリポイントのルーティング ----------

class TestInferenceEntrypoint:
    """inference() エントリポイントのルーティングテスト。"""

    def _make_request(self, body):
        request = MagicMock()
        request.method = "POST"
        request.get_json.return_value = body
        return request

    @patch("main._handle_image_request")
    def test_image_key_routes_to_handle_image_request(self, mock_handler):
        """imageキーを持つリクエストが_handle_image_requestにルーティングされること。"""
        from flask import Flask
        from main import inference

        mock_handler.return_value = ({"text_blocks": []}, 200)
        img_data = base64.b64encode(b"fake-image").decode()
        request = self._make_request({
            "provider": "vertex-claude",
            "model": "claude-sonnet-4-20250514",
            "image": {"base64Data": img_data, "mediaType": "image/png"},
            "targetLang": "日本語",
            "imageWidth": 800,
            "imageHeight": 600,
        })

        app = Flask(__name__)
        with app.app_context():
            inference(request)
        mock_handler.assert_called_once()

    @patch("main._handle_text_translation_request")
    def test_text_blocks_key_routes_to_handle_text_translation(self, mock_handler):
        """text_blocksキーを持つリクエストが_handle_text_translation_requestにルーティングされること。"""
        from flask import Flask
        from main import inference

        mock_handler.return_value = ({"text_blocks": []}, 200)
        request = self._make_request({
            "provider": "vertex-claude",
            "model": "claude-sonnet-4-20250514",
            "text_blocks": [{"id": 1, "content": "Hello"}],
            "targetLang": "日本語",
        })

        app = Flask(__name__)
        with app.app_context():
            inference(request)
        mock_handler.assert_called_once()

    def test_no_provider_returns_400(self):
        """providerなしのリクエストにHTTP 400が返ること。"""
        from flask import Flask
        from main import inference

        request = self._make_request({
            "model": "claude-sonnet-4-20250514",
            "image": {"base64Data": "abc", "mediaType": "image/png"},
        })

        app = Flask(__name__)
        with app.app_context():
            response = inference(request)
        assert response[1] == 400

    def test_no_image_or_text_blocks_returns_400(self):
        """imageもtext_blocksもないリクエストにHTTP 400が返ること。"""
        from flask import Flask
        from main import inference

        request = self._make_request({
            "provider": "vertex-claude",
            "model": "claude-sonnet-4-20250514",
            "targetLang": "日本語",
        })

        app = Flask(__name__)
        with app.app_context():
            response = inference(request)
        assert response[1] == 400
