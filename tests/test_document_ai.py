"""Tests for Document AI integration in bastet-inference."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------- _extract_image_from_messages ----------

class TestExtractImageFromMessages:
    def test_extracts_first_image(self):
        from main import _extract_image_from_messages

        img_data = base64.b64encode(b"fake-image-bytes").decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "mediaType": "image/png", "base64Data": img_data},
            ],
        }]
        image_bytes, mime_type = _extract_image_from_messages(messages)
        assert image_bytes == b"fake-image-bytes"
        assert mime_type == "image/png"

    def test_raises_when_no_image(self):
        from main import _extract_image_from_messages

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }]
        with pytest.raises(ValueError, match="No image found"):
            _extract_image_from_messages(messages)

    def test_extracts_from_multiple_blocks(self):
        from main import _extract_image_from_messages

        img_data = base64.b64encode(b"image-data").decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "ignored"},
                {"type": "image", "mediaType": "image/jpeg", "base64Data": img_data},
            ],
        }]
        image_bytes, mime_type = _extract_image_from_messages(messages)
        assert image_bytes == b"image-data"
        assert mime_type == "image/jpeg"


# ---------- _build_text_blocks_from_ocr ----------

class TestBuildTextBlocksFromOcr:
    def _make_document(self, paragraphs):
        """Build a mock Document AI Document with given paragraph specs."""
        doc = MagicMock()
        doc.text = ""

        mock_paragraphs = []
        offset = 0
        for text, vertices, confidence in paragraphs:
            doc.text += text
            seg = MagicMock()
            seg.start_index = offset
            seg.end_index = offset + len(text)
            offset += len(text)

            layout = MagicMock()
            layout.text_anchor.text_segments = [seg]
            layout.confidence = confidence

            mock_vertices = []
            for x, y in vertices:
                v = MagicMock()
                v.x = x
                v.y = y
                mock_vertices.append(v)
            layout.bounding_poly.normalized_vertices = mock_vertices

            para = MagicMock()
            para.layout = layout
            mock_paragraphs.append(para)

        page = MagicMock()
        page.paragraphs = mock_paragraphs
        doc.pages = [page]
        return doc

    def test_basic_conversion(self):
        from main import _build_text_blocks_from_ocr

        doc = self._make_document([
            ("Hello World", [(0.1, 0.2), (0.9, 0.2), (0.9, 0.3), (0.1, 0.3)], 0.95),
        ])
        blocks = _build_text_blocks_from_ocr(doc)

        assert len(blocks) == 1
        assert blocks[0]["id"] == 1
        assert blocks[0]["content"] == "Hello World"
        assert blocks[0]["bbox"]["x"] == pytest.approx(0.1, abs=1e-5)
        assert blocks[0]["bbox"]["y"] == pytest.approx(0.2, abs=1e-5)
        assert blocks[0]["bbox"]["w"] == pytest.approx(0.8, abs=1e-5)
        assert blocks[0]["bbox"]["h"] == pytest.approx(0.1, abs=1e-5)
        assert blocks[0]["confidence"] == pytest.approx(0.95, abs=1e-3)

    def test_skips_empty_text(self):
        from main import _build_text_blocks_from_ocr

        doc = self._make_document([
            ("   ", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 0.5),
        ])
        blocks = _build_text_blocks_from_ocr(doc)
        assert len(blocks) == 0

    def test_multiple_paragraphs_sequential_ids(self):
        from main import _build_text_blocks_from_ocr

        doc = self._make_document([
            ("First", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.1), (0.0, 0.1)], 0.9),
            ("Second", [(0.0, 0.2), (0.5, 0.2), (0.5, 0.3), (0.0, 0.3)], 0.85),
        ])
        blocks = _build_text_blocks_from_ocr(doc)

        assert len(blocks) == 2
        assert blocks[0]["id"] == 1
        assert blocks[1]["id"] == 2
        assert blocks[0]["content"] == "First"
        assert blocks[1]["content"] == "Second"


# ---------- _translate_blocks ----------

class TestTranslateBlocks:
    @patch("main._call_vertex_claude")
    def test_translates_with_claude(self, mock_claude):
        from main import _translate_blocks

        mock_claude.return_value = (
            {"text": json.dumps({"results": [
                {"id": 1, "type": "title", "translation": "翻訳テスト"},
            ]})},
            200,
        )

        blocks = [{"id": 1, "content": "Test text", "bbox": {"x": 0, "y": 0, "w": 1, "h": 0.1}, "confidence": 0.9}]
        result = _translate_blocks(blocks, "claude-sonnet-4-20250514", "日本語")

        assert result[0]["type"] == "title"
        assert result[0]["translation"] == "翻訳テスト"
        mock_claude.assert_called_once()

    @patch("main._call_vertex_gemini")
    def test_translates_with_gemini(self, mock_gemini):
        from main import _translate_blocks

        mock_gemini.return_value = (
            {"text": json.dumps({"results": [
                {"id": 1, "type": "paragraph", "translation": "Translated"},
            ]})},
            200,
        )

        blocks = [{"id": 1, "content": "テスト", "bbox": {"x": 0, "y": 0, "w": 1, "h": 0.1}, "confidence": 0.9}]
        result = _translate_blocks(blocks, "gemini-2.0-flash", "English")

        assert result[0]["translation"] == "Translated"
        mock_gemini.assert_called_once()

    @patch("main._call_vertex_claude")
    def test_fallback_on_llm_error(self, mock_claude):
        from main import _translate_blocks

        mock_claude.return_value = ({"error": "Server error"}, 500)

        blocks = [{"id": 1, "content": "Original", "bbox": {"x": 0, "y": 0, "w": 1, "h": 0.1}, "confidence": 0.9}]
        result = _translate_blocks(blocks, "claude-sonnet-4-20250514", "日本語")

        # Should fall back to original text
        assert result[0]["type"] == "paragraph"
        assert result[0]["translation"] == "Original"

    def test_empty_blocks(self):
        from main import _translate_blocks

        result = _translate_blocks([], "claude-sonnet-4-20250514", "日本語")
        assert result == []


# ---------- _call_document_ai (integration) ----------

class TestCallDocumentAi:
    @patch("main._translate_blocks")
    @patch("main._build_text_blocks_from_ocr")
    @patch("main._ocr_with_document_ai")
    def test_full_flow_success(self, mock_ocr, mock_build, mock_translate):
        from main import _call_document_ai

        mock_ocr.return_value = MagicMock()
        mock_build.return_value = [
            {"id": 1, "content": "Hello", "bbox": {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.1}, "confidence": 0.95},
        ]
        mock_translate.return_value = [
            {"id": 1, "content": "Hello", "type": "paragraph", "translation": "こんにちは",
             "bbox": {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.1}, "confidence": 0.95},
        ]

        img_data = base64.b64encode(b"fake-image").decode()
        messages = [{"role": "user", "content": [
            {"type": "image", "mediaType": "image/png", "base64Data": img_data},
        ]}]

        result, status = _call_document_ai(messages, "claude-sonnet-4-20250514", "日本語")

        assert status == 200
        parsed = json.loads(result["text"])
        assert len(parsed["text_blocks"]) == 1
        assert parsed["text_blocks"][0]["translation"] == "こんにちは"
        assert result["stop_reason"] == "end_turn"

    def test_no_image_returns_400(self):
        from main import _call_document_ai

        messages = [{"role": "user", "content": [{"type": "text", "text": "no image"}]}]
        result, status = _call_document_ai(messages, "claude-sonnet-4-20250514", "日本語")

        assert status == 400
        assert "error" in result

    @patch("main._ocr_with_document_ai")
    def test_empty_ocr_result(self, mock_ocr):
        from main import _call_document_ai

        mock_doc = MagicMock()
        mock_doc.pages = [MagicMock()]
        mock_doc.pages[0].paragraphs = []
        mock_ocr.return_value = mock_doc

        img_data = base64.b64encode(b"fake-image").decode()
        messages = [{"role": "user", "content": [
            {"type": "image", "mediaType": "image/png", "base64Data": img_data},
        ]}]

        # Need to also patch _build_text_blocks_from_ocr to return empty
        with patch("main._build_text_blocks_from_ocr", return_value=[]):
            result, status = _call_document_ai(messages, "claude-sonnet-4-20250514", "日本語")

        assert status == 200
        parsed = json.loads(result["text"])
        assert parsed["text_blocks"] == []


# ---------- Entry point dispatch ----------

class TestEntryDispatchesDocumentAi:
    @patch("main._call_document_ai")
    def test_provider_document_ai_dispatches(self, mock_docai):
        from main import inference

        mock_docai.return_value = (
            {"text": '{"text_blocks": []}', "stop_reason": "end_turn"},
            200,
        )

        img_data = base64.b64encode(b"fake-image").decode()
        request = MagicMock()
        request.method = "POST"
        request.get_json.return_value = {
            "provider": "document-ai",
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": [
                {"type": "image", "mediaType": "image/png", "base64Data": img_data},
            ]}],
            "target_lang": "English",
        }

        response = inference(request)
        mock_docai.assert_called_once_with(
            request.get_json.return_value["messages"],
            "claude-sonnet-4-20250514",
            "English",
        )
