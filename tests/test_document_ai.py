"""bastet-inferenceのDocument AI統合に関するテスト群。

main.pyの各内部関数およびエントリポイントを単体テストで検証する。
外部依存（Vertex AI、Document AI API）はすべてMockで代替する。
"""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------- _extract_image_from_messages ----------

class TestExtractImageFromMessages:
    """_extract_image_from_messages 関数のテスト。

    NormalizedMessage[] から画像ブロックを正しく抽出できることを検証する。
    """

    def test_extracts_first_image(self):
        """単一の画像ブロックからバイト列とMIMEタイプを正しく取り出せること。"""
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
        """画像ブロックが存在しない場合にValueErrorを送出すること。"""
        from main import _extract_image_from_messages

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }]
        with pytest.raises(ValueError, match="No image found"):
            _extract_image_from_messages(messages)

    def test_extracts_from_multiple_blocks(self):
        """テキストブロックと画像ブロックが混在する場合に最初の画像を取り出せること。"""
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
    """_build_text_blocks_from_ocr 関数のテスト。

    Document AIが返すDocumentオブジェクト（Mock）をtext_blocks形式に
    正しく変換できることを検証する。
    """

    def _make_document(self, paragraphs):
        """テスト用のDocument AIドキュメントMockを生成するヘルパー。

        paragraphsは (テキスト, [(x, y), ...], 信頼度) のリスト。
        document.textには各テキストを連結し、text_anchorで参照できるようにする。
        """
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

            # バウンディングボックスの頂点をMockで構築
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
        """1段落を正しくtext_block形式に変換できること（id, content, bbox, confidenceを検証）。"""
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
        """空白のみの段落はスキップされ、結果リストに含まれないこと。"""
        from main import _build_text_blocks_from_ocr

        doc = self._make_document([
            ("   ", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 0.5),
        ])
        blocks = _build_text_blocks_from_ocr(doc)
        assert len(blocks) == 0

    def test_multiple_paragraphs_sequential_ids(self):
        """複数の段落が連番IDで順番に変換されること。"""
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
    """_translate_blocks 関数のテスト。

    LLMによるブロック翻訳とタイプ分類が正しく行われること、
    およびエラー時のフォールバック動作を検証する。
    """

    @patch("main._call_vertex_claude")
    def test_translates_with_claude(self, mock_claude):
        """Claudeモデルを使って翻訳・タイプ分類が正しく実行されること。

        モックが一度だけ呼ばれ、翻訳結果とタイプがブロックに反映されることを確認する。
        """
        from main import _translate_blocks

        # モックがClaudeの正常レスポンスを返すよう設定
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
        """Geminiモデルを使って翻訳が正しく実行されること。

        モデル名が"gemini"で始まる場合にGeminiハンドラが選択されることを確認する。
        """
        from main import _translate_blocks

        # モックがGeminiの正常レスポンスを返すよう設定
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
        """LLMエラー時に元テキストをtranslationとして返すフォールバック動作。

        HTTP 500エラーが返った場合、翻訳なしで元コンテンツをそのまま返すことを確認する。
        """
        from main import _translate_blocks

        # モックがエラーレスポンスを返すよう設定
        mock_claude.return_value = ({"error": "Server error"}, 500)

        blocks = [{"id": 1, "content": "Original", "bbox": {"x": 0, "y": 0, "w": 1, "h": 0.1}, "confidence": 0.9}]
        result = _translate_blocks(blocks, "claude-sonnet-4-20250514", "日本語")

        # 元テキストがそのままtranslationにフォールバックされることを検証
        assert result[0]["type"] == "paragraph"
        assert result[0]["translation"] == "Original"

    def test_empty_blocks(self):
        """空のブロックリストを渡した場合は空リストをそのまま返すこと。"""
        from main import _translate_blocks

        result = _translate_blocks([], "claude-sonnet-4-20250514", "日本語")
        assert result == []


# ---------- _call_document_ai (統合テスト) ----------

class TestCallDocumentAi:
    """_call_document_ai 関数の統合テスト。

    OCR → テキストブロック変換 → LLM翻訳の全フローが正しく動作することを検証する。
    各ステップはモックで代替し、連携と最終レスポンス形式を確認する。
    """

    @patch("main._translate_blocks")
    @patch("main._build_text_blocks_from_ocr")
    @patch("main._ocr_with_document_ai")
    def test_full_flow_success(self, mock_ocr, mock_build, mock_translate):
        """OCR → 変換 → 翻訳の正常系フロー全体が正しく動作すること。

        最終レスポンスがClaude Vision互換のtext_blocks形式で返ること、
        翻訳結果が正しくマージされていることを確認する。
        """
        from main import _call_document_ai

        # 各ステップのモックを正常応答に設定
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
        """画像なしのリクエストに対してHTTP 400エラーが返ること。"""
        from main import _call_document_ai

        messages = [{"role": "user", "content": [{"type": "text", "text": "no image"}]}]
        result, status = _call_document_ai(messages, "claude-sonnet-4-20250514", "日本語")

        assert status == 400
        assert "error" in result

    @patch("main._ocr_with_document_ai")
    def test_empty_ocr_result(self, mock_ocr):
        """OCR結果が空の場合にtext_blocksが空リストで返ること。

        段落が存在しないドキュメントを処理したとき、
        HTTP 200かつ空のtext_blocksが返ることを確認する。
        """
        from main import _call_document_ai

        # 段落なしのドキュメントMockを設定
        mock_doc = MagicMock()
        mock_doc.pages = [MagicMock()]
        mock_doc.pages[0].paragraphs = []
        mock_ocr.return_value = mock_doc

        img_data = base64.b64encode(b"fake-image").decode()
        messages = [{"role": "user", "content": [
            {"type": "image", "mediaType": "image/png", "base64Data": img_data},
        ]}]

        # _build_text_blocks_from_ocrも空を返すようパッチ
        with patch("main._build_text_blocks_from_ocr", return_value=[]):
            result, status = _call_document_ai(messages, "claude-sonnet-4-20250514", "日本語")

        assert status == 200
        parsed = json.loads(result["text"])
        assert parsed["text_blocks"] == []


# ---------- エントリポイントのルーティング ----------

class TestEntryDispatchesDocumentAi:
    """inferenceエントリポイントのdocument-aiプロバイダルーティングテスト。

    document-aiプロバイダが指定されたとき、_handle_image_requestに正しくディスパッチされることを検証する。
    """

    @patch("main._handle_image_request")
    def test_image_request_document_ai(self, mock_handler):
        """provider="document-ai"のリクエストが_handle_image_requestに正しく転送されること。

        新形式リクエスト（imageキー）が正しくハンドラに渡されることを確認する。
        """
        from flask import Flask
        from main import inference

        app = Flask(__name__)

        # モックが成功レスポンスを返すよう設定
        mock_handler.return_value = ({"text_blocks": []}, 200)

        img_data = base64.b64encode(b"fake-image").decode()
        request = MagicMock()
        request.method = "POST"
        request.get_json.return_value = {
            "provider": "document-ai",
            "model": "claude-sonnet-4-20250514",
            "image": {"base64Data": img_data, "mediaType": "image/png"},
            "targetLang": "English",
            "imageWidth": 800,
            "imageHeight": 600,
        }

        with app.app_context():
            response = inference(request)
        # _handle_image_requestが正しい引数で呼ばれたことを検証
        mock_handler.assert_called_once_with(
            {"base64Data": img_data, "mediaType": "image/png"},
            "English",
            800,
            600,
            "claude-sonnet-4-20250514",
            "document-ai",
        )
