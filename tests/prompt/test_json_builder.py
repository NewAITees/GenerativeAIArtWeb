"""
JSONPromptBuilderのテストモジュール

このモジュールでは、JSONPromptBuilderクラスの機能を
ユニットテストで検証します。
"""

import os
import json
import pytest
from unittest.mock import mock_open, patch, MagicMock
from pathlib import Path
from src.prompt.json_builder import JSONPromptBuilder

# テスト用のサンプルテンプレート
SAMPLE_TEMPLATE = {
    "subjects": ["portrait", "landscape"],
    "styles": ["realistic", "fantasy"],
    "qualities": ["detailed", "high quality"],
    "lighting": ["natural lighting", "studio lighting"],
    "cameras": ["Canon EOS R5", "Nikon Z9"],
    "lenses": ["85mm f/1.2", "50mm f/1.4"]
}

class TestJSONPromptBuilder:
    """JSONPromptBuilderのテストクラス"""
    
    @pytest.fixture
    def mock_template_file(self):
        """テンプレートファイルのモックを作成"""
        with patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_TEMPLATE))):
            yield
    
    @pytest.fixture
    def builder(self, mock_template_file):
        """テスト用のJSONPromptBuilderインスタンスを作成"""
        return JSONPromptBuilder()
    
    def test_initialization(self, mock_template_file):
        """初期化とテンプレート読み込みのテスト"""
        builder = JSONPromptBuilder()
        assert builder.template == SAMPLE_TEMPLATE
        assert isinstance(builder.template, dict)
        assert len(builder.template) > 0
    
    def test_load_template_file_not_found(self):
        """テンプレートファイルが見つからない場合のテスト"""
        with patch("os.path.exists", return_value=False):
            builder = JSONPromptBuilder()
            # デフォルトテンプレートが設定されていることを確認
            assert "subjects" in builder.template
            assert "styles" in builder.template
    
    def test_build_prompt_empty(self, builder):
        """空の入力でのプロンプト構築テスト"""
        prompt = builder.build_prompt()
        assert prompt == ""
    
    def test_build_prompt_single_element(self, builder):
        """単一要素でのプロンプト構築テスト"""
        prompt = builder.build_prompt(subject="portrait")
        assert prompt == "portrait"
    
    def test_build_prompt_multiple_elements(self, builder):
        """複数要素でのプロンプト構築テスト"""
        prompt = builder.build_prompt(
            subject="portrait",
            style="realistic",
            quality="detailed"
        )
        assert prompt == "realistic portrait, detailed"
    
    def test_build_prompt_all_elements(self, builder):
        """全要素でのプロンプト構築テスト"""
        prompt = builder.build_prompt(
            subject="portrait",
            style="realistic",
            quality="detailed",
            lighting="natural lighting",
            camera="Canon EOS R5",
            lens="85mm f/1.2"
        )
        expected = "realistic portrait, detailed, natural lighting, Canon EOS R5, 85mm f/1.2"
        assert prompt == expected
    
    @patch("random.choice")
    def test_random_prompt(self, mock_choice, builder):
        """ランダムプロンプト生成のテスト"""
        mock_choice.side_effect = lambda x: x[0]  # 常に最初の要素を選択
        prompt = builder.random_prompt()
        expected = "portrait, realistic, detailed, natural lighting, Canon EOS R5, 85mm f/1.2"
        assert prompt == expected
    
    @patch("random.choice")
    def test_random_prompt_with_categories(self, mock_choice, builder):
        """特定カテゴリでのランダムプロンプト生成テスト"""
        mock_choice.side_effect = lambda x: x[0]
        prompt = builder.random_prompt(include_categories=["subjects", "styles"])
        assert prompt == "portrait, realistic"
    
    def test_build_prompt_with_weights(self, builder):
        """重み付きプロンプト構築のテスト"""
        elements = {
            "subject": "portrait",
            "style": "realistic",
            "quality": "detailed"
        }
        weights = {
            "subject": 1.2,
            "style": 0.8,
            "quality": 1.0
        }
        prompt = builder.build_prompt_with_weights(elements, weights)
        assert prompt == "(portrait:1.2), (realistic:0.8), (detailed:1.0)"
    
    def test_save_template(self, builder):
        """テンプレート保存機能のテスト"""
        test_path = "test_template.json"
        mock_open_obj = mock_open()
        
        with patch("builtins.open", mock_open_obj):
            result = builder.save_template(test_path)
            assert result is True
            
            # ファイルが開かれ、json.dumpが呼ばれたことを確認
            mock_open_obj.assert_called_once_with(test_path, "w")
            handle = mock_open_obj()
            assert handle.write.called
    
    def test_save_template_error(self, builder):
        """テンプレート保存エラーのテスト"""
        with patch("builtins.open", side_effect=IOError):
            result = builder.save_template("invalid/path/template.json")
            assert result is False 