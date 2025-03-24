"""
LLMを使用したプロンプト生成機能のテスト

このモジュールでは、LLMPromptGeneratorクラスの機能をテストします。
主な検証項目：
- 初期化処理
- プロンプト生成機能
- エラー処理
- モックレスポンス
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.prompt.llm_generator import LLMPromptGenerator


class TestLLMPromptGenerator:
    """LLMPromptGeneratorクラスのテスト"""
    
    def test_initialization(self):
        """初期化が正しく行われるかテスト"""
        generator = LLMPromptGenerator(model_name="llama3")
        assert generator.model_name == "llama3"
        assert generator.timeout == 30
        assert generator.host is None
    
    @patch('src.prompt.llm_generator.ollama')
    def test_initialize_client(self, mock_ollama):
        """ollamaクライアントの初期化処理をテスト"""
        # モックのセットアップ
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            'models': [{'name': 'llama3'}, {'name': 'llama2'}]
        }
        
        # 初期化（コンストラクタで_initialize_clientが呼ばれる）
        generator = LLMPromptGenerator(model_name="llama3")
        
        # 検証
        assert generator.initialized is True
        assert generator.client == mock_client
        assert mock_ollama.Client.call_count == 1
    
    @patch('src.prompt.llm_generator.ollama')
    def test_initialize_client_model_not_found(self, mock_ollama):
        """指定したモデルが見つからない場合のテスト"""
        # モックのセットアップ
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            'models': [{'name': 'llama2'}]  # llama3が含まれていない
        }
        
        # 初期化
        generator = LLMPromptGenerator(model_name="llama3")
        
        # 検証
        assert generator.initialized is False
    
    @patch('src.prompt.llm_generator.ollama')
    def test_initialize_client_exception(self, mock_ollama):
        """初期化中に例外が発生した場合のテスト"""
        # モックのセットアップ
        mock_ollama.Client.side_effect = Exception("Connection error")
        
        # 初期化
        generator = LLMPromptGenerator(model_name="llama3")
        
        # 検証
        assert generator.initialized is False
    
    @patch('src.prompt.llm_generator.ollama')
    def test_generate_prompt(self, mock_ollama):
        """プロンプト生成機能のテスト"""
        # モックのセットアップ
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            'models': [{'name': 'llama3'}]
        }
        mock_client.generate.return_value = {
            'response': 'A photorealistic image of a cat with detailed fur, sitting in a sunlit garden, 8k resolution',
            'done': True
        }
        
        # 初期化とプロンプト生成
        generator = LLMPromptGenerator(model_name="llama3")
        result = generator.generate_prompt("cat")
        
        # 検証
        assert "photorealistic" in result
        assert "cat" in result
        assert "detailed" in result
        mock_client.generate.assert_called_once()
    
    @patch('src.prompt.llm_generator.ollama')
    def test_generate_prompt_with_style(self, mock_ollama):
        """スタイル指定ありのプロンプト生成テスト"""
        # モックのセットアップ
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            'models': [{'name': 'llama3'}]
        }
        mock_client.generate.return_value = {
            'response': 'An anime style illustration of a cat with expressive eyes, vibrant colors, 2D style',
            'done': True
        }
        
        # 初期化とプロンプト生成
        generator = LLMPromptGenerator(model_name="llama3")
        result = generator.generate_prompt("cat", style="anime")
        
        # 検証
        assert "anime" in result
        assert "cat" in result
        mock_client.generate.assert_called_once()
    
    def test_mock_generate_when_not_initialized(self):
        """初期化されていない場合のモックレスポンステスト"""
        generator = LLMPromptGenerator(model_name="llama3")
        generator.initialized = False  # 強制的に未初期化状態にする
        
        result = generator.generate_prompt("cat")
        
        # モックレスポンスが返されることを確認
        assert "cat" in result
        assert "photorealistic" in result or "detailed" in result
    
    def test_batch_enhance(self):
        """複数プロンプトの一括強化テスト"""
        # generate_promptをモック
        with patch.object(LLMPromptGenerator, 'generate_prompt') as mock_generate:
            mock_generate.side_effect = [
                "A stunning photo of a cat",
                "A beautiful landscape of mountains"
            ]
            
            generator = LLMPromptGenerator(model_name="llama3")
            results = generator.batch_enhance(["cat", "mountains"])
            
            # 検証
            assert len(results) == 2
            assert "cat" in results[0]
            assert "mountains" in results[1]
            assert mock_generate.call_count == 2
    
    def test_save_and_load_prompt(self, tmp_path):
        """プロンプトの保存と読み込みテスト"""
        # テスト用データ
        test_data = {
            "prompt": "A beautiful cat",
            "style": "photorealistic",
            "settings": {"quality": "high"}
        }
        
        # 一時ファイルパスの作成
        file_path = tmp_path / "test_prompt.json"
        
        # LLMPromptGeneratorのインスタンス化
        generator = LLMPromptGenerator()
        
        # 保存テスト
        save_result = generator.save_prompt(test_data, str(file_path))
        assert save_result is True
        assert file_path.exists()
        
        # 読み込みテスト
        loaded_data = generator.load_prompt(str(file_path))
        assert loaded_data == test_data
        assert loaded_data["prompt"] == "A beautiful cat"
        assert loaded_data["style"] == "photorealistic"
        assert loaded_data["settings"]["quality"] == "high" 