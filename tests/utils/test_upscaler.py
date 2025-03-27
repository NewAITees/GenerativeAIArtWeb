"""
アップスケール機能のテスト

このモジュールでは、Upscalerクラスの機能をテストします。
主な検証項目：
- 画像のサイズ計算
- アップスケール処理
- バッチアップスケール処理
- エラー処理
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

# テスト対象のモジュール
from src.utils.upscaler import Upscaler

class TestUpscaler:
    """Upscalerクラスのテスト"""
    
    @pytest.fixture
    def upscaler(self):
        """テスト用のUpscalerインスタンスを準備"""
        return Upscaler()
    
    @pytest.fixture
    def mock_image(self):
        """モック画像の作成"""
        # PIL.Imageをモック
        mock_img = MagicMock(spec=Image.Image)
        mock_img.width = 512
        mock_img.height = 512
        mock_img.save.return_value = None
        return mock_img
    
    def test_initialization(self, upscaler):
        """初期化が正しく行われるかテスト"""
        assert upscaler is not None
        assert isinstance(upscaler, Upscaler)
    
    def test_get_optimal_dimensions(self, upscaler):
        """最適なサイズ計算をテスト"""
        # 基本的なケース
        width, height = upscaler.get_optimal_dimensions(512, 512, 2.0)
        assert width == 1024, f"Expected width to be 1024, got {width}"
        assert height == 1024, f"Expected height to be 1024, got {height}"
        
        # 小数スケールのケース
        width, height = upscaler.get_optimal_dimensions(512, 512, 1.5)
        assert width == 768, f"Expected width to be 768, got {width}"
        assert height == 768, f"Expected height to be 768, got {height}"
        
        # 非正方形画像のケース
        width, height = upscaler.get_optimal_dimensions(800, 600, 2.0)
        assert width == 1600, f"Expected width to be 1600, got {width}"
        assert height == 1200, f"Expected height to be 1200, got {height}"
    
    @patch('src.utils.upscaler.Image')
    def test_upscale_success(self, mock_image_module, upscaler, mock_image, tmp_path):
        """アップスケール処理の成功ケースをテスト"""
        # モックの設定
        mock_image_module.open.return_value.__enter__.return_value = mock_image
        mock_image.resize.return_value = mock_image
        
        # Image.Resampling.LANCZOSのモックを設定
        mock_resampling = MagicMock()
        mock_resampling.LANCZOS = Image.Resampling.LANCZOS
        mock_image_module.Resampling = mock_resampling
        
        # テストファイルのパス設定
        input_path = tmp_path / "test_input.png"
        output_path = tmp_path / "test_output.png"
        input_path.touch()
        
        # パスオブジェクトのモック
        with patch('src.utils.upscaler.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.parent = tmp_path
            mock_path.return_value.stem = "test_input"
            mock_path.return_value.suffix = ".png"
            
            # アップスケール実行
            result = upscaler.upscale(str(input_path), 2.0, str(output_path))
            
            # 検証
            assert result is not None
            mock_image.resize.assert_called_once_with((1024, 1024), mock_resampling.LANCZOS)
            mock_image.save.assert_called_once()
    
    def test_upscale_file_not_found(self, upscaler):
        """存在しないファイルのアップスケールテスト"""
        with patch('src.utils.upscaler.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = upscaler.upscale("nonexistent.png", 2.0)
            assert result is None
    
    def test_upscale_exception(self, upscaler):
        """アップスケール中の例外処理テスト"""
        with patch('src.utils.upscaler.Image') as mock_image_module:
            # Image.openで例外が発生するようにモック設定
            mock_image_module.open.side_effect = Exception("Test error")
            
            result = upscaler.upscale("test.png", 2.0)
            assert result is None
    
    @patch('src.utils.upscaler.Path')
    def test_batch_upscale_success(self, mock_path, upscaler, mock_image):
        """バッチアップスケール処理の成功ケースをテスト"""
        # モックの設定
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.mkdir.return_value = None
        mock_path.return_value.glob.return_value = [
            Path("test1.png"),
            Path("test2.jpg"),
            Path("test3.jpeg"),
            Path("test4.txt")  # 非画像ファイル
        ]
        
        # upscaleメソッドをモック
        with patch.object(upscaler, 'upscale') as mock_upscale:
            mock_upscale.side_effect = [
                Path("output1.png"),
                Path("output2.jpg"),
                Path("output3.jpeg"),
                None  # test4.txtは処理されない
            ]
            
            # バッチアップスケール実行
            results = upscaler.batch_upscale("input_dir", 2.0, "output_dir")
            
            # 検証
            assert len(results) == 3  # 画像ファイルのみ処理される
            assert mock_upscale.call_count == 3
            assert any(str(r).endswith("output1.png") for r in results)
            assert any(str(r).endswith("output2.jpg") for r in results)
            assert any(str(r).endswith("output3.jpeg") for r in results)
    
    def test_batch_upscale_directory_not_found(self, upscaler):
        """存在しないディレクトリのバッチアップスケールテスト"""
        with patch('src.utils.upscaler.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            results = upscaler.batch_upscale("nonexistent_dir", 2.0)
            assert len(results) == 0
    
    def test_get_supported_formats(self, upscaler):
        """サポートフォーマットの取得テスト"""
        formats = upscaler.get_supported_formats()
        assert isinstance(formats, list)
        assert ".png" in formats
        assert ".jpg" in formats
        assert ".jpeg" in formats
        assert len(formats) == 3  # サポートされているフォーマット数を確認 