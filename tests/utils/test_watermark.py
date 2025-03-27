"""
ウォーターマーク機能のテスト

このモジュールでは、Watermarkerクラスの機能をテストします。
主な検証項目：
- テキストウォーターマークの追加
- 画像ウォーターマークの追加
- ウォーターマークの位置計算
- フォント処理
- エラー処理
- バッチ処理
- エッジケース（極端なサイズの画像）
- 実画像を使用した統合テスト

テスト実行方法:
```bash
# 単体テストの実行
python -m pytest tests/utils/test_watermark.py -v

# カバレッジレポート付きでテスト実行
python -m pytest tests/utils/test_watermark.py --cov=src.utils.watermark
```

注意事項:
- PILライブラリのバージョンによって挙動が異なる場合があります
- フォントの読み込みはシステムに依存する場合があります
- 画像の読み込みにはテスト用の小さな画像ファイルを使用します
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw, ImageFont

# テスト対象のモジュール
from src.utils.watermark import Watermarker

@pytest.fixture
def mock_image():
    """モック画像を作成するフィクスチャ"""
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGB"
    mock_img.size = (1024, 1024)
    mock_img.readonly = False
    mock_img.copy.return_value = mock_img
    mock_img.convert.return_value = mock_img
    return mock_img

class TestWatermarker:
    """Watermarkerクラスのテスト"""
    
    @pytest.fixture
    def watermarker(self):
        """テスト用のWatermarkerインスタンスを準備"""
        with patch('src.utils.watermark.os.path.exists', return_value=False):
            # フォントファイルが存在しない状態でテスト
            return Watermarker()
    
    @pytest.fixture
    def watermarker_with_font(self):
        """フォント付きのWatermarkerインスタンスを準備"""
        with patch('src.utils.watermark.os.path.exists', return_value=True):
            with patch('src.utils.watermark.ImageFont.truetype') as mock_truetype:
                mock_truetype.return_value = MagicMock(spec=ImageFont.FreeTypeFont)
                return Watermarker(font_path="dummy/font.ttf")
    
    def test_initialization(self, watermarker):
        """初期化が正しく行われるかテスト"""
        assert watermarker is not None
        assert watermarker.font_path is None  # デフォルトフォントが使用される
    
    def test_initialization_with_font(self):
        """フォントパス指定時の初期化テスト"""
        with patch('src.utils.watermark.os.path.exists', return_value=True):
            custom_font_path = "custom/font.ttf"
            watermarker = Watermarker(font_path=custom_font_path)
            assert watermarker.font_path == custom_font_path
    
    def test_get_font(self, watermarker):
        """フォント取得メソッドのテスト"""
        with patch('src.utils.watermark.ImageFont.load_default') as mock_default:
            mock_default.return_value = MagicMock(spec=ImageFont.FreeTypeFont)
            font = watermarker._get_font(24)
            mock_default.assert_called_once()
            assert font == mock_default.return_value
    
    def test_get_font_with_custom_font(self, watermarker_with_font):
        """カスタムフォント使用時のテスト"""
        with patch('src.utils.watermark.ImageFont.truetype') as mock_truetype:
            mock_font = MagicMock(spec=ImageFont.FreeTypeFont)
            mock_truetype.return_value = mock_font
            
            font = watermarker_with_font._get_font(24)
            
            mock_truetype.assert_called_once()
            assert font == mock_font
    
    def test_get_font_with_error(self, watermarker_with_font):
        """フォント読み込みエラーのテスト"""
        with patch('src.utils.watermark.ImageFont.truetype', side_effect=Exception("Font error")), \
             patch('src.utils.watermark.ImageFont.load_default') as mock_default:
            mock_default.return_value = MagicMock(spec=ImageFont.FreeTypeFont)
            
            font = watermarker_with_font._get_font(24)
            
            mock_default.assert_called_once()
            assert font == mock_default.return_value
    
    def test_calculate_position(self, watermarker):
        """位置計算メソッドのテスト"""
        # テスト用パラメータ
        img_width, img_height = 1000, 800
        wm_width, wm_height = 200, 100
        padding = 10
        
        # 各位置のテスト
        positions = {
            "top-left": (padding, padding),
            "top-right": (img_width - wm_width - padding, padding),
            "bottom-left": (padding, img_height - wm_height - padding),
            "bottom-right": (img_width - wm_width - padding, img_height - wm_height - padding),
            "center": ((img_width - wm_width) // 2, (img_height - wm_height) // 2)
        }
        
        for position, expected in positions.items():
            result = watermarker._calculate_position(
                img_width, img_height, wm_width, wm_height, position, padding
            )
            assert result == expected, f"Position {position} calculation failed"
    
    def test_add_text_watermark(self, watermarker, mock_image):
        """テキストウォーターマーク追加のテスト"""
        with patch('PIL.Image.alpha_composite', return_value=mock_image) as mock_alpha:
            # 関数の呼び出し
            result = watermarker.add_text_watermark(
                mock_image,
                "Test",
                position="center",
                opacity=0.5
            )

            # 検証
            assert result is not None
            mock_image.convert.assert_called_with("RGBA")
            mock_alpha.assert_called_once()
    
    def test_add_text_watermark_with_exception(self, watermarker, mock_image):
        """テキストウォーターマーク追加時のエラー処理テスト"""
        with patch('src.utils.watermark.ImageDraw.Draw', side_effect=Exception("Draw error")):
            # 関数の呼び出し
            result = watermarker.add_text_watermark(mock_image, "Test Watermark")
            
            # 例外時には元の画像が返される
            assert result == mock_image
    
    def test_add_image_watermark(self, watermarker, mock_image):
        """画像ウォーターマーク追加のテスト"""
        mock_watermark_img = MagicMock(spec=Image.Image)
        mock_watermark_img.mode = "RGBA"
        mock_watermark_img.size = (200, 100)
        mock_watermark_img.resize.return_value = mock_watermark_img
        mock_watermark_img.putalpha.return_value = None
        
        mock_layer = MagicMock(spec=Image.Image)
        mock_layer.paste.return_value = None
        
        with patch('PIL.Image.open', return_value=mock_watermark_img) as mock_open, \
             patch('PIL.Image.new', return_value=mock_layer) as mock_new, \
             patch('PIL.Image.alpha_composite', return_value=mock_image) as mock_alpha:
            
            # 関数の呼び出し
            result = watermarker.add_image_watermark(
                mock_image,
                "watermark.png",
                position="center",
                opacity=0.7,
                scale=0.2
            )
            
            # 検証
            assert result is not None
            mock_image.convert.assert_called_with("RGBA")
            mock_watermark_img.resize.assert_called_once()
            mock_watermark_img.putalpha.assert_called_once()
            mock_layer.paste.assert_called_once()
            mock_alpha.assert_called_once()
    
    def test_add_image_watermark_with_exception(self, watermarker, mock_image):
        """画像ウォーターマーク追加時のエラー処理テスト"""
        with patch('src.utils.watermark.Image.open', side_effect=Exception("Open error")):
            # 関数の呼び出し
            result = watermarker.add_image_watermark(mock_image, "watermark.png")
            
            # 例外時には元の画像が返される
            assert result == mock_image
    
    def test_add_watermark_text(self, watermarker, mock_image):
        """ウォーターマーク追加ラッパー（テキスト）のテスト"""
        # add_text_watermarkメソッドをモック
        with patch.object(watermarker, 'add_text_watermark') as mock_text_wm:
            mock_text_wm.return_value = mock_image
            
            # 関数の呼び出し
            result = watermarker.add_watermark(
                mock_image, 
                text="Test Watermark"
            )
            
            # 検証
            mock_text_wm.assert_called_once()
            assert result == mock_image
    
    def test_add_watermark_image(self, watermarker, mock_image):
        """ウォーターマーク追加ラッパー（画像）のテスト"""
        # add_image_watermarkメソッドをモック
        with patch.object(watermarker, 'add_image_watermark') as mock_image_wm:
            mock_image_wm.return_value = mock_image
            
            # 関数の呼び出し
            result = watermarker.add_watermark(
                mock_image, 
                watermark_image="watermark.png"
            )
            
            # 検証
            mock_image_wm.assert_called_once()
            assert result == mock_image
    
    def test_add_watermark_no_params(self, watermarker, mock_image):
        """ウォーターマーク追加（パラメータなし）のテスト"""
        # 関数の呼び出し
        result = watermarker.add_watermark(mock_image)
        
        # パラメータがない場合は元の画像が返される
        assert result == mock_image
    
    @patch('src.utils.watermark.Path')
    @patch('src.utils.watermark.Image.open')
    def test_batch_add_watermark(self, mock_open, mock_path, watermarker, mock_image, tmp_path):
        """バッチウォーターマーク追加のテスト"""
        # モックの設定
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.parent = tmp_path
        mock_path.return_value.glob.return_value = [
            Path("image1.png"),
            Path("image2.jpg"),
            Path("notanimage.txt")
        ]
        
        # 画像オープンモックの設定
        mock_open.return_value.__enter__.return_value = mock_image
        
        # add_watermarkメソッドをモック
        with patch.object(watermarker, 'add_watermark') as mock_add_watermark:
            mock_add_watermark.return_value = mock_image
            
            # 関数の呼び出し
            result = watermarker.batch_add_watermark(
                str(tmp_path / "input"),
                str(tmp_path / "output"),
                text="Batch Test"
            )
            
            # 検証
            assert len(result) == 2  # 画像ファイルのみ処理される
            assert mock_add_watermark.call_count == 2
    
    def test_batch_add_watermark_directory_not_found(self, watermarker):
        """存在しないディレクトリのバッチ処理テスト"""
        with patch('src.utils.watermark.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = watermarker.batch_add_watermark("nonexistent_dir")
            assert len(result) == 0
    
    def test_get_supported_formats(self, watermarker):
        """サポートフォーマットの取得テスト"""
        formats = watermarker.get_supported_formats()
        assert isinstance(formats, list)
        assert ".png" in formats
        assert ".jpg" in formats
        assert ".jpeg" in formats
        assert len(formats) == 3  # サポートされているフォーマット数を確認
    
    @pytest.mark.parametrize("image_mode", ["RGB", "RGBA", "L"])
    def test_add_text_watermark_different_modes(self, watermarker, image_mode):
        """異なる画像モードでのテキストウォーターマーク追加テスト"""
        mock_img = MagicMock(spec=Image.Image)
        mock_img.mode = image_mode
        mock_img.size = (1024, 1024)
        mock_img.readonly = False
        mock_img.copy.return_value = mock_img
        mock_img.convert.return_value = mock_img

        with patch('PIL.Image.alpha_composite', return_value=mock_img) as mock_alpha:
            # 関数の呼び出し
            result = watermarker.add_text_watermark(mock_img, "Test")

            # 検証
            assert result is not None
            if image_mode != "RGBA":
                mock_img.convert.assert_called_with("RGBA")
            mock_alpha.assert_called_once()
    
    @pytest.mark.parametrize("image_size", [
        (1, 1),           # 最小サイズ
        (10000, 10000),   # 巨大サイズ
        (800, 1),         # 極端な縦横比
        (1, 600),         # 極端な縦横比
        (512, 512)        # 標準サイズ
    ])
    def test_add_text_watermark_edge_cases(self, watermarker, image_size):
        """極端なサイズの画像に対するテキストウォーターマークのテスト"""
        mock_img = MagicMock(spec=Image.Image)
        mock_img.mode = "RGB"
        mock_img.size = image_size
        mock_img.readonly = False
        mock_img.copy.return_value = mock_img
        mock_img.convert.return_value = mock_img

        with patch('PIL.Image.alpha_composite', return_value=mock_img) as mock_alpha:
            # ウォーターマーク追加
            result = watermarker.add_text_watermark(
                mock_img,
                "Test",
                position="center",
                opacity=0.5
            )

            # 検証
            assert result is not None
            mock_img.convert.assert_called_with("RGBA")
            mock_alpha.assert_called_once()

    @pytest.mark.integration
    def test_integration_with_real_image(self, watermarker, tmp_path):
        """実際の画像を使用した統合テスト"""
        # テスト用の小さな画像を作成
        test_image = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
        test_image_path = tmp_path / "test_image.png"
        test_image.save(test_image_path)

        # テスト用のウォーターマーク画像を作成
        watermark_image = Image.new('RGBA', (30, 30), (0, 0, 0, 128))
        watermark_path = tmp_path / "watermark.png"
        watermark_image.save(watermark_path)

        try:
            # 画像を読み込み
            with Image.open(test_image_path) as img:
                # テキストウォーターマークを追加
                result1 = watermarker.add_text_watermark(
                    img,
                    "Test",
                    position="center",
                    opacity=0.5
                )
                assert isinstance(result1, Image.Image)
                assert result1.size == (100, 100)

                # 画像ウォーターマークを追加
                result2 = watermarker.add_image_watermark(
                    img,
                    str(watermark_path),
                    position="bottom-right",
                    opacity=0.7
                )
                assert isinstance(result2, Image.Image)
                assert result2.size == (100, 100)

        finally:
            # クリーンアップ
            if test_image_path.exists():
                test_image_path.unlink()
            if watermark_path.exists():
                watermark_path.unlink()

    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_batch_add_watermark_performance(self, watermarker, tmp_path, batch_size):
        """バッチ処理のパフォーマンステスト"""
        # 入力ディレクトリの作成
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        try:
            # テスト用の画像を作成
            test_images = []
            for i in range(batch_size):
                img = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
                path = input_dir / f"test_image_{i}.png"
                img.save(path)
                test_images.append(path)

            # バッチ処理を実行
            results = watermarker.batch_add_watermark(
                str(input_dir),
                str(output_dir),
                text="Batch Test",
                position="center",
                opacity=0.5
            )

            # 検証
            assert len(results) == batch_size
            for result in results:
                assert Path(result).exists()
                assert Path(result).parent == output_dir

        finally:
            # クリーンアップ
            for path in test_images:
                if path.exists():
                    path.unlink()
            
            # 出力ファイルの削除
            for file in output_dir.glob("*"):
                file.unlink()
            
            if input_dir.exists():
                input_dir.rmdir()
            if output_dir.exists():
                output_dir.rmdir() 