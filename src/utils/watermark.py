"""
画像へのウォーターマーク追加機能

このモジュールでは、画像へのウォーターマーク追加機能を提供します。
"""

import os
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()


class Watermarker:
    """画像ウォーターマーククラス"""
    
    def __init__(self, font_path=None):
        """初期化メソッド
        
        Args:
            font_path (str or Path, optional): 使用するフォントファイルのパス。
                指定がない場合はデフォルトフォントを使用。
        """
        self.font_path = font_path
        self._initialize_font()
    
    def _initialize_font(self):
        """フォントを初期化する"""
        try:
            if self.font_path and os.path.exists(self.font_path):
                logger.info(f"カスタムフォントを使用: {self.font_path}")
            else:
                # デフォルトフォントパスを設定
                default_font_path = os.path.join(project_root, "assets", "fonts", "default.ttf")
                if os.path.exists(default_font_path):
                    self.font_path = default_font_path
                    logger.info(f"デフォルトフォントを使用: {default_font_path}")
                else:
                    logger.warning("フォントファイルが見つかりません。システムデフォルトフォントを使用します。")
                    self.font_path = None
        except Exception as e:
            logger.error(f"フォント初期化エラー: {e}")
            self.font_path = None
    
    def add_text_watermark(self, image, text, position="bottom-right", opacity=0.3, color=(255, 255, 255), font_size=24):
        """画像にテキストウォーターマークを追加する
        
        Args:
            image (PIL.Image): 入力画像
            text (str): ウォーターマークのテキスト
            position (str, optional): ウォーターマークの位置。デフォルトは"bottom-right"。
            opacity (float, optional): 不透明度（0.0〜1.0）。デフォルトは0.3。
            color (tuple, optional): テキストの色（RGB）。デフォルトは白。
            font_size (int, optional): フォントサイズ。デフォルトは24。
        
        Returns:
            PIL.Image: ウォーターマークを追加した画像
        """
        try:
            # 画像をコピーしてRGBAモードに変換
            img = image.copy()
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            
            # ウォーターマーク用の透明レイヤーを作成
            watermark = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)
            
            # フォントを設定
            font = self._get_font(font_size)
            
            # テキストサイズを取得
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # テキスト位置を計算
            pos = self._calculate_position(
                img.size[0], img.size[1],
                text_width, text_height,
                position
            )
            
            # テキストを描画
            draw.text(pos, text, font=font, fill=(*color, int(255 * opacity)))
            
            # 画像とウォーターマークを合成
            result = Image.alpha_composite(img, watermark)
            return result
            
        except Exception as e:
            logger.error(f"テキストウォーターマーク追加エラー: {e}")
            return image
    
    def add_image_watermark(self, image, watermark_image_path, position="bottom-right", opacity=0.3, scale=0.2):
        """画像にイメージウォーターマークを追加する
        
        Args:
            image (PIL.Image): 入力画像
            watermark_image_path (str or Path): ウォーターマーク画像のパス
            position (str, optional): ウォーターマークの位置。デフォルトは"bottom-right"。
            opacity (float, optional): 不透明度（0.0〜1.0）。デフォルトは0.3。
            scale (float, optional): ウォーターマークのスケール（元画像に対する比率）。デフォルトは0.2。
        
        Returns:
            PIL.Image: ウォーターマークを追加した画像
        """
        try:
            # 画像をコピーしてRGBAモードに変換
            img = image.copy()
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            
            # ウォーターマーク画像を読み込み
            watermark_img = Image.open(watermark_image_path)
            if watermark_img.mode != "RGBA":
                watermark_img = watermark_img.convert("RGBA")
            
            # ウォーターマークのサイズを調整
            wm_width = int(img.size[0] * scale)
            wm_height = int(wm_width * watermark_img.size[1] / watermark_img.size[0])
            watermark_img = watermark_img.resize((wm_width, wm_height))
            
            # 透明度を調整
            watermark_img.putalpha(int(255 * opacity))
            
            # ウォーターマークの位置を計算
            pos = self._calculate_position(
                img.size[0], img.size[1],
                wm_width, wm_height,
                position
            )
            
            # 新しい透明レイヤーを作成
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            layer.paste(watermark_img, (int(pos[0]), int(pos[1])))
            
            # 画像とウォーターマークを合成
            result = Image.alpha_composite(img, layer)
            return result
            
        except Exception as e:
            logger.error(f"イメージウォーターマーク追加エラー: {e}")
            return image
    
    def _calculate_position(self, img_width, img_height, wm_width, wm_height, position, padding=0):
        """ウォーターマークの位置を計算する
        
        Args:
            img_width (int): 元画像の幅
            img_height (int): 元画像の高さ
            wm_width (int): ウォーターマークの幅
            wm_height (int): ウォーターマークの高さ
            position (str): 位置指定（"top-left", "top-right", "bottom-left", "bottom-right", "center"）
            padding (int, optional): ウォーターマークと画像端との間隔。デフォルトは0。
        
        Returns:
            tuple: (x座標, y座標)
        """
        if position == "top-left":
            return (padding, padding)
        elif position == "top-right":
            return (img_width - wm_width - padding, padding)
        elif position == "bottom-left":
            return (padding, img_height - wm_height - padding)
        elif position == "bottom-right":
            return (img_width - wm_width - padding, img_height - wm_height - padding)
        else:  # center
            return ((img_width - wm_width) // 2, (img_height - wm_height) // 2)
    
    def _get_font(self, size):
        """指定されたサイズのフォントを取得する
        
        Args:
            size (int): フォントサイズ
        
        Returns:
            PIL.ImageFont: フォントオブジェクト
        """
        try:
            if self.font_path:
                return ImageFont.truetype(self.font_path, size)
            else:
                return ImageFont.load_default()
        except Exception as e:
            logger.warning(f"フォント読み込みエラー: {e}")
            return ImageFont.load_default()
    
    def add_watermark(self, image, text=None, watermark_image=None, position="bottom-right", opacity=0.3):
        """画像にウォーターマークを追加する（テキストまたは画像）
        
        Args:
            image (PIL.Image): 入力画像
            text (str, optional): テキストウォーターマーク
            watermark_image (str or Path, optional): 画像ウォーターマークのパス
            position (str, optional): ウォーターマークの位置。デフォルトは"bottom-right"。
            opacity (float, optional): 不透明度（0.0〜1.0）。デフォルトは0.3。
        
        Returns:
            PIL.Image: ウォーターマークを追加した画像
        """
        if text:
            return self.add_text_watermark(image, text, position, opacity)
        elif watermark_image:
            return self.add_image_watermark(image, watermark_image, position, opacity)
        else:
            logger.warning("テキストまたは画像ウォーターマークを指定してください")
            return image
    
    def batch_add_watermark(self, input_dir, output_dir=None, text=None, watermark_image=None, position="bottom-right", opacity=0.3):
        """ディレクトリ内の画像に一括でウォーターマークを追加する
        
        Args:
            input_dir (str or Path): 入力画像のディレクトリ
            output_dir (str or Path, optional): 出力先ディレクトリ。
                指定がない場合は入力ディレクトリに "_watermarked" を付加。
            text (str, optional): テキストウォーターマーク
            watermark_image (str or Path, optional): 画像ウォーターマークのパス
            position (str, optional): ウォーターマークの位置。デフォルトは"bottom-right"。
            opacity (float, optional): 不透明度（0.0〜1.0）。デフォルトは0.3。
        
        Returns:
            list: ウォーターマークを追加した画像のパスのリスト
        """
        # パスをPathオブジェクトに変換
        input_dir = Path(input_dir)
        
        # 入力ディレクトリが存在するか確認
        if not input_dir.exists():
            logger.error(f"入力ディレクトリが見つかりません: {input_dir}")
            return []
        
        # 出力ディレクトリが指定されていない場合は自動生成
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_watermarked"
        else:
            output_dir = Path(output_dir)
        
        # 出力ディレクトリを作成
        output_dir.mkdir(exist_ok=True)
        
        # ウォーターマークを追加した画像のパスを格納するリスト
        watermarked_paths = []
        
        # 画像ファイルを検索してウォーターマークを追加
        for file in input_dir.glob("*"):
            if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    # 画像を読み込み
                    with Image.open(file) as img:
                        # ウォーターマークを追加
                        watermarked = self.add_watermark(
                            img, text, watermark_image, position, opacity
                        )
                        
                        # 出力パスを生成
                        output_path = output_dir / f"{file.stem}_watermarked{file.suffix}"
                        
                        # 画像を保存
                        watermarked.save(output_path)
                        watermarked_paths.append(output_path)
                        
                except Exception as e:
                    logger.error(f"画像処理エラー: {file}: {e}")
        
        return watermarked_paths
    
    def get_supported_formats(self):
        """サポートしている画像フォーマットのリストを取得する
        
        Returns:
            list: サポートしている画像フォーマットの拡張子リスト
        """
        return [".png", ".jpg", ".jpeg"] 