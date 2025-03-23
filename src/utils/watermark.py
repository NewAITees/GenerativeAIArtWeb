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


class ImageWatermarker:
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
    
    def add_watermark(self, image_path, text, opacity=0.3, position="bottom-right", output_path=None):
        """画像にウォーターマークを追加する
        
        Args:
            image_path (str or Path): 入力画像のパス
            text (str): ウォーターマークのテキスト
            opacity (float, optional): 不透明度（0.0〜1.0）。デフォルトは0.3。
            position (str, optional): ウォーターマークの位置。
                "top-left", "top-right", "bottom-left", "bottom-right", "center"から選択。
                デフォルトは"bottom-right"。
            output_path (str or Path, optional): 出力画像のパス。
                指定がない場合は入力画像のパスに "_watermarked" を付加。
        
        Returns:
            Path: ウォーターマークを追加した画像のパス。失敗時はNone。
        """
        try:
            # パスをPathオブジェクトに変換
            image_path = Path(image_path)
            
            # 入力画像が存在するか確認
            if not image_path.exists():
                logger.error(f"入力画像が見つかりません: {image_path}")
                return None
            
            # 出力パスが指定されていない場合は自動生成
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_watermarked{image_path.suffix}"
            else:
                output_path = Path(output_path)
            
            # 画像を読み込み
            with Image.open(image_path) as img:
                # RGBAモードに変換
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                
                # ウォーターマーク用の透明レイヤーを作成
                watermark = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(watermark)
                
                # フォントサイズを画像サイズに応じて調整
                font_size = min(img.width, img.height) // 20
                try:
                    if self.font_path:
                        font = ImageFont.truetype(self.font_path, font_size)
                    else:
                        font = ImageFont.load_default()
                except Exception as e:
                    logger.warning(f"フォント読み込みエラー: {e}")
                    font = ImageFont.load_default()
                
                # テキストサイズを取得
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # テキスト位置を計算
                padding = 20
                if position == "top-left":
                    pos = (padding, padding)
                elif position == "top-right":
                    pos = (img.width - text_width - padding, padding)
                elif position == "bottom-left":
                    pos = (padding, img.height - text_height - padding)
                elif position == "bottom-right":
                    pos = (img.width - text_width - padding, img.height - text_height - padding)
                else:  # center
                    pos = ((img.width - text_width) // 2, (img.height - text_height) // 2)
                
                # テキストを描画
                draw.text(pos, text, font=font, fill=(255, 255, 255, int(255 * opacity)))
                
                # 画像とウォーターマークを合成
                result = Image.alpha_composite(img, watermark)
                
                # 画像を保存
                result.save(output_path)
            
            logger.info(f"ウォーターマークを追加しました: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"ウォーターマーク追加エラー: {e}")
            return None
    
    def batch_add_watermark(self, input_dir, text, opacity=0.3, position="bottom-right", output_dir=None):
        """ディレクトリ内の画像に一括でウォーターマークを追加する
        
        Args:
            input_dir (str or Path): 入力画像のディレクトリ
            text (str): ウォーターマークのテキスト
            opacity (float, optional): 不透明度（0.0〜1.0）。デフォルトは0.3。
            position (str, optional): ウォーターマークの位置。デフォルトは"bottom-right"。
            output_dir (str or Path, optional): 出力先ディレクトリ。
                指定がない場合は入力ディレクトリに "_watermarked" を付加。
        
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
                output_path = output_dir / f"{file.stem}_watermarked{file.suffix}"
                result = self.add_watermark(file, text, opacity, position, output_path)
                if result:
                    watermarked_paths.append(result)
        
        return watermarked_paths
    
    def get_supported_positions(self):
        """サポートしているウォーターマーク位置のリストを取得する
        
        Returns:
            list: サポートしているウォーターマーク位置のリスト
        """
        return ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
    
    def get_supported_formats(self):
        """サポートしている画像フォーマットのリストを取得する
        
        Returns:
            list: サポートしている画像フォーマットの拡張子リスト
        """
        return [".png", ".jpg", ".jpeg"] 