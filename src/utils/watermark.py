"""
ウォーターマーク機能

このモジュールでは、生成された画像にウォーターマークを追加する機能を提供します。
"""

import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()


class WatermarkProcessor:
    """ウォーターマーク処理クラス"""
    
    def __init__(self, font_path=None):
        """初期化メソッド
        
        Args:
            font_path (str, optional): 使用するフォントのパス。
                指定がない場合はデフォルトフォントを使用。
        """
        # デフォルトのフォントパス（環境に依存）
        self.font_path = font_path
        self.default_font_size = 24
        
        # フォント読み込み試行（存在する場合）
        self._initialize_font()
    
    def _initialize_font(self):
        """フォントを初期化する"""
        self.font = None
        
        try:
            if self.font_path and os.path.exists(self.font_path):
                self.font = ImageFont.truetype(self.font_path, self.default_font_size)
                logger.info(f"カスタムフォントを読み込みました: {self.font_path}")
        except Exception as e:
            logger.warning(f"カスタムフォント読み込みエラー: {e}")
            self.font = None
    
    def add_watermark(self, image, text="Generated with SD3.5", position="bottom-right", opacity=0.5):
        """画像にウォーターマークを追加する
        
        Args:
            image: PIL.Imageまたはnumpy.ndarray形式の画像
            text (str, optional): ウォーターマークテキスト。デフォルトは "Generated with SD3.5"
            position (str, optional): ウォーターマークの位置。
                "top-left", "top-right", "bottom-left", "bottom-right" のいずれか。
                デフォルトは "bottom-right"
            opacity (float, optional): ウォーターマークの不透明度（0.0〜1.0）。デフォルトは0.5
        
        Returns:
            PIL.Image: ウォーターマークが追加された画像
        """
        if image is None:
            logger.warning("ウォーターマークを追加する画像がNoneです")
            return None
        
        # NumPy配列の場合はPILイメージに変換
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 画像オブジェクトでない場合はそのまま返す
        if not isinstance(image, Image.Image):
            logger.warning(f"サポートされていない画像タイプです: {type(image)}")
            return image
        
        try:
            # 元の画像をコピー
            watermarked = image.copy()
            
            # アルファチャンネルがない場合は追加
            if watermarked.mode != 'RGBA':
                watermarked = watermarked.convert('RGBA')
            
            # ウォーターマーク用のオーバーレイ画像
            overlay = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # フォントサイズを画像サイズに応じて調整
            font_size = max(int(min(watermarked.width, watermarked.height) * 0.03), 12)
            
            # フォントの取得（カスタムフォントがない場合はデフォルトフォント）
            if self.font:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                # デフォルトフォント（サイズのみ指定）
                font = ImageFont.load_default()
            
            # テキストサイズの取得
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # テキスト位置の計算
            padding = int(font_size * 0.5)  # パディング
            
            if position == "top-left":
                text_position = (padding, padding)
            elif position == "top-right":
                text_position = (watermarked.width - text_width - padding, padding)
            elif position == "bottom-left":
                text_position = (padding, watermarked.height - text_height - padding)
            else:  # "bottom-right" または他の値の場合
                text_position = (watermarked.width - text_width - padding, 
                                 watermarked.height - text_height - padding)
            
            # テキストの描画（半透明の背景付き）
            bg_padding = int(font_size * 0.2)
            bg_opacity = int(opacity * 128)  # 背景の不透明度（テキストの半分）
            
            # 背景の矩形
            bg_bbox = (
                text_position[0] - bg_padding,
                text_position[1] - bg_padding,
                text_position[0] + text_width + bg_padding,
                text_position[1] + text_height + bg_padding
            )
            
            # 背景を描画
            draw.rectangle(bg_bbox, fill=(0, 0, 0, bg_opacity))
            
            # テキストを描画
            draw.text(text_position, text, font=font, fill=(255, 255, 255, int(opacity * 255)))
            
            # オーバーレイを元画像と合成
            watermarked = Image.alpha_composite(watermarked, overlay)
            
            # 元の画像形式に戻す
            if image.mode != 'RGBA':
                watermarked = watermarked.convert(image.mode)
            
            return watermarked
        except Exception as e:
            logger.error(f"ウォーターマーク追加エラー: {e}")
            return image
    
    def get_available_positions(self):
        """利用可能なウォーターマーク位置の一覧を取得する
        
        Returns:
            list: 利用可能なウォーターマーク位置のリスト
        """
        return ["top-left", "top-right", "bottom-left", "bottom-right"] 