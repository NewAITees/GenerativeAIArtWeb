"""
画像のアップスケール機能

このモジュールでは、画像のアップスケール処理を提供します。
"""

import os
import logging
from pathlib import Path
from PIL import Image

# ロギング設定
logger = logging.getLogger(__name__)


class ImageUpscaler:
    """画像アップスケールクラス"""
    
    def __init__(self):
        """初期化メソッド"""
        pass
    
    def upscale(self, image_path, scale_factor=2.0, output_path=None):
        """画像をアップスケールする
        
        Args:
            image_path (str or Path): 入力画像のパス
            scale_factor (float, optional): アップスケール倍率。デフォルトは2.0倍。
            output_path (str or Path, optional): 出力画像のパス。
                指定がない場合は入力画像のパスに "_upscaled" を付加。
        
        Returns:
            Path: アップスケールした画像のパス。失敗時はNone。
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
                output_path = image_path.parent / f"{image_path.stem}_upscaled{image_path.suffix}"
            else:
                output_path = Path(output_path)
            
            # 画像を読み込み
            with Image.open(image_path) as img:
                # 新しいサイズを計算
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                # アップスケール処理
                upscaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 画像を保存
                upscaled_img.save(output_path)
            
            logger.info(f"画像をアップスケールしました: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"アップスケール処理エラー: {e}")
            return None
    
    def batch_upscale(self, input_dir, scale_factor=2.0, output_dir=None):
        """ディレクトリ内の画像を一括でアップスケールする
        
        Args:
            input_dir (str or Path): 入力画像のディレクトリ
            scale_factor (float, optional): アップスケール倍率。デフォルトは2.0倍。
            output_dir (str or Path, optional): 出力先ディレクトリ。
                指定がない場合は入力ディレクトリに "_upscaled" を付加。
        
        Returns:
            list: アップスケールした画像のパスのリスト
        """
        # パスをPathオブジェクトに変換
        input_dir = Path(input_dir)
        
        # 入力ディレクトリが存在するか確認
        if not input_dir.exists():
            logger.error(f"入力ディレクトリが見つかりません: {input_dir}")
            return []
        
        # 出力ディレクトリが指定されていない場合は自動生成
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_upscaled"
        else:
            output_dir = Path(output_dir)
        
        # 出力ディレクトリを作成
        output_dir.mkdir(exist_ok=True)
        
        # アップスケールした画像のパスを格納するリスト
        upscaled_paths = []
        
        # 画像ファイルを検索してアップスケール
        for file in input_dir.glob("*"):
            if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                output_path = output_dir / f"{file.stem}_upscaled{file.suffix}"
                result = self.upscale(file, scale_factor, output_path)
                if result:
                    upscaled_paths.append(result)
        
        return upscaled_paths
    
    def get_supported_formats(self):
        """サポートしている画像フォーマットのリストを取得する
        
        Returns:
            list: サポートしている画像フォーマットの拡張子リスト
        """
        return [".png", ".jpg", ".jpeg"] 