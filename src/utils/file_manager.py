"""
ファイル管理機能

このモジュールでは、生成された画像のカスタム保存や管理機能を提供します。
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()


class FileManager:
    """画像ファイル管理クラス"""
    
    def __init__(self, base_directory=None):
        """初期化メソッド
        
        Args:
            base_directory (str or Path, optional): 保存の基準ディレクトリ。
                指定がない場合はプロジェクトルートのoutputsディレクトリを使用。
        """
        self.base_directory = Path(base_directory) if base_directory else project_root / "outputs"
        
        # 基本ディレクトリが存在しない場合は作成
        self.base_directory.mkdir(exist_ok=True)
        
        # サブディレクトリの作成
        self._initialize_directories()
    
    def _initialize_directories(self):
        """基本ディレクトリ構造を初期化する"""
        # デフォルトのサブディレクトリ
        subdirs = ["general", "portraits", "landscapes", "abstracts"]
        
        for subdir in subdirs:
            dir_path = self.base_directory / subdir
            dir_path.mkdir(exist_ok=True)
    
    def save_image(self, image, directory=None, filename_pattern=None, metadata=None):
        """画像を指定したディレクトリとファイル名パターンで保存する
        
        Args:
            image: PIL.Imageまたはnumpy.ndarray形式の画像
            directory (str, optional): 保存先ディレクトリ。基本ディレクトリからの相対パス。
                指定がなければ基本ディレクトリに保存。
            filename_pattern (str, optional): ファイル名のパターン。
                以下のプレースホルダーを使用可能:
                - {date}: 現在の日付
                - {time}: 現在の時刻
                - {prompt}: メタデータ内のプロンプト
                - {seed}: メタデータ内のシード値
                デフォルトは "{date}_{time}" 形式。
            metadata (dict, optional): 保存する画像のメタデータ。
                ファイル名パターンにも使用されます。
        
        Returns:
            str: 保存されたファイルの絶対パス
        """
        if image is None:
            logger.warning("保存する画像がNoneです")
            return None
        
        # NumPy配列の場合はPILイメージに変換
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 画像オブジェクトでない場合はエラー
        if not isinstance(image, Image.Image):
            logger.warning(f"サポートされていない画像タイプです: {type(image)}")
            return None
        
        try:
            # 保存先ディレクトリの準備
            if directory:
                save_dir = self.base_directory / directory
            else:
                save_dir = self.base_directory
            
            # ディレクトリが存在しない場合は作成
            save_dir.mkdir(exist_ok=True)
            
            # メタデータの準備（デフォルト値）
            if metadata is None:
                metadata = {}
            
            # 現在の日時
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H%M%S")
            
            # ファイル名パターンの処理
            if not filename_pattern:
                filename_pattern = "{date}_{time}"
            
            # パターン内のプレースホルダーを置換
            filename = filename_pattern
            filename = filename.replace("{date}", date_str)
            filename = filename.replace("{time}", time_str)
            
            # メタデータからのプレースホルダー置換
            for key, value in metadata.items():
                placeholder = f"{{{key}}}"
                if placeholder in filename:
                    # 特殊文字を置換してファイル名に適した形式に
                    safe_value = self._sanitize_filename(str(value))
                    filename = filename.replace(placeholder, safe_value[:50])  # 長さ制限
            
            # ファイル名から無効な文字を除去
            filename = self._sanitize_filename(filename)
            
            # 拡張子を追加
            if not filename.lower().endswith(".png"):
                filename += ".png"
            
            # ファイルパスの作成
            file_path = save_dir / filename
            
            # ファイル名の衝突を回避（同名ファイルが存在する場合）
            counter = 1
            while file_path.exists():
                name_parts = filename.rsplit(".", 1)
                if len(name_parts) > 1:
                    new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    new_filename = f"{filename}_{counter}"
                file_path = save_dir / new_filename
                counter += 1
            
            # 画像の保存
            image.save(file_path)
            
            # メタデータをサイドカーファイルとして保存
            metadata_path = file_path.with_suffix(".json")
            self._save_metadata(metadata_path, metadata)
            
            logger.info(f"画像を保存しました: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"画像保存エラー: {e}")
            
            # エラー時はフォールバックとして基本ディレクトリに保存
            try:
                fallback_path = self.base_directory / f"error_{os.urandom(4).hex()}.png"
                image.save(fallback_path)
                return str(fallback_path)
            except:
                return None
    
    def _sanitize_filename(self, filename):
        """ファイル名から無効な文字を除去する
        
        Args:
            filename (str): 処理するファイル名
        
        Returns:
            str: 安全な形式のファイル名
        """
        # 無効な文字を置換
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
        # 先頭と末尾の空白や記号を削除
        sanitized = sanitized.strip(" ._-")
        # 長すぎる場合は切り詰める
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        # 空文字列になった場合はデフォルト名
        if not sanitized:
            sanitized = "image"
        return sanitized
    
    def _save_metadata(self, metadata_path, metadata):
        """メタデータをJSONファイルとして保存する
        
        Args:
            metadata_path (Path): メタデータファイルのパス
            metadata (dict): 保存するメタデータ
        """
        try:
            # タイムスタンプを追加
            metadata["saved_at"] = datetime.now().isoformat()
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")
    
    def get_directories(self):
        """利用可能な保存ディレクトリの一覧を取得する
        
        Returns:
            list: 利用可能なディレクトリのリスト（基本ディレクトリからの相対パス）
        """
        try:
            # 基本ディレクトリが存在しない場合は作成
            self.base_directory.mkdir(exist_ok=True)
            
            # サブディレクトリを取得
            dirs = [""]  # 空文字はルートディレクトリを表す
            
            for item in self.base_directory.iterdir():
                if item.is_dir():
                    # 基本ディレクトリからの相対パス
                    dirs.append(item.name)
            
            return sorted(dirs)
        except Exception as e:
            logger.error(f"ディレクトリ一覧取得エラー: {e}")
            return [""]  # エラー時はルートディレクトリのみ
    
    def create_directory(self, directory_name):
        """新しいサブディレクトリを作成する
        
        Args:
            directory_name (str): 作成するディレクトリ名
        
        Returns:
            bool: 作成成功時はTrue、失敗時はFalse
        """
        try:
            # ディレクトリ名のサニタイズ
            safe_name = self._sanitize_filename(directory_name)
            
            # 空文字列になった場合は失敗とする
            if not safe_name:
                logger.warning(f"無効なディレクトリ名です: {directory_name}")
                return False
            
            # ディレクトリの作成
            dir_path = self.base_directory / safe_name
            dir_path.mkdir(exist_ok=True)
            
            logger.info(f"ディレクトリを作成しました: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"ディレクトリ作成エラー: {e}")
            return False 