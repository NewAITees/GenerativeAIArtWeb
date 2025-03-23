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
from typing import Optional, Dict, Any, List, Union

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()

class FileManager:
    """画像ファイル管理クラス"""
    
    # 画像ファイルの拡張子セット（クラス変数として定義）
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    
    def __init__(self, output_dir: Optional[str] = None):
        """初期化メソッド
        
        Args:
            output_dir (str or Path, optional): 保存の基準ディレクトリ。
                指定がない場合はプロジェクトルートのoutputsディレクトリを使用。
        """
        self.output_dir = str(output_dir) if output_dir else str(project_root / "outputs")
        
        # 基本ディレクトリが存在しない場合は作成
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # サブディレクトリの作成
        self._initialize_directories()
    
    def _initialize_directories(self):
        """基本ディレクトリ構造を初期化する"""
        # デフォルトのサブディレクトリ
        subdirs = ["general", "portraits", "landscapes", "abstracts"]
        
        for subdir in subdirs:
            dir_path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
    
    def save_image(self, image: Any, filename_prefix: str,
                  directory: Optional[str] = None) -> Optional[str]:
        """画像を指定したディレクトリとファイル名で保存する
        
        Args:
            image: PIL.Imageまたはnumpy.ndarray形式の画像
            filename_prefix (str): ファイル名の接頭辞
            directory (str, optional): 保存先ディレクトリ。基本ディレクトリからの相対パス。
                指定がなければ基本ディレクトリに保存。
        
        Returns:
            str: 保存されたファイルの絶対パス
        """
        if image is None:
            logger.warning("保存する画像がNoneです")
            return None
        
        try:
            # 保存先ディレクトリの準備
            save_dir = os.path.join(self.output_dir, directory) if directory else self.output_dir
            
            # ディレクトリが存在しない場合は作成
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # ファイル名を生成
            filename = self.generate_filename(filename_prefix)
            
            # ファイルパスの作成
            file_path = os.path.join(save_dir, filename)
            
            # ファイル名の衝突を回避（同名ファイルが存在する場合）
            counter = 1
            while os.path.exists(file_path):
                name_parts = filename.rsplit(".", 1)
                if len(name_parts) > 1:
                    new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    new_filename = f"{filename}_{counter}"
                file_path = os.path.join(save_dir, new_filename)
                counter += 1
            
            # 画像の保存
            if hasattr(image, 'save'):
                image.save(file_path)
            else:
                logger.warning(f"画像オブジェクトにsaveメソッドがありません: {type(image)}")
                return None
            
            logger.info(f"画像を保存しました: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"画像保存エラー: {e}")
            return None
    
    def generate_filename(self, prompt: str, prefix: str = "",
                        include_date: bool = True, include_time: bool = True,
                        extension: str = "png") -> str:
        """ファイル名を生成する
        
        Args:
            prompt (str): プロンプトテキスト
            prefix (str, optional): ファイル名の接頭辞
            include_date (bool, optional): 日付を含めるかどうか
            include_time (bool, optional): 時刻を含めるかどうか
            extension (str, optional): ファイル拡張子
        
        Returns:
            str: 生成されたファイル名
        """
        # プロンプトの短縮と正規化
        short_prompt = self._sanitize_filename(prompt)[:30]
        
        # 現在の日時
        now = datetime.now()
        date_str = now.strftime("%Y%m%d") if include_date else ""
        time_str = now.strftime("%H%M%S") if include_time else ""
        
        # 各部分を組み合わせてファイル名を作成
        parts = []
        if prefix:
            parts.append(prefix)
        if short_prompt:
            parts.append(short_prompt)
        if date_str:
            parts.append(date_str)
        if time_str:
            parts.append(time_str)
        
        # 拡張子が指定されている場合は先頭のドットを削除
        if extension.startswith('.'):
            extension = extension[1:]
        
        # ファイル名の構築
        return "_".join(parts) + f".{extension}"
    
    def _sanitize_filename(self, filename: str) -> str:
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
        # 空文字列になった場合はデフォルト名
        return sanitized if sanitized else "image"
    
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
            os.makedirs(self.output_dir, exist_ok=True)
            
            # サブディレクトリを取得
            dirs = [""]  # 空文字はルートディレクトリを表す
            
            for item in os.listdir(self.output_dir):
                if os.path.isdir(os.path.join(self.output_dir, item)):
                    # 基本ディレクトリからの相対パス
                    dirs.append(item)
            
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
            dir_path = os.path.join(self.output_dir, safe_name)
            os.makedirs(dir_path, exist_ok=True)
            
            logger.info(f"ディレクトリを作成しました: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"ディレクトリ作成エラー: {e}")
            return False
    
    def organize_by_folder(self, folder_name: str, filename: str,
                         folder_type: str = "prompt") -> str:
        """画像を指定のフォルダに整理する
        
        Args:
            folder_name (str): フォルダ名
            filename (str): ファイル名
            folder_type (str, optional): フォルダのタイプ（"prompt", "date", "custom"など）
        
        Returns:
            str: 整理後のファイルパス
        """
        try:
            # フォルダ名のサニタイズ
            safe_folder = self._sanitize_filename(folder_name)
            
            # フォルダパスの作成
            folder_path = os.path.join(self.output_dir, folder_type, safe_folder)
            
            # フォルダの作成
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            
            # 新しいファイルパス
            return os.path.join(folder_path, filename)
        except Exception as e:
            logger.error(f"フォルダ整理エラー: {e}")
            return os.path.join(self.output_dir, filename)
    
    def save_image_with_metadata(self, image: Any, filename_prefix: str,
                               metadata: Dict[str, Any]) -> Optional[str]:
        """画像とメタデータを保存する
        
        Args:
            image: PIL.Imageまたはnumpy.ndarray形式の画像
            filename_prefix (str): ファイル名の接頭辞
            metadata (dict): 保存するメタデータ
        
        Returns:
            str: 保存された画像ファイルのパス
        """
        # 画像の保存
        image_path = self.save_image(image, filename_prefix)
        if not image_path:
            return None
        
        try:
            # メタデータファイルのパスを生成
            metadata_path = os.path.splitext(image_path)[0] + ".json"
            
            # メタデータの保存
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"メタデータを保存しました: {metadata_path}")
            return image_path
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")
            return image_path
    
    def list_generated_images(self, directory: Optional[str] = None) -> List[str]:
        """生成された画像ファイルの一覧を取得する
        
        Args:
            directory (str, optional): 検索対象のディレクトリ
        
        Returns:
            list: 画像ファイル名のリスト
        """
        try:
            # 検索対象ディレクトリの設定
            target_dir = os.path.join(self.output_dir, directory) if directory else self.output_dir
            
            # 画像ファイルのリストを作成
            image_files = []
            for filename in os.listdir(target_dir):
                if os.path.isfile(os.path.join(target_dir, filename)):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in self.IMAGE_EXTENSIONS:
                        image_files.append(filename)
            
            return sorted(image_files)
        except Exception as e:
            logger.error(f"画像一覧取得エラー: {e}")
            return []
    
    def batch_rename(self, prefix: Optional[str] = None,
                    remove_pattern: Optional[str] = None,
                    directory: Optional[str] = None) -> List[str]:
        """ファイルの一括リネーム
        
        Args:
            prefix (str, optional): 新しい接頭辞
            remove_pattern (str, optional): 削除するパターン
            directory (str, optional): 対象ディレクトリ
        
        Returns:
            list: 新しいファイル名のリスト
        """
        try:
            # 対象ディレクトリの設定
            target_dir = os.path.join(self.output_dir, directory) if directory else self.output_dir
            
            # ディレクトリが存在しない場合は空リストを返す
            if not os.path.exists(target_dir):
                logger.warning(f"ディレクトリが存在しません: {target_dir}")
                return []
            
            # 画像ファイルのリストを取得
            image_files = self.list_generated_images(directory)
            new_names = []
            
            for filename in image_files:
                old_path = os.path.join(target_dir, filename)
                
                # 新しいファイル名を生成
                new_filename = filename
                
                # パターンを削除
                if remove_pattern:
                    new_filename = new_filename.replace(remove_pattern, "")
                
                # 接頭辞を追加
                if prefix:
                    name_parts = new_filename.rsplit(".", 1)
                    if len(name_parts) > 1:
                        new_filename = f"{prefix}{name_parts[0]}.{name_parts[1]}"
                    else:
                        new_filename = f"{prefix}{new_filename}"
                
                # 新しいパスを作成
                new_path = os.path.join(target_dir, new_filename)
                
                # ファイル名の衝突を回避
                counter = 1
                while os.path.exists(new_path) and new_path != old_path:
                    name_parts = new_filename.rsplit(".", 1)
                    if len(name_parts) > 1:
                        new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        new_filename = f"{new_filename}_{counter}"
                    new_path = os.path.join(target_dir, new_filename)
                    counter += 1
                
                # ファイルをリネーム
                os.rename(old_path, new_path)
                new_names.append(new_filename)
            
            return new_names
        except Exception as e:
            logger.error(f"一括リネームエラー: {e}")
            return []
    
    def load_image_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """画像のメタデータを読み込む
        
        Args:
            image_path (str): 画像ファイルのパス
        
        Returns:
            dict: メタデータ辞書
        """
        try:
            # メタデータファイルのパスを生成
            metadata_path = os.path.splitext(image_path)[0] + ".json"
            
            # メタデータファイルが存在しない場合はNoneを返す
            if not os.path.exists(metadata_path):
                logger.warning(f"メタデータファイルが見つかりません: {metadata_path}")
                return None
            
            # メタデータの読み込み
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"メタデータ読み込みエラー: {e}")
            return None 