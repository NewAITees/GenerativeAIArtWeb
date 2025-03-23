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
    
    def __init__(self, output_dir=None):
        """初期化メソッド
        
        Args:
            output_dir (str or Path, optional): 保存の基準ディレクトリ。
                指定がない場合はプロジェクトルートのoutputsディレクトリを使用。
        """
        self.output_dir = Path(output_dir) if output_dir else project_root / "outputs"
        
        # 基本ディレクトリが存在しない場合は作成
        self.output_dir.mkdir(exist_ok=True)
        
        # サブディレクトリの作成
        self._initialize_directories()
    
    def _initialize_directories(self):
        """基本ディレクトリ構造を初期化する"""
        # デフォルトのサブディレクトリ
        subdirs = ["general", "portraits", "landscapes", "abstracts"]
        
        for subdir in subdirs:
            dir_path = self.output_dir / subdir
            dir_path.mkdir(exist_ok=True)
    
    def save_image(self, image, filename_prefix, directory=None):
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
                save_dir = self.output_dir / directory
            else:
                save_dir = self.output_dir
            
            # ディレクトリが存在しない場合は作成
            save_dir.mkdir(exist_ok=True)
            
            # ファイル名を生成
            filename = self.generate_filename(filename_prefix)
            
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
            
            logger.info(f"画像を保存しました: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"画像保存エラー: {e}")
            return None
    
    def generate_filename(self, prompt, prefix="", include_date=True, include_time=True, extension="png"):
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
        short_prompt = self._sanitize_filename(prompt)
        if len(short_prompt) > 30:
            short_prompt = short_prompt[:30]
        
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
        filename = "_".join(parts) + f".{extension}"
        
        return filename
    
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
            self.output_dir.mkdir(exist_ok=True)
            
            # サブディレクトリを取得
            dirs = [""]  # 空文字はルートディレクトリを表す
            
            for item in self.output_dir.iterdir():
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
            dir_path = self.output_dir / safe_name
            dir_path.mkdir(exist_ok=True)
            
            logger.info(f"ディレクトリを作成しました: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"ディレクトリ作成エラー: {e}")
            return False
    
    def organize_by_folder(self, folder_name, filename, folder_type="prompt"):
        """画像を指定のフォルダに整理する
        
        Args:
            folder_name (str): フォルダ名
            filename (str): ファイル名
            folder_type (str, optional): フォルダのタイプ（"prompt", "date", "custom"など）
        
        Returns:
            str: 整理後のファイルパス
        """
        # フォルダ名を正規化
        safe_folder_name = self._sanitize_filename(folder_name)
        
        # フォルダタイプに基づいて保存先を決定
        if folder_type == "prompt":
            target_dir = self.output_dir / "by_prompt" / safe_folder_name
        elif folder_type == "date":
            target_dir = self.output_dir / "by_date" / safe_folder_name
        else:  # custom
            target_dir = self.output_dir / safe_folder_name
        
        # ディレクトリが存在しない場合は作成
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名から絶対パスを取得
        if os.path.isabs(filename):
            src_path = Path(filename)
        else:
            src_path = self.output_dir / filename
        
        # ファイル名のみを取得
        file_basename = os.path.basename(filename)
        
        # 移動先のパスを作成
        dest_path = target_dir / file_basename
        
        return str(dest_path)
    
    def save_image_with_metadata(self, image, filename_prefix, metadata):
        """画像とメタデータを保存する
        
        Args:
            image: PIL.Imageまたはnumpy.ndarray形式の画像
            filename_prefix (str): ファイル名の接頭辞
            metadata (dict): 保存するメタデータ
        
        Returns:
            str: 保存されたファイルの絶対パス
        """
        # 画像を保存
        image_path = self.save_image(image, filename_prefix)
        if not image_path:
            return None
        
        # メタデータを保存
        metadata_path = Path(image_path).with_suffix('.json')
        try:
            # タイムスタンプを追加
            metadata_with_timestamp = metadata.copy()
            metadata_with_timestamp['saved_at'] = datetime.now().isoformat()
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_with_timestamp, f, ensure_ascii=False, indent=2)
                
            logger.info(f"メタデータを保存しました: {metadata_path}")
            return image_path
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")
            return image_path  # 画像は保存できたのでそのパスを返す
    
    def list_generated_images(self, directory=None):
        """生成された画像のリストを取得する
        
        Args:
            directory (str, optional): 検索対象のディレクトリ。指定がなければ基本ディレクトリを検索。
        
        Returns:
            list: 画像ファイル名のリスト
        """
        if directory:
            search_dir = self.output_dir / directory
        else:
            search_dir = self.output_dir
            
        # ディレクトリが存在しない場合は空リストを返す
        if not search_dir.exists():
            return []
            
        # 画像ファイルの拡張子
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        
        # 画像ファイルを検索
        image_files = []
        for file in os.listdir(search_dir):
            file_path = os.path.join(search_dir, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    image_files.append(file)
        
        return image_files
    
    def batch_rename(self, prefix=None, remove_pattern=None, directory=None):
        """画像ファイルの一括リネーム
        
        Args:
            prefix (str, optional): 新しいファイル名の接頭辞
            remove_pattern (str, optional): 削除するパターン
            directory (str, optional): 対象ディレクトリ。指定がなければ基本ディレクトリ。
        
        Returns:
            list: 新しいファイル名のリスト
        """
        if directory:
            target_dir = self.output_dir / directory
        else:
            target_dir = self.output_dir
            
        # ディレクトリが存在しない場合は空リストを返す
        if not target_dir.exists():
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
    
    def load_image_metadata(self, image_path):
        """画像のメタデータを読み込む
        
        Args:
            image_path (str): 画像ファイルのパス
        
        Returns:
            dict: メタデータ辞書。ファイルが存在しない場合はNone。
        """
        # 画像ファイルのパスがPATH、文字列どちらでも動くように
        path = Path(image_path)
        
        # メタデータファイルのパスを作成（拡張子をjsonに変更）
        metadata_path = path.with_suffix('.json')
        
        # メタデータファイルが存在するか確認
        if not metadata_path.exists():
            logger.warning(f"メタデータファイルが見つかりません: {metadata_path}")
            return None
        
        try:
            # メタデータの読み込み
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            return metadata
        except Exception as e:
            logger.error(f"メタデータ読み込みエラー: {e}")
            return None 