"""
ファイル管理機能

このモジュールでは、生成された画像のカスタム保存や管理機能を提供します。
主な機能:
- 画像ファイルの保存と管理
- メタデータの保存と読み込み
- ディレクトリ構造の管理
- ファイル名の生成と正規化
"""

import os
import re
import json
import stat
import logging
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, List, Union, TypeVar, cast

from src.models.file_manager import FileMetadata, FilenameConfig, FileExtension
from src.config.file_manager_config import (
    PROJECT_ROOT,
    FILE_PERMISSIONS,
    IMAGE_CONFIG,
    DIRECTORY_STRUCTURE,
    METADATA_CONFIG,
    FILENAME_CONFIG,
    ERROR_MESSAGES,
    LOG_MESSAGES
)

# ロギング設定
logger = logging.getLogger(__name__)

# 型変数の定義
ImageType = TypeVar('ImageType', Image.Image, np.ndarray)

class FileManager:
    """画像ファイル管理クラス
    
    このクラスは、画像ファイルの保存、メタデータの管理、ディレクトリ構造の
    管理など、ファイル関連の操作を提供します。
    
    Attributes:
        output_dir (Path): 出力ディレクトリのパス
        IMAGE_EXTENSIONS (Set[str]): サポートされる画像ファイルの拡張子
    
    Example:
        ```python
        # FileManagerのインスタンス化
        manager = FileManager("outputs")
        
        # 画像の保存
        image_path = manager.save_image(image, "test_image")
        
        # メタデータの保存
        manager.save_metadata(image_path, {"prompt": "test", "steps": 30})
        ```
    """
    
    # クラス変数の定義
    IMAGE_EXTENSIONS: set[str] = IMAGE_CONFIG['extensions']
    DIR_PERMISSION: int = FILE_PERMISSIONS['directory']
    FILE_PERMISSION: int = FILE_PERMISSIONS['file']
    
    def __init__(self, output_dir: Optional[str] = None):
        """初期化メソッド
        
        Args:
            output_dir (str or Path, optional): 保存の基準ディレクトリ。
                指定がない場合はプロジェクトルートのoutputsディレクトリを使用。
        """
        # 環境変数からテスト用ディレクトリを取得
        if 'TEST_OUTPUT_DIR' in os.environ and not output_dir:
            self.output_dir = Path(os.environ['TEST_OUTPUT_DIR'])
        else:
            self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / DIRECTORY_STRUCTURE['root']
        
        try:
            # 基本ディレクトリが存在しない場合は作成
            os.makedirs(str(self.output_dir), exist_ok=True)
            
            # 権限を設定
            self._ensure_directory_permissions(self.output_dir)
            
            # サブディレクトリの作成
            self._initialize_directories()
            
        except PermissionError as e:
            logger.error(ERROR_MESSAGES['permission_denied'].format(path=self.output_dir))
            raise
        except FileNotFoundError as e:
            logger.error(ERROR_MESSAGES['file_not_found'].format(path=self.output_dir))
            raise
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            raise
    
    def _initialize_directories(self) -> None:
        """必要なサブディレクトリを作成する"""
        for subdir in DIRECTORY_STRUCTURE['subdirs']:
            try:
                path = self.output_dir / subdir
                os.makedirs(str(path), exist_ok=True)
                self._ensure_directory_permissions(path)
            except Exception as e:
                logger.error(f"サブディレクトリの初期化に失敗しました: {subdir}: {e}")
                raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """ファイル名を正規化する
        
        Args:
            filename (str): 正規化するファイル名
        
        Returns:
            str: 正規化されたファイル名
        """
        # 無効な文字を置換
        filename = re.sub(FILENAME_CONFIG['invalid_chars'], '_', filename)
        # 空白文字を_に置換
        filename = re.sub(r'\s+', '_', filename)
        # 連続する_を単一の_に置換
        filename = re.sub(r'_+', '_', filename)
        # 先頭と末尾の_を削除
        filename = filename.strip('_')
        # 最大長を制限
        return filename[:FILENAME_CONFIG['max_length']]
    
    def generate_filename(
        self, 
        prompt: str, 
        *,
        prefix: str = "",
        include_date: bool = True,
        include_time: bool = False,
        extension: str = "png"
    ) -> str:
        """ファイル名を生成する
        
        Args:
            prompt (str): プロンプトテキスト
            prefix (str, optional): ファイル名の接頭辞
            include_date (bool, optional): 日付を含めるかどうか
            include_time (bool, optional): 時刻を含めるかどうか
            extension (str, optional): ファイルの拡張子
        
        Returns:
            str: 生成されたファイル名
        """
        # プロンプトの短縮と正規化
        short_prompt = self._sanitize_filename(prompt)
        
        # 現在の日時
        now = datetime.now()
        date_str = now.strftime(FILENAME_CONFIG['date_format']) if include_date else ""
        time_str = now.strftime(FILENAME_CONFIG['time_format']) if include_time else ""
        
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
        
        # 拡張子の処理
        if extension.startswith('.'):
            extension = extension[1:]
        
        # ファイル名の構築
        return FILENAME_CONFIG['separator'].join(filter(None, parts)) + f".{extension}"
    
    def save_image_with_metadata(
        self, 
        image: Any, 
        filename_prefix: str,
        metadata: Union[Dict[str, Any], FileMetadata]
    ) -> Optional[str]:
        """画像とメタデータを保存する
        
        Args:
            image: PIL.Imageまたはnumpy.ndarray形式の画像
            filename_prefix (str): ファイル名の接頭辞
            metadata (dict or FileMetadata): 保存するメタデータ
        
        Returns:
            str: 保存された画像ファイルのパス
        """
        # 画像の保存
        image_path = self.save_image(image, filename_prefix)
        if not image_path:
            return None
        
        try:
            # メタデータをFileMetadataに変換
            if isinstance(metadata, dict):
                file_metadata = FileMetadata.model_validate(metadata)
            else:
                file_metadata = metadata
            
            # メタデータファイルのパスを生成
            metadata_path = Path(os.path.splitext(image_path)[0] + ".json")
            
            # メタデータの保存
            self._save_metadata(metadata_path, file_metadata.model_dump())
            
            logger.info(f"メタデータを保存しました: {metadata_path}")
            return image_path
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")
            return image_path
    
    def _ensure_directory_permissions(self, directory: Path) -> None:
        """ディレクトリの権限を設定する
        
        Args:
            directory (Path): 権限を設定するディレクトリ
        """
        try:
            # ディレクトリが存在しない場合は作成
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory), exist_ok=True)
            
            # ディレクトリの権限を設定（755 = rwxr-xr-x）
            os.chmod(str(directory), self.DIR_PERMISSION)
        except FileExistsError:
            # ディレクトリがすでに存在する場合は無視
            logger.debug(f"ディレクトリはすでに存在します: {directory}")
        except Exception as e:
            logger.warning(f"ディレクトリの権限設定に失敗しました: {e}")
            raise
    
    def _save_metadata(self, path: Path, metadata: Dict[str, Any]) -> None:
        """メタデータをJSONファイルとして保存する
        
        Args:
            path (Path): 保存先のパス
            metadata (dict): 保存するメタデータ
        
        Raises:
            IOError: ファイルの書き込みに失敗した場合
            JSONDecodeError: JSONのエンコードに失敗した場合
            PermissionError: ファイルの権限設定に失敗した場合
        """
        temp_path = path.with_suffix(path.suffix + '.tmp')
        try:
            # 一時ファイルに書き込み
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 一時ファイルの権限を設定
            os.chmod(temp_path, self.FILE_PERMISSION)
            
            # アトミックな移動操作
            shutil.move(temp_path, path)
            
        except json.JSONEncodeError as e:
            logger.error(f"JSONエンコードエラー: {e}")
            self._cleanup_temp_file(temp_path)
            raise
        except IOError as e:
            logger.error(f"ファイル書き込みエラー: {e}")
            self._cleanup_temp_file(temp_path)
            raise
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")
            self._cleanup_temp_file(temp_path)
            raise
    
    def _cleanup_temp_file(self, path: Path) -> None:
        """一時ファイルを削除する
        
        Args:
            path (Path): 削除する一時ファイルのパス
        """
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.warning(f"一時ファイルの削除に失敗しました: {path}: {e}")
    
    def save_image(self, image: ImageType, filename_prefix: str,
                  directory: Optional[str] = None) -> Optional[str]:
        """画像を指定したディレクトリとファイル名で保存する
        
        Args:
            image: PIL.Imageまたはnumpy.ndarray形式の画像
            filename_prefix (str): ファイル名の接頭辞
            directory (str, optional): 保存先ディレクトリ。基本ディレクトリからの相対パス。
                指定がなければ基本ディレクトリに保存。
        
        Returns:
            str: 保存されたファイルの絶対パス
            None: 保存に失敗した場合
        
        Raises:
            ValueError: 画像オブジェクトが無効な場合
            IOError: ファイルの書き込みに失敗した場合
            PermissionError: ファイルの権限設定に失敗した場合
        """
        if image is None:
            logger.warning("保存する画像がNoneです")
            return None
        
        if not hasattr(image, 'save'):
            logger.error(f"無効な画像オブジェクトです: {type(image)}")
            raise ValueError(f"画像オブジェクトにsaveメソッドがありません: {type(image)}")
        
        try:
            # 保存先ディレクトリの準備
            save_dir = Path(self.output_dir) / (directory or "")
            save_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_directory_permissions(save_dir)
            
            # ファイル名を生成
            filename = self.generate_filename(filename_prefix)
            file_path = save_dir / filename
            
            # ファイル名の衝突を回避
            file_path = self._get_unique_filepath(file_path)
            
            # 画像の保存
            image.save(str(file_path))
            os.chmod(str(file_path), self.FILE_PERMISSION)
            
            logger.info(f"画像を保存しました: {file_path}")
            return str(file_path)
            
        except IOError as e:
            logger.error(f"画像保存エラー (IO): {e}")
            raise
        except PermissionError as e:
            logger.error(f"画像保存エラー (権限): {e}")
            raise
        except Exception as e:
            logger.error(f"画像保存エラー: {e}")
            raise
    
    def _get_unique_filepath(self, file_path: Path) -> Path:
        """重複しないファイルパスを取得する
        
        Args:
            file_path (Path): 元のファイルパス
        
        Returns:
            Path: 重複しない新しいファイルパス
        """
        if not file_path.exists():
            return file_path
        
        counter = 1
        while True:
            stem = file_path.stem
            suffix = file_path.suffix
            new_path = file_path.with_name(f"{stem}_{counter}{suffix}")
            if not new_path.exists():
                return new_path
            counter += 1
    
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
            None: メタデータが存在しないまたは読み込みに失敗した場合
        
        Raises:
            FileNotFoundError: メタデータファイルが存在しない場合
            JSONDecodeError: JSONのデコードに失敗した場合
            PermissionError: ファイルの読み込み権限がない場合
        """
        metadata_path = Path(os.path.splitext(image_path)[0] + ".json")
        
        if not metadata_path.exists():
            logger.warning(f"メタデータファイルが見つかりません: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSONデコードエラー: {metadata_path}: {e}")
            raise
        except IOError as e:
            logger.error(f"メタデータ読み込みエラー: {metadata_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"予期せぬエラー: {metadata_path}: {e}")
            raise 