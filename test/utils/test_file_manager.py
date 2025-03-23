"""
Tests for the file management utility.
This tests the functionality for saving, organizing, and managing generated images.
"""
import pytest
import os
import json
import stat
import shutil
from unittest.mock import patch, MagicMock, mock_open, call, ANY
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np

from src.utils.file_manager import FileManager
from src.models.file_manager import FileMetadata, FilenameConfig, FileExtension
from src.config.file_manager_config import (
    FILE_PERMISSIONS,
    IMAGE_CONFIG,
    DIRECTORY_STRUCTURE,
    METADATA_CONFIG,
    FILENAME_CONFIG
)

@pytest.fixture
def mock_image():
    """モック画像オブジェクトを作成する"""
    image = MagicMock(spec=Image.Image)
    image.save = MagicMock()
    return image

@pytest.fixture
def file_manager(tmp_path):
    """FileManagerインスタンスを作成する"""
    return FileManager(output_dir=str(tmp_path))

class TestFileManager:
    """Test class for file management utility."""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self, tmp_path):
        """Set up test environment and clean up after each test."""
        self.test_dir = tmp_path
        
        # テスト環境変数を設定
        with patch.dict('os.environ', {'TEST_OUTPUT_DIR': str(self.test_dir)}):
            yield
    
    def test_initialization(self, file_manager):
        """初期化のテスト"""
        # 基本ディレクトリの存在を確認
        assert file_manager.output_dir.exists()
        
        # サブディレクトリの存在を確認
        for subdir in DIRECTORY_STRUCTURE['subdirs']:
            assert (file_manager.output_dir / subdir).exists()
        
        # 権限の確認
        assert (os.stat(file_manager.output_dir).st_mode & 0o777) == (FILE_PERMISSIONS['directory'] & 0o777)
    
    def test_save_image_with_permissions(self, file_manager, mock_image):
        """画像保存と権限設定のテスト"""
        # 画像を保存
        file_path = file_manager.save_image(mock_image, "test")
        
        # 保存の呼び出しを確認
        mock_image.save.assert_called_once()
        
        # ファイルの存在を確認
        assert os.path.exists(file_path)
        
        # 権限の確認
        assert (os.stat(file_path).st_mode & 0o777) == (FILE_PERMISSIONS['file'] & 0o777)
    
    def test_save_metadata_atomicity(self, file_manager):
        """メタデータの保存とアトミック性のテスト"""
        metadata = FileMetadata(
            prompt="test prompt",
            steps=30,
            seed=12345,
            model="test_model"
        )
        
        # メタデータを保存
        test_path = file_manager.output_dir / "test.json"
        file_manager._save_metadata(test_path, metadata.model_dump())
        
        # ファイルの存在を確認
        assert test_path.exists()
        
        # 権限の確認
        assert (os.stat(test_path).st_mode & 0o777) == (FILE_PERMISSIONS['file'] & 0o777)
        
        # 内容の確認
        with open(test_path, 'r', encoding=METADATA_CONFIG['encoding']) as f:
            saved_data = json.load(f)
            assert saved_data['prompt'] == metadata.prompt
            assert saved_data['steps'] == metadata.steps
    
    def test_generate_filename(self, file_manager):
        """ファイル名生成のテスト"""
        # 基本的なファイル名生成
        filename = file_manager.generate_filename("test prompt")
        assert "test_prompt" in filename
        assert datetime.now().strftime(FILENAME_CONFIG['date_format']) in filename
        assert filename.endswith(".png")
        
        # カスタムパラメータでのテスト
        filename = file_manager.generate_filename(
            "test",
            prefix="gen",
            include_date=False,
            include_time=True,
            extension="jpg"
        )
        assert filename.startswith("gen")
        assert "test" in filename
        assert datetime.now().strftime(FILENAME_CONFIG['time_format']) in filename
        assert filename.endswith(".jpg")
        
        # 無効な文字を含むプロンプトのテスト
        filename = file_manager.generate_filename("test/with\\invalid:chars")
        assert "test_with_invalid_chars" in filename
        assert not any(c in filename for c in FILENAME_CONFIG['invalid_chars'])
    
    def test_save_image_with_metadata(self, file_manager, mock_image):
        """画像とメタデータの保存テスト"""
        metadata = FileMetadata(
            prompt="test prompt",
            steps=30,
            seed=12345,
            model="test_model"
        )
        
        # 画像とメタデータを保存
        image_path = file_manager.save_image_with_metadata(
            mock_image,
            "test",
            metadata
        )
        
        # 画像ファイルの存在を確認
        assert os.path.exists(image_path)
        
        # メタデータファイルの存在を確認
        metadata_path = os.path.splitext(image_path)[0] + METADATA_CONFIG['extension']
        assert os.path.exists(metadata_path)
        
        # メタデータの内容を確認
        with open(metadata_path, 'r', encoding=METADATA_CONFIG['encoding']) as f:
            saved_data = json.load(f)
            assert saved_data['prompt'] == metadata.prompt
            assert saved_data['steps'] == metadata.steps
    
    def test_error_handling(self, file_manager):
        """エラーハンドリングのテスト"""
        # 無効な画像オブジェクトでの保存テスト
        with pytest.raises(ValueError):
            file_manager.save_image("not an image", "test")
        
        # 権限エラーのテスト
        with patch('os.chmod', side_effect=PermissionError("Access denied")), \
             pytest.raises(PermissionError):
            file_manager._ensure_directory_permissions(Path("test_dir"))
        
        # JSONデコードエラーのテスト
        with patch('builtins.open', mock_open(read_data="invalid json")), \
             pytest.raises(json.JSONDecodeError):
            file_manager.load_image_metadata("test.png")
    
    def test_cleanup_temp_files(self, file_manager):
        """一時ファイルのクリーンアップテスト"""
        metadata = {"test": "data"}
        test_path = file_manager.output_dir / "test.json"
        
        # 書き込みエラーをシミュレート
        with patch('builtins.open', side_effect=Exception("Write error")), \
             pytest.raises(Exception):
            file_manager._save_metadata(test_path, metadata)
        
        # 一時ファイルが残っていないことを確認
        temp_path = test_path.with_suffix(test_path.suffix + METADATA_CONFIG['temp_suffix'])
        assert not temp_path.exists()
    
    def test_unique_filepath_generation(self, file_manager, mock_image):
        """重複ファイル名の処理テスト"""
        # 最初のファイルを保存
        first_path = file_manager.save_image(mock_image, "test")
        
        # 同じ名前で2つ目のファイルを保存
        second_path = file_manager.save_image(mock_image, "test")
        
        # パスが異なることを確認
        assert first_path != second_path
        assert os.path.exists(first_path)
        assert os.path.exists(second_path)
        
        # 命名規則の確認
        first_name = os.path.basename(first_path)
        second_name = os.path.basename(second_path)
        assert first_name != second_name
        assert "_1" in second_name 