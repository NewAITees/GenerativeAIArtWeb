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
from src.utils.file_manager import FileManager
import pathlib
from pathlib import Path
from PIL import Image
import numpy as np

@pytest.fixture
def mock_image():
    """モック画像オブジェクトを作成する"""
    image = MagicMock(spec=Image.Image)
    image.save = MagicMock()
    return image

@pytest.fixture
def file_manager():
    """FileManagerインスタンスを作成する"""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('os.chmod') as mock_chmod, \
         patch('os.stat') as mock_stat:
        
        # 基本的なモックの設定
        mock_stat.return_value.st_mode = 0o755
        
        # カスタム出力ディレクトリを使用
        manager = FileManager("custom_outputs")
        
        # モックオブジェクトをマネージャに追加
        manager._mock_mkdir = mock_mkdir
        manager._mock_chmod = mock_chmod
        manager._mock_stat = mock_stat
        
        return manager

class TestFileManager:
    """Test class for file management utility."""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Set up test environment and clean up after each test."""
        # テスト用ディレクトリのパス
        self.test_dir = "/tmp/test_outputs"
        
        # テスト前にディレクトリが存在する場合は削除
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # テストディレクトリを作成
        os.makedirs(self.test_dir)
        
        # テスト環境変数を設定
        os.environ["TEST_OUTPUT_DIR"] = self.test_dir
        
        yield
        
        # テスト後のクリーンアップ
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if "TEST_OUTPUT_DIR" in os.environ:
            del os.environ["TEST_OUTPUT_DIR"]
    
    def test_initialization(self, file_manager):
        """初期化のテスト"""
        # 出力ディレクトリの作成を確認
        file_manager._mock_mkdir.assert_any_call("custom_outputs", exist_ok=True)
        
        # 権限の設定を確認
        expected_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        file_manager._mock_chmod.assert_any_call("custom_outputs", expected_mode)
    
    def test_output_dir_creation(self, file_manager):
        """出力ディレクトリ作成のテスト"""
        # 新しいディレクトリを作成
        new_dir = "new_dir"
        with patch('pathlib.Path.exists', return_value=False), \
             patch('os.stat') as mock_stat, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_stat.return_value.st_mode = 0o755
            file_manager._ensure_directory_permissions(Path(new_dir))
        
        # ディレクトリの作成を確認
        mock_mkdir.assert_called_with(new_dir, exist_ok=True)
        
        # 権限の設定を確認
        expected_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        file_manager._mock_chmod.assert_any_call(new_dir, expected_mode)
    
    def test_directory_permissions(self):
        """Test directory permission management."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('os.stat') as mock_stat, \
             patch('os.chmod') as mock_chmod, \
             patch('pathlib.Path.mkdir'), \
             patch('os.makedirs'):
            
            # 不適切な権限を持つディレクトリをシミュレート
            mock_stat.return_value.st_mode = 0o600  # 読み書きのみ
            
            manager = FileManager("test_dir")
            
            # 権限が修正されたことを確認
            expected_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            mock_chmod.assert_any_call("test_dir", expected_mode)
    
    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('os.chmod', side_effect=PermissionError("Access denied")), \
             pytest.raises(PermissionError) as exc_info:
            
            FileManager(output_dir="test_dir")
            assert "Access denied" in str(exc_info.value)
    
    def test_test_output_dir_environment_variable(self):
        """Test that TEST_OUTPUT_DIR environment variable is respected."""
        with patch.dict('os.environ', {'TEST_OUTPUT_DIR': '/tmp/test_outputs'}), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('os.chmod') as mock_chmod, \
             patch('os.stat') as mock_stat:
            
            mock_stat_result = MagicMock()
            mock_stat_result.st_mode = 0o755
            mock_stat.return_value = mock_stat_result
            
            manager = FileManager()
            assert str(manager.output_dir) == '/tmp/test_outputs'
    
    def test_generate_filename(self, file_manager):
        """Test generating a filename."""
        # Test basic filename generation
        filename = file_manager.generate_filename("test prompt")
        assert "test_prompt" in filename  # スペースはアンダースコアに変換される
        assert datetime.now().strftime("%Y%m%d") in filename
        assert filename.endswith(".png")
        
        # Test with custom parameters
        filename = file_manager.generate_filename(
            "test",
            prefix="gen",
            include_date=False,
            include_time=True,
            extension="jpg"
        )
        assert filename.startswith("gen")
        assert "test" in filename
        assert datetime.now().strftime("%H%M%S") in filename
        assert filename.endswith(".jpg")
        
        # Test with invalid characters in prompt
        filename = file_manager.generate_filename("test/with\\invalid:chars")
        assert "test_with_invalid_chars" in filename
        assert not any(c in filename for c in r'\/:"<>|')
    
    def test_organize_by_folder(self, file_manager):
        """Test organizing images into folders."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            path = file_manager.organize_by_folder("test_cat", "image.png", "prompt")
            
            mock_mkdir.assert_called_once()
            assert "test_cat" in path
            assert path.endswith("image.png")
    
    def test_list_generated_images(self, file_manager):
        """Test listing generated images."""
        mock_files = ["img1.png", "img2.jpg", "data.json", "img3.jpeg"]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('os.listdir', return_value=mock_files), \
             patch('pathlib.Path.isfile', return_value=True):
            
            images = file_manager.list_generated_images()
            assert len(images) == 3
            assert all(f in images for f in ["img1.png", "img2.jpg", "img3.jpeg"])
            assert "data.json" not in images
    
    def test_save_image_with_permissions(self, file_manager, mock_image):
        """画像保存と権限設定のテスト"""
        # 画像の保存
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.isfile', return_value=True), \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('os.chmod') as mock_chmod:
            
            # 画像を保存
            file_path = file_manager.save_image(mock_image, "test")
        
        # 保存の呼び出しを確認
        mock_image.save.assert_called_once()
        
        # 権限の設定を確認（644権限）
        expected_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        mock_chmod.assert_any_call(ANY, expected_mode)
        
        # ディレクトリの作成と権限設定を確認
        mock_mkdir.assert_called_with("custom_outputs", exist_ok=True)
        
        # ディレクトリの権限設定を確認（755権限）
        expected_dir_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        mock_chmod.assert_any_call("custom_outputs", expected_dir_mode)
    
    def test_save_metadata_atomicity(self):
        """Test atomic metadata saving operation."""
        metadata = {"prompt": "test", "steps": 30}
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('os.chmod') as mock_chmod, \
             patch('shutil.move') as mock_move, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.isfile', return_value=True), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            manager = FileManager()
            test_path = Path("test.json")
            manager._save_metadata(test_path, metadata)
            
            # ファイルのオープンと書き込みを確認
            mock_file.assert_called_once()
            mock_file().write.assert_called()
            
            # 一時ファイルの権限設定を確認（644権限）
            expected_file_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            temp_path = str(test_path.with_name(test_path.name + ".tmp"))
            mock_chmod.assert_called_once_with(temp_path, expected_file_mode)
            
            # アトミックな移動操作を確認
            mock_move.assert_called_once_with(temp_path, str(test_path))
    
    def test_cleanup_temp_files(self):
        """Test cleanup of temporary files on error."""
        metadata = {"prompt": "test", "steps": 30}
        
        with patch('builtins.open', side_effect=Exception("Write error")), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.isfile', return_value=True), \
             patch('os.remove') as mock_remove, \
             pytest.raises(Exception) as exc_info:
            
            manager = FileManager()
            test_path = Path("test.json")
            manager._save_metadata(test_path, metadata)
            
            # エラーメッセージを確認
            assert "Write error" in str(exc_info.value)
            
            # 一時ファイルのクリーンアップを確認
            temp_path = str(test_path.with_name(test_path.name + ".tmp"))
            mock_remove.assert_called_once_with(temp_path)
    
    def test_load_image_metadata(self, file_manager):
        """Test loading metadata for an image."""
        metadata = {
            "prompt": "test",
            "steps": 30,
            "saved_at": "2024-03-23T12:00:00"
        }
        
        # 正常系のテスト
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(metadata))):
            
            loaded = file_manager.load_image_metadata("test.png")
            assert loaded == metadata
            assert "prompt" in loaded
            assert loaded["steps"] == 30
            assert loaded["saved_at"] == "2024-03-23T12:00:00"
        
        # 存在しないファイルのテスト
        with patch('pathlib.Path.exists', return_value=False):
            assert file_manager.load_image_metadata("missing.png") is None
        
        # JSONデコードエラーのテスト
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")):
            
            assert file_manager.load_image_metadata("invalid.png") is None

    def test_save_image_none(self):
        """Noneの画像保存テスト"""
        manager = FileManager()
        assert manager.save_image(None, "test") is None

    def test_save_image_invalid(self):
        """無効な画像オブジェクトの保存テスト"""
        manager = FileManager()
        invalid_image = object()  # saveメソッドを持たないオブジェクト
        assert manager.save_image(invalid_image, "test") is None

    def test_save_metadata(self, file_manager):
        """メタデータ保存のテスト"""
        metadata = {"test": "data"}
        metadata_path = Path("test_metadata.json")
        
        # 一時ファイルの操作をモック
        with patch('builtins.open', MagicMock()) as mock_open, \
             patch('json.dump') as mock_dump, \
             patch('shutil.move') as mock_move, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.isfile', return_value=True), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            # ファイルハンドルのモックを設定
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # 一時ファイルのパスを作成
            temp_path = str(metadata_path.with_name(metadata_path.name + ".tmp"))
            
            # メタデータを保存
            file_manager._save_metadata(metadata_path, metadata)
            
            # メタデータの書き込みを確認
            mock_dump.assert_called_once()
            assert mock_dump.call_args[0][0]["test"] == "data"
            
            # 一時ファイルの権限設定を確認
            expected_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            mock_mkdir.assert_called_once_with(metadata_path.parent, exist_ok=True)
            mock_chmod.assert_any_call(temp_path, expected_mode)
            
            # 一時ファイルの移動を確認
            mock_move.assert_called_once_with(temp_path, str(metadata_path))

    def test_cleanup_on_error(self, file_manager):
        """エラー時のクリーンアップテスト"""
        metadata = {"test": "data"}
        metadata_path = Path("test_metadata.json")
        
        # 一時ファイルの操作をモック
        with patch('builtins.open') as mock_open, \
             patch('json.dump') as mock_dump, \
             patch('pathlib.Path.isfile', return_value=True), \
             patch('os.remove') as mock_remove, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            # エラーを発生させる
            mock_dump.side_effect = Exception("Test error")
            mock_exists.return_value = True
            
            # エラーが発生することを確認
            with pytest.raises(Exception):
                file_manager._save_metadata(metadata_path, metadata)
            
            # クリーンアップが実行されたことを確認
            mock_remove.assert_called_once() 