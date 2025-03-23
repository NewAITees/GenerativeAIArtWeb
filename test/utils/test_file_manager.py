"""
Tests for the file management utility.
This tests the functionality for saving, organizing, and managing generated images.
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

@pytest.fixture
def mock_pil_image():
    """Create a mock PIL Image object."""
    mock_image = MagicMock()
    mock_image.save = MagicMock()
    return mock_image

class TestFileManager:
    """Test class for file management utility."""
    
    def test_initialization(self):
        """Test that the FileManager initializes correctly."""
        with patch('os.path.exists', return_value=True), \
             patch('os.makedirs') as mock_makedirs:
            from utils.file_manager import FileManager
            
            # Initialize with default settings
            manager = FileManager()
            assert manager.output_dir.endswith("outputs")
            mock_makedirs.assert_called_with(manager.output_dir, exist_ok=True)
            
            # Test with custom output directory
            custom_dir = "custom_outputs"
            manager = FileManager(output_dir=custom_dir)
            assert manager.output_dir.endswith(custom_dir)
            mock_makedirs.assert_called_with(custom_dir, exist_ok=True)
    
    def test_output_dir_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs:
            from utils.file_manager import FileManager
            manager = FileManager(output_dir="new_dir")
            mock_makedirs.assert_called_with("new_dir", exist_ok=True)
    
    def test_save_image(self, mock_pil_image):
        """Test saving an image."""
        with patch('os.path.exists', return_value=True), \
             patch('os.makedirs'), \
             patch.object(mock_pil_image, 'save') as mock_save:
            from utils.file_manager import FileManager
            
            manager = FileManager()
            filename = manager.save_image(mock_pil_image, "test_image")
            
            mock_save.assert_called_once()
            assert "test_image" in filename
            
            # Test with None image
            assert manager.save_image(None, "test_image") is None
            
            # Test with invalid image object
            invalid_image = MagicMock()
            assert manager.save_image(invalid_image, "test_image") is None
    
    def test_generate_filename(self):
        """Test generating a filename."""
        from utils.file_manager import FileManager
        manager = FileManager()
        
        # Test basic filename generation
        filename = manager.generate_filename("test prompt")
        assert "test_prompt" in filename
        assert datetime.now().strftime("%Y%m%d") in filename
        assert filename.endswith(".png")
        
        # Test with custom parameters
        filename = manager.generate_filename(
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
    
    def test_organize_by_folder(self):
        """Test organizing images into folders."""
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs:
            from utils.file_manager import FileManager
            
            manager = FileManager()
            path = manager.organize_by_folder("test_cat", "image.png", "prompt")
            
            mock_makedirs.assert_called_once()
            assert "test_cat" in path
            assert path.endswith("image.png")
    
    def test_save_with_metadata(self, mock_pil_image):
        """Test saving an image with metadata."""
        metadata = {"prompt": "test", "steps": 30}
        mock_open_handler = mock_open()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.makedirs'), \
             patch.object(mock_pil_image, 'save'), \
             patch('builtins.open', mock_open_handler), \
             patch('json.dump') as mock_json_dump:
            
            from utils.file_manager import FileManager
            manager = FileManager()
            
            filename = manager.save_image_with_metadata(mock_pil_image, "test", metadata)
            
            assert filename is not None
            mock_open_handler.assert_called_once()
            mock_json_dump.assert_called_once_with(metadata, mock_open_handler(), indent=2)
    
    def test_list_generated_images(self):
        """Test listing generated images."""
        mock_files = ["img1.png", "img2.jpg", "data.json", "img3.jpeg"]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=mock_files), \
             patch('os.path.isfile', return_value=True):
            
            from utils.file_manager import FileManager
            manager = FileManager()
            
            images = manager.list_generated_images()
            assert len(images) == 3
            assert all(f in images for f in ["img1.png", "img2.jpg", "img3.jpeg"])
            assert "data.json" not in images
    
    def test_batch_rename(self):
        """Test batch renaming of files."""
        mock_files = ["old1.png", "old2.png"]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=mock_files), \
             patch('os.path.isfile', return_value=True), \
             patch('os.rename') as mock_rename:
            
            from utils.file_manager import FileManager
            manager = FileManager()
            
            new_names = manager.batch_rename(prefix="new_", remove_pattern="old")
            assert len(new_names) == 2
            assert all("new_" in name for name in new_names)
            assert all("old" not in name for name in new_names)
            assert mock_rename.call_count == 2
    
    def test_load_image_metadata(self):
        """Test loading metadata for an image."""
        metadata = {"prompt": "test", "steps": 30}
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(metadata))):
            
            from utils.file_manager import FileManager
            manager = FileManager()
            
            loaded = manager.load_image_metadata("test.png")
            assert loaded == metadata
            
            # Test with non-existent file
            with patch('os.path.exists', return_value=False):
                assert manager.load_image_metadata("missing.png") is None 