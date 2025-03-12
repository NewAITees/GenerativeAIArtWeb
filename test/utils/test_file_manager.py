"""
Tests for the file management utility.
This tests the functionality for saving, organizing, and managing generated images.
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

class TestFileManager:
    """Test class for file management utility."""
    
    def test_initialization(self):
        """Test that the FileManager initializes correctly."""
        with patch('os.path.exists', return_value=True):
            from utils.file_manager import FileManager
            
            # Initialize with default settings
            manager = FileManager()
            
            # Check default properties
            assert manager.output_dir.endswith("outputs")
            
            # Test with custom output directory
            manager = FileManager(output_dir="custom_outputs")
            
            # Check custom properties
            assert manager.output_dir.endswith("custom_outputs")
    
    def test_output_dir_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs:
            
            from utils.file_manager import FileManager
            
            # Initialize with non-existent output dir
            manager = FileManager(output_dir="new_dir")
            
            # Check that makedirs was called
            mock_makedirs.assert_called_once_with("new_dir", exist_ok=True)
    
    def test_save_image(self, mock_pil_image):
        """Test saving an image."""
        with patch('os.path.exists', return_value=True), \
             patch.object(mock_pil_image, 'save') as mock_save:
            
            from utils.file_manager import FileManager
            
            manager = FileManager()
            
            # Test saving an image
            filename = manager.save_image(mock_pil_image, "test_image")
            
            # Check that the image was saved
            mock_save.assert_called_once()
            
            # Check that the filename uses the provided name
            assert "test_image" in filename
    
    def test_generate_filename(self):
        """Test generating a filename."""
        from utils.file_manager import FileManager
        
        manager = FileManager()
        
        # Test generating a filename with default format
        filename = manager.generate_filename("cat")
        
        # Check that the filename includes the prompt
        assert "cat" in filename
        
        # Check that the filename has a timestamp
        assert datetime.now().strftime("%Y%m%d") in filename
        
        # Check that the file extension is correct
        assert filename.endswith(".png")
        
        # Test with custom format
        filename = manager.generate_filename(
            "mountain",
            prefix="gen",
            include_date=False,
            include_time=True,
            extension="jpg"
        )
        
        # Check prefix
        assert filename.startswith("gen")
        
        # Check prompt
        assert "mountain" in filename
        
        # Check time format
        time_format = datetime.now().strftime("%H%M%S")
        assert time_format in filename
        
        # Check extension
        assert filename.endswith(".jpg")
    
    def test_organize_by_folder(self):
        """Test organizing images into folders."""
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs, \
             patch('os.path.join', side_effect=os.path.join):
            
            from utils.file_manager import FileManager
            
            manager = FileManager()
            
            # Test organizing by folder
            path = manager.organize_by_folder("cat", "cat_image.png", folder_type="prompt")
            
            # Check that folder was created
            mock_makedirs.assert_called_once()
            assert "cat" in mock_makedirs.call_args[0][0]
            
            # Check that the returned path includes the folder
            assert "cat" in path
            assert path.endswith("cat_image.png")
    
    def test_save_with_metadata(self, mock_pil_image):
        """Test saving an image with metadata."""
        mock_open_handler = mock_open()
        
        with patch('os.path.exists', return_value=True), \
             patch.object(mock_pil_image, 'save') as mock_save, \
             patch('builtins.open', mock_open_handler), \
             patch('json.dump') as mock_json_dump:
            
            from utils.file_manager import FileManager
            
            manager = FileManager()
            
            # Metadata to save
            metadata = {
                "prompt": "cat",
                "steps": 30,
                "cfg_scale": 4.5,
                "seed": 42
            }
            
            # Test saving with metadata
            filename = manager.save_image_with_metadata(mock_pil_image, "test_image", metadata)
            
            # Check that the image was saved
            mock_save.assert_called_once()
            
            # Check that the metadata file was opened
            metadata_filename = filename.replace(".png", ".json")
            mock_open_handler.assert_called_with(metadata_filename, 'w')
            
            # Check that the metadata was saved
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            assert args[0] == metadata
    
    def test_list_generated_images(self):
        """Test listing generated images."""
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=["img1.png", "img2.png", "data.json", "img3.png"]), \
             patch('os.path.isfile', return_value=True):
            
            from utils.file_manager import FileManager
            
            manager = FileManager()
            
            # Test listing images
            images = manager.list_generated_images()
            
            # Check that only image files are returned
            assert len(images) == 3
            assert "img1.png" in images
            assert "img2.png" in images
            assert "img3.png" in images
            assert "data.json" not in images
    
    def test_batch_rename(self):
        """Test batch renaming of files."""
        mock_files = ["img1.png", "img2.png", "img3.png"]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=mock_files), \
             patch('os.path.isfile', return_value=True), \
             patch('os.rename') as mock_rename:
            
            from utils.file_manager import FileManager
            
            manager = FileManager()
            
            # Test batch renaming
            new_names = manager.batch_rename(prefix="new_", remove_pattern="img")
            
            # Check that rename was called for each file
            assert mock_rename.call_count == 3
            
            # Check the new names
            assert all("new_" in name for name in new_names)
            assert all("img" not in name for name in new_names)
    
    def test_load_image_metadata(self):
        """Test loading metadata for an image."""
        metadata = {
            "prompt": "cat",
            "steps": 30,
            "cfg_scale": 4.5,
            "seed": 42
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(metadata))):
            
            from utils.file_manager import FileManager
            
            manager = FileManager()
            
            # Test loading metadata
            loaded_metadata = manager.load_image_metadata("test_image.png")
            
            # Check the loaded metadata
            assert loaded_metadata == metadata
            assert loaded_metadata["prompt"] == "cat"
            assert loaded_metadata["steps"] == 30 