"""
Tests for the settings management utility.
This tests the functionality for saving, loading, and managing user settings.
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

class TestSettingsManager:
    """Test class for settings management utility."""
    
    @pytest.fixture
    def sample_settings(self):
        """Fixture providing a sample settings dictionary."""
        return {
            "prompt": "a cat in a garden",
            "steps": 30,
            "cfg_scale": 4.5,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "sampler": "euler",
            "upscale": 1.0,
            "watermark": True,
            "watermark_text": "Generated with AI",
            "watermark_opacity": 0.3
        }
    
    def test_initialization(self):
        """Test that settings manager initializes correctly."""
        with patch('os.path.exists', return_value=True):
            from utils.settings import SettingsManager
            
            # Initialize with default settings dir
            manager = SettingsManager()
            assert str(manager.settings_dir).endswith("settings")
            
            # Initialize with custom settings dir
            custom_dir = "custom_settings"
            manager = SettingsManager(settings_dir=custom_dir)
            assert str(manager.settings_dir).endswith(custom_dir)
    
    def test_settings_dir_creation(self):
        """Test that settings directory is created if it doesn't exist."""
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs:
            
            from utils.settings import SettingsManager
            
            # Initialize with non-existent settings dir
            manager = SettingsManager(settings_dir="new_settings_dir")
            
            # Check that makedirs was called
            mock_makedirs.assert_called_once_with("new_settings_dir", exist_ok=True)
    
    def test_save_settings_profile(self, sample_settings):
        """Test saving settings as a profile."""
        mock_file_handle = mock_open()
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_file_handle), \
             patch('json.dump') as mock_json_dump:
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # Test saving settings
            success = manager.save_profile("test_profile", sample_settings)
            
            # Check that the file was opened for writing
            profile_path = os.path.join(manager.settings_dir, "test_profile.json")
            mock_file_handle.assert_called_once_with(profile_path, 'w', encoding='utf-8')
            
            # Check that json.dump was called with the settings
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            assert args[0] == sample_settings
            
            # Check the return value
            assert success is True
    
    def test_load_settings_profile(self, sample_settings):
        """Test loading settings from a profile."""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(sample_settings))), \
             patch('json.load', return_value=sample_settings) as mock_json_load:
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # Test loading settings
            loaded_settings = manager.load_profile("test_profile")
            
            # Check that json.load was called
            mock_json_load.assert_called_once()
            
            # Check the loaded settings
            assert loaded_settings == sample_settings
            assert loaded_settings["prompt"] == "a cat in a garden"
            assert loaded_settings["steps"] == 30
    
    def test_load_nonexistent_profile(self):
        """Test loading a profile that doesn't exist."""
        with patch('os.path.exists', return_value=False):
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # Test loading a non-existent profile
            loaded_settings = manager.load_profile("nonexistent_profile")
            
            # Check that None was returned
            assert loaded_settings is None
    
    def test_list_profiles(self):
        """Test listing available profiles."""
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=["profile1.json", "profile2.json", "other_file.txt"]):
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # Test listing profiles
            profiles = manager.list_profiles()
            
            # Check the profiles
            assert len(profiles) == 2
            assert "profile1" in profiles
            assert "profile2" in profiles
            assert "other_file" not in profiles
    
    def test_delete_profile(self):
        """Test deleting a profile."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # Test deleting a profile
            success = manager.delete_profile("test_profile")
            
            # Check that remove was called with the correct path
            profile_path = os.path.join(manager.settings_dir, "test_profile.json")
            mock_remove.assert_called_once_with(profile_path)
            
            # Check the return value
            assert success is True
    
    def test_get_default_settings(self):
        """Test getting default settings."""
        from utils.settings import SettingsManager
        
        manager = SettingsManager()
        
        # Test getting default settings
        defaults = manager.get_default_settings()
        
        # Check some expected defaults
        assert "prompt" in defaults
        assert "steps" in defaults
        assert "cfg_scale" in defaults
        assert "width" in defaults
        assert "height" in defaults
    
    def test_update_settings(self, sample_settings):
        """Test updating settings with new values."""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('json.dump') as mock_json_dump:
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # New settings to update
            updates = {
                "prompt": "a dog on a beach",
                "steps": 40,
                "seed": 123
            }
            
            # Test updating settings
            updated = manager.update_profile("test_profile", updates, base_settings=sample_settings)
            
            # Check that json.dump was called with updated settings
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            
            # Check the updated settings
            assert args[0]["prompt"] == "a dog on a beach"
            assert args[0]["steps"] == 40
            assert args[0]["seed"] == 123
            # Check that other settings remain unchanged
            assert args[0]["cfg_scale"] == 4.5
            assert args[0]["width"] == 1024
            
            # Check the return value
            assert updated == args[0]
    
    def test_export_settings(self, sample_settings):
        """Test exporting settings to a file."""
        mock_file_handle = mock_open()
        
        with patch('builtins.open', mock_file_handle), \
             patch('json.dump') as mock_json_dump:
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # Test exporting settings
            success = manager.export_settings(sample_settings, "exported_settings.json")
            
            # Check that the file was opened for writing
            mock_file_handle.assert_called_once_with("exported_settings.json", 'w', encoding='utf-8')
            
            # Check that json.dump was called with the settings
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            assert args[0] == sample_settings
            
            # Check the return value
            assert success is True
    
    def test_import_settings(self, sample_settings):
        """Test importing settings from a file."""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(sample_settings))), \
             patch('json.load', return_value=sample_settings) as mock_json_load:
            
            from utils.settings import SettingsManager
            
            manager = SettingsManager()
            
            # Test importing settings
            imported_settings = manager.import_settings("imported_settings.json")
            
            # Check that json.load was called
            mock_json_load.assert_called_once()
            
            # Check the imported settings
            assert imported_settings == sample_settings
            
            # Check specific settings
            assert imported_settings["prompt"] == "a cat in a garden"
            assert imported_settings["steps"] == 30 