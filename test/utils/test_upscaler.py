"""
Tests for the image upscaler utility.
This tests the functionality for enhancing image resolution.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

class TestUpscaler:
    """Test class for image upscaler utility."""
    
    @pytest.fixture
    def mock_image_data(self):
        """Fixture providing mock image data as numpy array."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """Test upscaler initialization."""
        from utils.upscaler import Upscaler
        
        upscaler = Upscaler()
        assert upscaler is not None
    
    def test_upscale_pil_image(self, mock_pil_image):
        """Test upscaling using PIL-based methods."""
        with patch('PIL.Image.open', return_value=mock_pil_image), \
             patch('PIL.Image.resize') as mock_resize:
            
            mock_resize.return_value = MagicMock()
            mock_resize.return_value.size = (2048, 2048)
            
            from utils.upscaler import Upscaler
            
            upscaler = Upscaler(method="bicubic")
            
            # Test upscaling an image
            result = upscaler.upscale("test_image.png", scale=2.0)
            
            # Check that Image.resize was called with the correct parameters
            mock_resize.assert_called_once()
            args, kwargs = mock_resize.call_args
            
            # Check the size (2x the original 1024x1024)
            assert args[0] == (2048, 2048)
            
            # Check the resampling method
            from PIL import Image
            assert kwargs["resample"] == Image.Resampling.BICUBIC
            
            # Check the result size
            assert result.size == (2048, 2048)
    
    def test_upscale_with_different_methods(self, mock_pil_image):
        """Test upscaling with different resampling methods."""
        with patch('PIL.Image.open', return_value=mock_pil_image), \
             patch('PIL.Image.resize') as mock_resize:
            
            from utils.upscaler import Upscaler
            from PIL import Image
            
            # Map of method names to PIL resampling constants
            method_map = {
                "nearest": Image.Resampling.NEAREST,
                "box": Image.Resampling.BOX,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS
            }
            
            # Test each method
            for method_name, resampling_method in method_map.items():
                mock_resize.reset_mock()
                
                upscaler = Upscaler(method=method_name)
                upscaler.upscale("test_image.png", scale=2.0)
                
                # Check the resampling method used
                _, kwargs = mock_resize.call_args
                assert kwargs["resample"] == resampling_method
    
    def test_upscale_with_denoise(self, mock_pil_image):
        """Test upscaling with denoising."""
        with patch('PIL.Image.open', return_value=mock_pil_image), \
             patch('PIL.Image.resize') as mock_resize, \
             patch('utils.upscaler.denoise_image') as mock_denoise:
            
            mock_resize.return_value = mock_pil_image
            mock_denoise.return_value = mock_pil_image
            
            from utils.upscaler import Upscaler
            
            upscaler = Upscaler(method="bicubic", denoise_level=2)
            
            # Test upscaling with denoising
            result = upscaler.upscale("test_image.png", scale=2.0)
            
            # Check that denoise_image was called
            mock_denoise.assert_called_once()
            args, kwargs = mock_denoise.call_args
            
            # Check the denoise level
            assert kwargs["strength"] == 2
    
    def test_batch_upscale(self, mock_pil_image):
        """Test batch upscaling of multiple images."""
        with patch('utils.upscaler.os.path.exists', return_value=True), \
             patch('utils.upscaler.os.listdir', return_value=['img1.png', 'img2.png', 'img3.png']), \
             patch('utils.upscaler.os.path.isfile', return_value=True), \
             patch('PIL.Image.open', return_value=mock_pil_image), \
             patch('PIL.Image.resize', return_value=mock_pil_image) as mock_resize:
            
            from utils.upscaler import Upscaler
            
            upscaler = Upscaler()
            
            # Test batch upscaling
            results = upscaler.batch_upscale("input_dir", "output_dir", scale=2.0)
            
            # Check that resize was called for each image
            assert mock_resize.call_count == 3
            
            # Check the results
            assert len(results) == 3
            assert all(path.startswith("output_dir") for path in results)
    
    def test_save_upscaled_image(self, mock_pil_image):
        """Test saving upscaled images."""
        with patch('PIL.Image.open', return_value=mock_pil_image), \
             patch('PIL.Image.resize', return_value=mock_pil_image), \
             patch.object(mock_pil_image, 'save') as mock_save:
            
            from utils.upscaler import Upscaler
            
            upscaler = Upscaler()
            
            # Test upscaling and saving
            output_path = upscaler.upscale("test_image.png", scale=2.0, output_path="upscaled.png")
            
            # Check that the image was saved
            mock_save.assert_called_once_with("upscaled.png")
            
            # Check the returned path
            assert output_path == "upscaled.png"
    
    def test_get_optimal_dimensions(self):
        """Test calculating optimal dimensions for upscaling."""
        from utils.upscaler import Upscaler
        
        upscaler = Upscaler()
        
        # Test with different original dimensions and scales
        test_cases = [
            # original_width, original_height, scale, expected_width, expected_height
            (1024, 1024, 2.0, 2048, 2048),
            (800, 600, 1.5, 1200, 900),
            (1920, 1080, 4.0, 7680, 4320),
            (512, 512, 0.5, 256, 256)  # Downscaling
        ]
        
        for orig_w, orig_h, scale, exp_w, exp_h in test_cases:
            width, height = upscaler.get_optimal_dimensions(orig_w, orig_h, scale)
            assert width == exp_w
            assert height == exp_h 