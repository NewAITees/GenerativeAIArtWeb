"""
Tests for the watermark utility.
This tests the functionality for adding watermarks to generated images.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image, ImageDraw, ImageFont

class TestWatermark:
    """Test class for watermark utility."""
    
    @pytest.fixture
    def mock_image_data(self):
        """Fixture providing mock image data as numpy array."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """Test that the Watermarker initializes correctly."""
        from utils.watermark import Watermarker
        
        # Initialize with default settings
        watermarker = Watermarker()
        
        # Check default properties
        assert watermarker.default_text == "Generated with AI"
        assert watermarker.default_opacity == 0.3
        assert watermarker.default_position == "bottom-right"
        
        # Test with custom settings
        watermarker = Watermarker(
            default_text="Custom Watermark",
            default_opacity=0.5,
            default_position="center"
        )
        
        # Check custom properties
        assert watermarker.default_text == "Custom Watermark"
        assert watermarker.default_opacity == 0.5
        assert watermarker.default_position == "center"
    
    def test_add_text_watermark(self, mock_pil_image):
        """Test adding a text watermark to an image."""
        with patch('PIL.Image.new') as mock_new_image, \
             patch('PIL.ImageDraw.Draw') as mock_draw, \
             patch('PIL.ImageFont.truetype') as mock_font, \
             patch('PIL.Image.alpha_composite') as mock_composite, \
             patch.object(mock_pil_image, 'convert') as mock_convert:
            
            # Configure mocks
            mock_watermark_img = MagicMock()
            mock_new_image.return_value = mock_watermark_img
            mock_draw_obj = MagicMock()
            mock_draw.return_value = mock_draw_obj
            mock_font_obj = MagicMock()
            mock_font.return_value = mock_font_obj
            mock_convert.return_value = mock_pil_image
            mock_composite.return_value = mock_pil_image
            
            from utils.watermark import Watermarker
            
            watermarker = Watermarker()
            
            # Test adding a text watermark
            result = watermarker.add_text_watermark(
                mock_pil_image,
                text="Test Watermark",
                position="bottom-right",
                opacity=0.4,
                color=(255, 255, 255),
                font_size=24
            )
            
            # Check that ImageDraw.Draw was called
            mock_draw.assert_called_once()
            
            # Check that text was drawn
            mock_draw_obj.text.assert_called_once()
            args, kwargs = mock_draw_obj.text.call_args
            
            # Check the text content
            assert "Test Watermark" in kwargs.values()
            
            # Check the font size
            mock_font.assert_called_once()
            args, _ = mock_font.call_args
            assert args[1] == 24
            
            # Check the color
            assert (255, 255, 255) in kwargs.values()
            
            # Check the result
            assert result is mock_pil_image
    
    def test_add_image_watermark(self, mock_pil_image):
        """Test adding an image watermark to an image."""
        with patch('PIL.Image.open') as mock_open_image, \
             patch('PIL.Image.new') as mock_new_image, \
             patch('PIL.Image.alpha_composite') as mock_composite, \
             patch.object(mock_pil_image, 'convert') as mock_convert:
            
            # Configure mocks
            mock_watermark_img = MagicMock()
            mock_watermark_img.resize.return_value = mock_watermark_img
            mock_open_image.return_value = mock_watermark_img
            mock_new_layer = MagicMock()
            mock_new_image.return_value = mock_new_layer
            mock_convert.return_value = mock_pil_image
            mock_composite.return_value = mock_pil_image
            
            from utils.watermark import Watermarker
            
            watermarker = Watermarker()
            
            # Test adding an image watermark
            result = watermarker.add_image_watermark(
                mock_pil_image,
                watermark_image_path="logo.png",
                position="bottom-right",
                opacity=0.4,
                scale=0.2
            )
            
            # Check that the watermark image was opened
            mock_open_image.assert_called_once_with("logo.png")
            
            # Check that the watermark was resized
            mock_watermark_img.resize.assert_called_once()
            
            # Check the result
            assert result is mock_pil_image
    
    def test_calculate_position(self):
        """Test calculating the position for a watermark."""
        from utils.watermark import Watermarker
        
        watermarker = Watermarker()
        
        # Test image and watermark dimensions
        img_width, img_height = 1024, 768
        wm_width, wm_height = 200, 100
        
        # Test positions
        positions = {
            "top-left": (0, 0),
            "top-right": (img_width - wm_width, 0),
            "bottom-left": (0, img_height - wm_height),
            "bottom-right": (img_width - wm_width, img_height - wm_height),
            "center": ((img_width - wm_width) // 2, (img_height - wm_height) // 2),
        }
        
        for position_name, expected_coords in positions.items():
            coords = watermarker._calculate_position(
                img_width, img_height, 
                wm_width, wm_height, 
                position_name
            )
            assert coords == expected_coords
    
    def test_batch_add_watermark(self, mock_pil_image):
        """Test batch adding watermarks to multiple images."""
        with patch('PIL.Image.open', return_value=mock_pil_image), \
             patch('utils.watermark.Watermarker.add_text_watermark', return_value=mock_pil_image) as mock_add_watermark, \
             patch.object(mock_pil_image, 'save') as mock_save, \
             patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=["img1.png", "img2.png", "img3.png"]), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            
            from utils.watermark import Watermarker
            
            watermarker = Watermarker()
            
            # Test batch adding watermarks
            results = watermarker.batch_add_watermark(
                input_dir="input",
                output_dir="output",
                text="Batch Watermark",
                position="bottom-right",
                opacity=0.3
            )
            
            # Check that add_text_watermark was called for each image
            assert mock_add_watermark.call_count == 3
            
            # Check that save was called for each image
            assert mock_save.call_count == 3
            
            # Check the results
            assert len(results) == 3
            assert all("output" in path for path in results)
    
    def test_remove_watermark(self, mock_pil_image):
        """Test removal of watermark from an image."""
        with patch('utils.watermark.remove_watermark_impl', return_value=mock_pil_image) as mock_remove:
            
            from utils.watermark import Watermarker
            
            watermarker = Watermarker()
            
            # Test removing a watermark
            result = watermarker.remove_watermark(mock_pil_image)
            
            # Check that the removal function was called
            mock_remove.assert_called_once_with(mock_pil_image)
            
            # Check the result
            assert result is mock_pil_image
    
    def test_get_font(self):
        """Test getting a font for watermarking."""
        with patch('PIL.ImageFont.truetype') as mock_font:
            mock_font_obj = MagicMock()
            mock_font.return_value = mock_font_obj
            
            from utils.watermark import Watermarker
            
            watermarker = Watermarker()
            
            # Test getting a font
            font = watermarker._get_font(24)
            
            # Check that the font was created
            assert font is mock_font_obj
            
            # Check the font size
            mock_font.assert_called_once()
            args, _ = mock_font.call_args
            assert args[1] == 24 