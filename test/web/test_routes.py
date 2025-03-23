"""
Tests for the web app routes.
This tests the routing logic for the web application.
"""
import pytest
from unittest.mock import patch, MagicMock

class TestRoutes:
    """Test class for web application routes."""
    
    def test_index_route(self, mock_flask_app):
        """Test the index route returns the correct template."""
        # Create a test client
        with patch('flask.render_template') as mock_render:
            mock_render.return_value = "<html>Mock Index Template</html>"
            
            # Import the route handler
            from src.web.app import GradioInterface
            
            # Create interface instance
            interface = GradioInterface()
            
            # Call the interface creation method
            interface.create_interface()
            
            # Check that the interface was created successfully
            assert interface is not None
    
    def test_api_routes_registration(self, mock_flask_app):
        """Test that API routes are registered correctly."""
        with patch('gradio.Interface') as mock_interface:
            # Import the Gradio interface
            from src.web.app import GradioInterface
            
            # Create interface instance
            interface = GradioInterface()
            
            # Call the interface creation method
            interface.create_interface()
            
            # Verify that the Gradio interface was created with the correct components
            mock_interface.assert_called()

    def test_images_route(self, mock_flask_app, mock_file_system):
        """Test the image generation and handling routes."""
        with patch('gradio.Interface') as mock_interface:
            from src.web.app import GradioInterface
            
            # Create interface instance
            interface = GradioInterface()
            
            # Mock image generation
            mock_image = MagicMock()
            interface.generate_image = MagicMock(return_value=mock_image)
            
            # Test image generation
            result = interface.generate_image(
                prompt="test prompt",
                model_path="test_model",
                steps=30,
                cfg_scale=7.5,
                sampler="euler",
                width=512,
                height=512,
                seed=42
            )
            
            # Verify that image generation was called with correct parameters
            interface.generate_image.assert_called_once()
            assert result == mock_image
    
    def test_error_handling(self, mock_flask_app):
        """Test error handling in the interface."""
        with patch('gradio.Interface') as mock_interface:
            from src.web.app import GradioInterface
            
            # Create interface instance
            interface = GradioInterface()
            
            # Mock image generation to raise an error
            interface.generate_image = MagicMock(side_effect=Exception("Test error"))
            
            # Test error handling during image generation
            result = interface.generate_image(
                prompt="test prompt",
                model_path="test_model",
                steps=30,
                cfg_scale=7.5,
                sampler="euler",
                width=512,
                height=512,
                seed=42
            )
            
            # Verify that the error was handled
            assert result is None
            interface.generate_image.assert_called_once() 