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
            from web.routes import index
            
            # Call the index route
            response = index()
            
            # Check that render_template was called with the correct template
            mock_render.assert_called_once_with('index.html')
            assert response == "<html>Mock Index Template</html>"
    
    def test_api_routes_registration(self, mock_flask_app):
        """Test that API routes are registered correctly."""
        # Mock the Flask app and Blueprint
        mock_blueprint = MagicMock()
        
        with patch('flask.Blueprint', return_value=mock_blueprint) as mock_bp:
            # Import the register_routes function
            from web.routes import register_routes
            
            # Call the function to register routes
            register_routes(mock_flask_app)
            
            # Check that Blueprint was created
            mock_bp.assert_called_once_with('api', __name__, url_prefix='/api')
            
            # Check that routes were added to the blueprint
            # Here we check for some expected routes
            route_calls = [call[0][0] for call in mock_blueprint.route.call_args_list]
            
            assert '/generate' in route_calls
            assert '/enhance_prompt' in route_calls
            assert '/save_settings' in route_calls
            assert '/load_settings' in route_calls
            
            # Check that the blueprint was registered with the app
            mock_flask_app.register_blueprint.assert_called_once_with(mock_blueprint)
    
    def test_static_routes(self, mock_flask_app):
        """Test that static routes serve the correct files."""
        with patch('flask.send_from_directory') as mock_send:
            mock_send.return_value = "mock_file_content"
            
            # Import the static file route handler
            from web.routes import serve_static
            
            # Call the route handler for a CSS file
            response = serve_static('css/style.css')
            
            # Check that send_from_directory was called correctly
            mock_send.assert_called_once()
            args, kwargs = mock_send.call_args
            # The static directory path should be passed
            assert 'static' in args[0]
            # The filepath should be passed
            assert args[1] == 'css/style.css'
            
            assert response == "mock_file_content"
    
    def test_images_route(self, mock_flask_app, mock_file_system):
        """Test the route for serving generated images."""
        with patch('flask.send_from_directory') as mock_send:
            mock_send.return_value = "mock_image_data"
            
            # Import the image serving route handler
            from web.routes import serve_image
            
            # Call the route handler for an image
            response = serve_image('generated_image.png')
            
            # Check that send_from_directory was called correctly
            mock_send.assert_called_once()
            args, kwargs = mock_send.call_args
            # The output directory path should be passed
            assert 'outputs' in args[0]
            # The image filename should be passed
            assert args[1] == 'generated_image.png'
            
            assert response == "mock_image_data"
    
    def test_404_handler(self, mock_flask_app):
        """Test the 404 error handler."""
        with patch('flask.render_template') as mock_render:
            mock_render.return_value = "<html>404 Not Found</html>"
            
            # Import the error handler
            from web.routes import page_not_found
            
            # Call the error handler
            response, status_code = page_not_found(Exception("Not Found"))
            
            # Check that render_template was called with the error template
            mock_render.assert_called_once_with('error.html', error_code=404, message='Page not found')
            assert response == "<html>404 Not Found</html>"
            assert status_code == 404 