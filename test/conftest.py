"""
This file contains pytest fixtures that can be used across all tests.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the src directory to the path so we can import modules from there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock for the Stable Diffusion 3.5 model
@pytest.fixture
def mock_sd3_model():
    mock = MagicMock()
    mock.generate.return_value = MagicMock()  # Mock generated image
    return mock

# Mock for the T5 text encoder
@pytest.fixture
def mock_t5_model():
    mock = MagicMock()
    mock.encode_text.return_value = MagicMock()  # Mock text embeddings
    return mock

# Mock for the CLIP model
@pytest.fixture
def mock_clip_model():
    mock = MagicMock()
    mock.encode_image.return_value = MagicMock()  # Mock image embeddings
    mock.encode_text.return_value = MagicMock()   # Mock text embeddings
    return mock

# Mock for a PIL Image
@pytest.fixture
def mock_pil_image():
    mock = MagicMock()
    mock.size = (1024, 1024)
    mock.save.return_value = None
    return mock

# Mock for Flask app
@pytest.fixture
def mock_flask_app():
    with patch('flask.Flask') as mock_flask:
        app_instance = MagicMock()
        mock_flask.return_value = app_instance
        yield app_instance

# Mock for Gradio app
@pytest.fixture
def mock_gradio_app():
    with patch('gradio.Blocks') as mock_gradio:
        app_instance = MagicMock()
        mock_gradio.return_value = app_instance
        yield app_instance

# Mock for LLM (e.g., ollama)
@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.generate.return_value = "Enhanced prompt with artistic details"
    return mock

# Mock for file system operations
@pytest.fixture
def mock_file_system():
    with patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs, \
         patch('os.listdir') as mock_listdir:
        mock_exists.return_value = True
        mock_makedirs.return_value = None
        mock_listdir.return_value = ["file1.png", "file2.png"]
        yield {
            'exists': mock_exists,
            'makedirs': mock_makedirs,
            'listdir': mock_listdir
        } 