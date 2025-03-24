"""
This file contains pytest fixtures that can be used across all tests.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state before each test."""
    # Pre-test setup
    yield
    # Post-test cleanup

@pytest.fixture(autouse=True)
def reset_all_mocks():
    """Reset all mocks before and after each test."""
    mocks = []
    
    # LLM Generator mocks
    llm_patcher = patch('src.prompt.llm_generator.LLMPromptGenerator')
    llm_mock = llm_patcher.start()
    llm_mock.return_value.model_name = 'llama3'
    llm_mock.return_value.initialized = True
    llm_mock.return_value.generate_prompt.return_value = "A photorealistic image of a cat with detailed fur"
    mocks.append(llm_mock)
    
    # JSON Builder mocks
    json_patcher = patch('src.prompt.json_builder.JSONPromptBuilder')
    json_mock = json_patcher.start()
    template = {
        "subjects": ["cat", "dog", "bird"],
        "styles": ["photorealistic", "anime", "oil painting"],
        "qualities": ["detailed", "masterpiece", "best quality"]
    }
    json_mock.return_value.template = template
    json_mock.return_value.build_prompt.return_value = "A detailed photorealistic cat"
    json_mock.return_value.random_prompt.return_value = "A mysterious cat in moonlight"
    json_mock.return_value.build_prompt_with_weights.return_value = "A masterpiece of a cat (style:1.2)"
    mocks.append(json_mock)
    
    # Web interface mocks
    gradio_patcher = patch('src.web.app.gr')
    gradio_mock = gradio_patcher.start()
    mocks.append(gradio_mock)
    
    yield
    
    # Reset all mocks after test
    for mock in mocks:
        mock.reset_mock()
    
    # Stop all patchers
    llm_patcher.stop()
    json_patcher.stop()
    gradio_patcher.stop()

@pytest.fixture
def mock_llm_generator():
    """Provide a fresh LLMPromptGenerator mock for each test."""
    with patch('src.prompt.llm_generator.LLMPromptGenerator') as mock:
        mock.return_value.model_name = 'llama3'
        mock.return_value.initialized = True
        mock.return_value.client = MagicMock()
        mock.return_value.client.generate.return_value = {
            'response': 'A photorealistic image of a cat with detailed fur',
            'model': 'llama3',
            'created_at': '2025-03-23T12:34:56Z',
            'done': True
        }
        yield mock

@pytest.fixture
def mock_json_builder():
    """Provide a fresh JSONPromptBuilder mock for each test."""
    with patch('src.prompt.json_builder.JSONPromptBuilder') as mock:
        template = {
            "subjects": ["cat", "dog", "bird"],
            "styles": ["photorealistic", "anime", "oil painting"],
            "qualities": ["detailed", "masterpiece", "best quality"]
        }
        mock.return_value.template = template
        mock.return_value.build_prompt.return_value = "A detailed photorealistic cat"
        mock.return_value.random_prompt.return_value = "A mysterious cat in moonlight"
        mock.return_value.build_prompt_with_weights.return_value = "A masterpiece of a cat (style:1.2)"
        yield mock

# Async support
@pytest.fixture
async def async_mock():
    """Base fixture for async mocks."""
    async def async_magic():
        return Mock()
    return async_magic 