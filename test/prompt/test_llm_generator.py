"""
Tests for the LLM-based prompt generator.
This tests the functionality for generating and enhancing prompts using Large Language Models.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open

@pytest.fixture
def mock_ollama_client():
    """Fixture providing a mock ollama client."""
    with patch('ollama.Client') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        
        # Mock the list method to return available models
        client_instance.list.return_value = {
            'models': [
                {'name': 'llama3'},
                {'name': 'codellama'},
                {'name': 'mistral'}
            ]
        }
        
        # Mock the generate method
        client_instance.generate.return_value = {
            'response': 'A photorealistic image of a cat with detailed fur, sitting in a sunlit garden',
            'model': 'llama3',
            'created_at': '2025-03-23T12:34:56Z',
            'done': True
        }
        
        yield client_instance

class TestLLMPromptGenerator:
    """Test class for LLM-based prompt generator."""
    
    def test_initialization(self, mock_ollama_client):
        """Test that the LLMPromptGenerator initializes correctly."""
        from src.prompt.llm_generator import LLMPromptGenerator
        
        # Initialize with default settings
        generator = LLMPromptGenerator()
        
        # Check default properties
        assert generator.model_name == "llama3"
        assert generator.initialized == True
        
        # Test with custom model
        generator = LLMPromptGenerator(model_name="mistral")
        
        # Check custom properties
        assert generator.model_name == "mistral"
    
    def test_generate_prompt(self, mock_ollama_client):
        """Test prompt generation with LLM."""
        from src.prompt.llm_generator import LLMPromptGenerator
        
        generator = LLMPromptGenerator()
        
        # Test basic prompt generation
        enhanced = generator.generate_prompt("cat")
        
        # Check that the LLM was called with appropriate params
        mock_ollama_client.generate.assert_called_once()
        call_kwargs = mock_ollama_client.generate.call_args[1]
        
        # Verify the prompt contains the original text
        assert "cat" in call_kwargs['prompt']
        # Verify the system message instructs for detailed prompt creation
        assert "Stable Diffusion" in call_kwargs['system']
        
        # Check the enhanced output
        assert enhanced == "A photorealistic image of a cat with detailed fur, sitting in a sunlit garden"
    
    def test_generate_prompt_with_style(self, mock_ollama_client):
        """Test prompt enhancement with specific style."""
        from src.prompt.llm_generator import LLMPromptGenerator
        
        generator = LLMPromptGenerator()
        
        # Configure mock to return a different response for style prompt
        mock_ollama_client.generate.return_value = {
            'response': 'A watercolor painting of a cat, soft brushstrokes, vibrant colors',
        }
        
        # Test style-specific enhancement
        enhanced = generator.generate_prompt("cat", style="watercolor")
        
        # Check that the LLM was called with style in prompt
        call_kwargs = mock_ollama_client.generate.call_args[1]
        assert "watercolor" in call_kwargs['prompt']
        
        # Check the output includes style elements
        assert "watercolor painting" in enhanced
        assert "brushstrokes" in enhanced
    
    def test_batch_enhance(self, mock_ollama_client):
        """Test batch enhancement of multiple prompts."""
        from src.prompt.llm_generator import LLMPromptGenerator
        
        generator = LLMPromptGenerator()
        
        # Configure mock to return different responses for different calls
        mock_ollama_client.generate.side_effect = [
            {'response': 'enhanced cat prompt'},
            {'response': 'enhanced dog prompt'},
            {'response': 'enhanced bird prompt'}
        ]
        
        # Test batch enhancement
        prompts = ["cat", "dog", "bird"]
        enhanced_prompts = generator.batch_enhance(prompts)
        
        # Check that the LLM was called correct number of times
        assert mock_ollama_client.generate.call_count == 3
        
        # Check the results
        assert len(enhanced_prompts) == 3
        assert enhanced_prompts[0] == 'enhanced cat prompt'
        assert enhanced_prompts[1] == 'enhanced dog prompt'
        assert enhanced_prompts[2] == 'enhanced bird prompt'
    
    def test_client_initialization_failure(self):
        """Test handling of client initialization failure."""
        with patch('ollama.Client', side_effect=Exception("Connection failed")):
            from src.prompt.llm_generator import LLMPromptGenerator
            
            generator = LLMPromptGenerator()
            
            # Check that initialization failed gracefully
            assert generator.initialized == False
            assert generator.client is None
            
            # Test that mock response is returned when client fails
            result = generator.generate_prompt("cat")
            assert result != ""
            assert "detailed image" in result or "photorealistic" in result
    
    def test_model_not_available(self):
        """Test handling of unavailable model."""
        with patch('ollama.Client') as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance
            
            # Mock list to return models without the requested one
            client_instance.list.return_value = {
                'models': [
                    {'name': 'codellama'},
                    {'name': 'mistral'}
                ]
            }
            
            from src.prompt.llm_generator import LLMPromptGenerator
            
            # Initialize with a model that's not available
            generator = LLMPromptGenerator(model_name="llama3")
            
            # Check that initialization detected the missing model
            assert generator.initialized == False
            
            # Test that mock response is returned for unavailable model
            result = generator.generate_prompt("cat")
            assert result != ""
            assert "detailed image" in result or "photorealistic" in result
    
    def test_save_and_load_prompts(self, mock_ollama_client):
        """Test saving and loading enhanced prompts."""
        with patch('builtins.open', mock_open()) as mock_file:
            from src.prompt.llm_generator import LLMPromptGenerator
            
            generator = LLMPromptGenerator()
            
            # Mock data to save
            prompts_data = {
                "original": "cat",
                "enhanced": "a beautiful cat with detailed fur"
            }
            
            # Test saving prompts
            with patch('json.dump') as mock_json_dump:
                success = generator.save_prompt(prompts_data, "saved_prompts.json")
                
                # Check that file was opened for writing
                mock_file.assert_called_once_with("saved_prompts.json", "w", encoding="utf-8")
                
                # Check that json.dump was called with the correct data
                mock_json_dump.assert_called_once()
                args, _ = mock_json_dump.call_args
                assert args[0] == prompts_data
                
                # Check return value
                assert success == True
            
            # Test loading prompts
            with patch('json.load', return_value=prompts_data) as mock_json_load:
                loaded_data = generator.load_prompt("saved_prompts.json")
                
                # Check that json.load was called
                mock_json_load.assert_called_once()
                
                # Check the loaded data
                assert loaded_data == prompts_data
                assert loaded_data["original"] == "cat"
                assert loaded_data["enhanced"] == "a beautiful cat with detailed fur" 