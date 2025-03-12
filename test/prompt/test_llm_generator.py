"""
Tests for the LLM-based prompt generator.
This tests the functionality for generating and enhancing prompts using Large Language Models.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open

class TestLLMPromptGenerator:
    """Test class for LLM-based prompt generator."""
    
    def test_initialization(self, mock_llm):
        """Test that the LLMPromptGenerator initializes correctly."""
        with patch('prompt.llm_generator.ollama') as mock_ollama:
            mock_ollama.Client.return_value = mock_llm
            
            from prompt.llm_generator import LLMPromptGenerator
            
            # Initialize with default settings
            generator = LLMPromptGenerator()
            
            # Check that ollama Client was initialized
            mock_ollama.Client.assert_called_once()
            
            # Test with custom model
            generator = LLMPromptGenerator(model_name="llama3")
            
            # Check custom properties
            assert generator.model_name == "llama3"
    
    def test_enhance_prompt(self, mock_llm):
        """Test prompt enhancement with LLM."""
        with patch('prompt.llm_generator.ollama') as mock_ollama:
            mock_ollama.Client.return_value = mock_llm
            mock_llm.generate.return_value = {
                'response': 'a stunning photograph of a cat with detailed fur, warm lighting, shallow depth of field, shot on a Canon EOS R5 with 85mm f/1.2 lens'
            }
            
            from prompt.llm_generator import LLMPromptGenerator
            
            generator = LLMPromptGenerator()
            
            # Test basic prompt enhancement
            enhanced = generator.enhance_prompt("cat")
            
            # Check that the LLM was called with appropriate system message and prompt
            mock_llm.generate.assert_called_once()
            call_args = mock_llm.generate.call_args[1]
            
            # Verify the prompt contains the original text
            assert "cat" in call_args['prompt']
            # Verify the system message instructs for detailed prompt creation
            assert "image generation prompt" in call_args['system']
            
            # Check the enhanced output
            assert "stunning photograph" in enhanced
            assert "Canon EOS" in enhanced
            
    def test_enhance_prompt_with_style(self, mock_llm):
        """Test prompt enhancement with specific style."""
        with patch('prompt.llm_generator.ollama') as mock_ollama:
            mock_ollama.Client.return_value = mock_llm
            mock_llm.generate.return_value = {
                'response': 'a watercolor painting of a cat, soft brushstrokes, vibrant colors, in the style of traditional Japanese art'
            }
            
            from prompt.llm_generator import LLMPromptGenerator
            
            generator = LLMPromptGenerator()
            
            # Test style-specific enhancement
            enhanced = generator.enhance_prompt("cat", style="watercolor")
            
            # Check that style was included in the prompt
            call_args = mock_llm.generate.call_args[1]
            assert "watercolor" in call_args['prompt']
            
            # Check the output includes style elements
            assert "watercolor painting" in enhanced
            assert "brushstrokes" in enhanced

    def test_batch_enhance(self, mock_llm):
        """Test batch enhancement of multiple prompts."""
        with patch('prompt.llm_generator.ollama') as mock_ollama:
            mock_ollama.Client.return_value = mock_llm
            
            # Different responses for different calls
            mock_llm.generate.side_effect = [
                {'response': 'enhanced cat prompt'},
                {'response': 'enhanced dog prompt'},
                {'response': 'enhanced bird prompt'}
            ]
            
            from prompt.llm_generator import LLMPromptGenerator
            
            generator = LLMPromptGenerator()
            
            # Test batch enhancement
            prompts = ["cat", "dog", "bird"]
            enhanced_prompts = generator.batch_enhance(prompts)
            
            # Check that the LLM was called correct number of times
            assert mock_llm.generate.call_count == 3
            
            # Check the results
            assert len(enhanced_prompts) == 3
            assert enhanced_prompts[0] == 'enhanced cat prompt'
            assert enhanced_prompts[1] == 'enhanced dog prompt'
            assert enhanced_prompts[2] == 'enhanced bird prompt'
    
    def test_save_and_load_prompts(self, mock_llm):
        """Test saving and loading enhanced prompts."""
        with patch('prompt.llm_generator.ollama') as mock_ollama, \
             patch('builtins.open', mock_open()) as mock_file:
            
            mock_ollama.Client.return_value = mock_llm
            
            from prompt.llm_generator import LLMPromptGenerator
            
            generator = LLMPromptGenerator()
            
            # Mock data to save
            prompts_data = {
                "original": "cat",
                "enhanced": "a beautiful cat with detailed fur"
            }
            
            # Test saving prompts
            with patch('json.dump') as mock_json_dump:
                generator.save_prompt(prompts_data, "saved_prompts.json")
                
                # Check that file was opened for writing
                mock_file.assert_called_once_with("saved_prompts.json", "w")
                
                # Check that json.dump was called with the correct data
                mock_json_dump.assert_called_once()
                args, _ = mock_json_dump.call_args
                assert args[0] == prompts_data
            
            # Test loading prompts
            with patch('json.load', return_value=prompts_data) as mock_json_load:
                loaded_data = generator.load_prompt("saved_prompts.json")
                
                # Check that json.load was called
                mock_json_load.assert_called_once()
                
                # Check the loaded data
                assert loaded_data == prompts_data
                assert loaded_data["original"] == "cat"
                assert loaded_data["enhanced"] == "a beautiful cat with detailed fur" 