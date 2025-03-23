"""
Tests for the JSON-based prompt builder.
This tests the functionality for building prompts by combining elements from JSON templates.
"""
import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open

class TestJSONPromptBuilder:
    """Test class for JSON-based prompt builder."""
    
    @pytest.fixture
    def sample_template(self):
        """Fixture providing a sample JSON template."""
        return {
            "subjects": [
                "cat", "dog", "bird", "mountain", "ocean", "forest"
            ],
            "styles": [
                "photorealistic", "anime", "oil painting", "watercolor", "digital art"
            ],
            "qualities": [
                "detailed", "high quality", "masterpiece", "best quality", "intricate"
            ],
            "lighting": [
                "natural lighting", "golden hour", "studio lighting", "dramatic lighting"
            ],
            "cameras": [
                "Canon EOS R5", "Nikon Z9", "Sony A7R IV", "Hasselblad X1D"
            ],
            "lenses": [
                "85mm f/1.2", "50mm f/1.4", "35mm f/1.8", "24-70mm f/2.8"
            ]
        }
    
    def test_initialization(self, sample_template):
        """Test that the JSONPromptBuilder initializes correctly."""
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_template))):
            from prompt.json_builder import JSONPromptBuilder
            
            # Initialize with default path
            builder = JSONPromptBuilder()
            
            # Check that the template was loaded
            assert hasattr(builder, 'template')
            assert len(builder.template) > 0
            
            # Check that the categories match
            assert "subjects" in builder.template
            assert "styles" in builder.template
            assert len(builder.template["subjects"]) == 6
    
    def test_build_prompt_simple(self, sample_template):
        """Test building a simple prompt by selecting elements."""
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_template))):
            from prompt.json_builder import JSONPromptBuilder
            
            builder = JSONPromptBuilder()
            
            # Build a simple prompt with just subject and style
            prompt = builder.build_prompt(subject="cat", style="oil painting")
            
            # Check the prompt format
            assert "cat" in prompt
            assert "oil painting" in prompt
    
    def test_build_prompt_with_all_elements(self, sample_template):
        """Test building a comprehensive prompt with all elements."""
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_template))):
            from prompt.json_builder import JSONPromptBuilder
            
            builder = JSONPromptBuilder()
            
            # Build a prompt with all possible elements
            prompt = builder.build_prompt(
                subject="mountain",
                style="photorealistic",
                quality="masterpiece",
                lighting="golden hour",
                camera="Canon EOS R5",
                lens="24-70mm f/2.8"
            )
            
            # Check that all elements are in the prompt
            assert "mountain" in prompt
            assert "photorealistic" in prompt
            assert "masterpiece" in prompt
            assert "golden hour" in prompt
            assert "Canon EOS R5" in prompt
            assert "24-70mm f/2.8" in prompt
    
    def test_random_prompt(self, sample_template):
        """Test generating a random prompt."""
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_template))), \
             patch('random.choice') as mock_choice:
            
            # Configure mock_choice to return specific values
            mock_choice.side_effect = ["cat", "watercolor", "detailed", "natural lighting"]
            
            from prompt.json_builder import JSONPromptBuilder
            
            builder = JSONPromptBuilder()
            
            # Generate a random prompt
            prompt = builder.random_prompt(include_categories=["subjects", "styles", "qualities", "lighting"])
            
            # Check that the random elements were included
            assert "cat" in prompt
            assert "watercolor" in prompt
            assert "detailed" in prompt
            assert "natural lighting" in prompt
            
            # Check that random.choice was called the expected number of times
            assert mock_choice.call_count == 4
    
    def test_prompt_with_weights(self, sample_template):
        """Test building a prompt with weighted elements."""
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_template))):
            from prompt.json_builder import JSONPromptBuilder
            
            builder = JSONPromptBuilder()
            
            # Build a prompt with weights for certain elements
            prompt = builder.build_prompt_with_weights(
                elements={
                    "subject": "cat",
                    "style": "digital art",
                    "quality": "masterpiece"
                },
                weights={
                    "style": 1.2,
                    "quality": 1.5
                }
            )
            
            # Check that the elements are in the prompt
            assert "cat" in prompt
            assert "digital art" in prompt
            assert "masterpiece" in prompt
            
            # Check that weights are correctly formatted
            assert "(digital art:1.2)" in prompt or "(1.2:digital art)" in prompt
            assert "(masterpiece:1.5)" in prompt or "(1.5:masterpiece)" in prompt
    
    def test_save_and_load_custom_template(self, sample_template):
        """Test saving and loading a custom template."""
        # Mock for file opening operations
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file):
            from prompt.json_builder import JSONPromptBuilder
            
            # First, mock loading the default template
            mock_file.return_value.read.return_value = json.dumps(sample_template)
            
            builder = JSONPromptBuilder()
            
            # Add a new category to the template
            builder.template["moods"] = ["happy", "sad", "mysterious", "romantic"]
            
            # Save the modified template
            with patch('json.dump') as mock_json_dump:
                builder.save_template("custom_template.json")
                
                # Check that file was opened for writing
                mock_file.assert_called_with("custom_template.json", "w")
                
                # Check that json.dump was called with the modified template
                mock_json_dump.assert_called_once()
                args, _ = mock_json_dump.call_args
                assert "moods" in args[0]
                assert args[0]["moods"] == ["happy", "sad", "mysterious", "romantic"]
            
            # Test loading the custom template
            custom_template = sample_template.copy()
            custom_template["moods"] = ["happy", "sad", "mysterious", "romantic"]
            
            mock_file.return_value.read.return_value = json.dumps(custom_template)
            
            builder = JSONPromptBuilder(template_path="custom_template.json")
            
            # Check that the custom template was loaded
            assert "moods" in builder.template
            assert builder.template["moods"] == ["happy", "sad", "mysterious", "romantic"] 