import pytest
from core.rag import BookmarkRAG
from unittest.mock import Mock, patch
import json

@pytest.fixture
def mock_llama():
    """Create a mock Llama instance"""
    mock = Mock()
    mock.create_completion.return_value = {
        'choices': [{
            'text': 'This is a test response about Python tutorials.',
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 50,
            'completion_tokens': 20
        }
    }
    return mock

@pytest.fixture
def rag_with_mock_llm(mock_llama):
    """Create RAG instance with mock LLM"""
    with patch('core.rag.Llama', return_value=mock_llama):
        rag = BookmarkRAG()
        return rag, mock_llama

def test_llm_response_generation(rag_with_mock_llm):
    """Test LLM response generation"""
    rag, mock_llama = rag_with_mock_llm
    
    response = rag._generate_response(
        system_prompt="What Python tutorials do I have?",
        temperature=0.7,
        max_tokens=500
    )
    
    assert isinstance(response, dict)
    assert 'text' in response
    assert 'confidence' in response
    assert 'model_info' in response
    assert mock_llama.create_completion.called