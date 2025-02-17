import pytest
from core.rag import BookmarkRAG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_query():
    """Test RAG with a real query using Mistral"""
    # Initialize RAG
    rag = BookmarkRAG()
    
    # Test query
    query = "What Python tutorials do I have in my bookmarks?"
    response = rag.chat(
        user_input=query,
        temperature=0.7
    )
    
    # Verify response structure
    assert isinstance(response, dict)
    assert 'response' in response
    assert 'sources' in response
    assert 'metadata' in response
    
    # Print for manual inspection
    print("\nQuery:", query)
    print("\nResponse:", response['response'])
    print("\nSources:", response['sources'])
    print("\nMetadata:", response['metadata'])

if __name__ == "__main__":
    pytest.main([__file__, '-v'])