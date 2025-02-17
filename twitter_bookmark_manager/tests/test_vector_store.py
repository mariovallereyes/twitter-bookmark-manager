import pytest
from database.vector_store import ChromaStore
from unittest.mock import Mock, patch
import chromadb

@pytest.fixture
def mock_chroma_client():
    """Create a mock Chroma client"""
    mock_client = Mock()
    mock_collection = Mock()
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client, mock_collection

@pytest.fixture
def vector_store(mock_chroma_client):
    """Create a vector store instance with mocked client"""
    mock_client, _ = mock_chroma_client
    with patch('database.vector_store.Client', return_value=mock_client):
        store = ChromaStore(persist_directory="./test_vector_db")
        return store

def test_initialization(vector_store):
    """Test vector store initialization"""
    assert vector_store is not None
    assert vector_store.collection is not None

def test_add_bookmark(vector_store, mock_chroma_client):
    """Test adding a bookmark"""
    _, mock_collection = mock_chroma_client
    
    vector_store.add_bookmark(
        bookmark_id="test-id-1",
        text="Python tutorial about async programming",
        metadata={"url": "https://twitter.com/user/status/1"}
    )
    
    mock_collection.add.assert_called_once()
    call_args = mock_collection.add.call_args[1]
    assert "test-id-1" in call_args["ids"]
    assert "Python tutorial" in call_args["documents"][0]

def test_search(vector_store, mock_chroma_client):
    """Test searching bookmarks"""
    _, mock_collection = mock_chroma_client
    
    # Mock search results
    mock_collection.query.return_value = {
        'ids': [["test-id-1"]],
        'documents': [["Python tutorial"]],
        'metadatas': [[{"url": "https://twitter.com/user/status/1"}]],
        'distances': [[0.8]]
    }
    
    results = vector_store.search("python tutorial", n_results=1)
    
    assert len(results) == 1
    assert results[0]["bookmark_id"] == "test-id-1"
    assert "Python tutorial" in results[0]["text"]
    assert results[0]["distance"] == 0.8

def test_delete_bookmark(vector_store, mock_chroma_client):
    """Test deleting a bookmark"""
    _, mock_collection = mock_chroma_client
    
    vector_store.delete_bookmark("test-id-1")
    
    mock_collection.delete.assert_called_once_with(ids=["test-id-1"])

def test_get_collection_stats(vector_store, mock_chroma_client):
    """Test getting collection statistics"""
    _, mock_collection = mock_chroma_client
    mock_collection.count.return_value = 5
    mock_collection.name = "bookmarks"
    
    stats = vector_store.get_collection_stats()
    
    assert stats["count"] == 5
    assert stats["name"] == "bookmarks"

def test_error_handling(vector_store, mock_chroma_client):
    """Test error handling"""
    _, mock_collection = mock_chroma_client
    mock_collection.add.side_effect = Exception("Test error")
    
    with pytest.raises(Exception):
        vector_store.add_bookmark("test-id", "test text")