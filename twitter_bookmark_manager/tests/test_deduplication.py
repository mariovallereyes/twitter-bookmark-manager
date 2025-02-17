import pytest
from core.deduplication import BookmarkDeduplicator
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store"""
    mock_store = Mock()
    # Set up default mock responses
    mock_store.collection.get.return_value = {
        'embeddings': [[0.1] * 384],  # List of 384 elements
        'metadatas': [{'text': 'test tweet'}]
    }
    # Use small distance to ensure high similarity
    mock_store.query_similar.return_value = {
        'ids': [['1', '2']],
        'distances': [[0.01, 0.5]],  # Will convert to high similarity
        'metadatas': [[
            {'text': 'similar tweet 1'},
            {'text': 'different tweet'}
        ]]
    }
    return mock_store

@pytest.fixture
def deduplicator(mock_vector_store):
    """Create a BookmarkDeduplicator instance with mocked vector store"""
    with patch('core.deduplication.get_vector_store', return_value=mock_vector_store):
        return BookmarkDeduplicator(similarity_threshold=0.95)

def test_initialization(deduplicator):
    """Test deduplicator initialization"""
    assert deduplicator.similarity_threshold == 0.95
    assert deduplicator.vector_store is not None

def test_find_potential_duplicates(deduplicator, mock_vector_store):
    """Test finding potential duplicates"""
    results = deduplicator.find_potential_duplicates('test_id')
    
    # Should only return the first result (0.9 similarity)
    assert len(results) == 1
    assert results[0]['similarity'] >= 0.95
    assert results[0]['requires_confirmation'] is True
    
    # Verify mock was called correctly
    mock_vector_store.collection.get.assert_called_once_with(
        ids=['test_id'],
        include=['embeddings', 'metadatas']
    )

@patch('core.deduplication.get_db_session')
def test_mark_as_duplicate(mock_get_session, deduplicator):
    """Test marking a bookmark as duplicate"""
    # Mock session and bookmark
    mock_session = Mock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    
    mock_bookmark = Mock()
    mock_session.query().get.return_value = mock_bookmark
    
    result = deduplicator.mark_as_duplicate(
        original_id='1',
        duplicate_id='2',
        similarity=0.95,
        user_confirmed=True
    )
    
    assert result['original_id'] == '1'
    assert result['duplicate_id'] == '2'
    assert result['similarity'] == 0.95
    assert result['user_confirmed'] is True
    assert mock_bookmark.is_duplicate is True
    assert mock_bookmark.user_confirmed is True

@patch('core.deduplication.get_db_session')
def test_mark_as_not_duplicate(mock_get_session, deduplicator):
    """Test marking a bookmark as not duplicate"""
    # Mock session and bookmark
    mock_session = Mock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    
    mock_bookmark = Mock()
    mock_session.query().get.return_value = mock_bookmark
    
    result = deduplicator.mark_as_not_duplicate(
        original_id='1',
        potential_duplicate_id='2'
    )
    
    assert result['original_id'] == '1'
    assert result['potential_duplicate_id'] == '2'
    assert result['status'] == 'marked_as_not_duplicate'
    assert mock_bookmark.is_duplicate is False
    assert mock_bookmark.user_confirmed is True

def test_process_bookmark(deduplicator, mock_vector_store):
    """Test processing a single bookmark"""
    result = deduplicator.process_bookmark('test_id')
    
    assert result['bookmark_id'] == 'test_id'
    assert result['requires_user_confirmation'] is True
    assert len(result['potential_duplicates']) > 0
    
    # Verify mock was called
    mock_vector_store.collection.get.assert_called_with(
        ids=['test_id'],
        include=['embeddings', 'metadatas']
    )

def test_batch_processing(deduplicator):
    """Test batch processing of bookmarks"""
    bookmark_ids = ['1', '2', '3']
    
    with patch.object(deduplicator, 'process_bookmark') as mock_process:
        # Mock successful processing
        mock_process.return_value = {
            'bookmark_id': '1',
            'potential_duplicates_found': 1,
            'requires_user_confirmation': True
        }
        
        results = deduplicator.batch_process_bookmarks(bookmark_ids)
        
        assert len(results) == len(bookmark_ids)
        assert all(isinstance(r, dict) for r in results)
        assert mock_process.call_count == len(bookmark_ids)

def test_confirm_duplicate(deduplicator):
    """Test duplicate confirmation flow"""
    with patch.object(deduplicator, 'mark_as_duplicate') as mock_mark_duplicate:
        with patch.object(deduplicator, 'mark_as_not_duplicate') as mock_mark_not_duplicate:
            # Test confirming as duplicate
            deduplicator.confirm_duplicate('1', '2', 0.95, True)
            mock_mark_duplicate.assert_called_once()
            mock_mark_not_duplicate.assert_not_called()
            
            # Reset mocks
            mock_mark_duplicate.reset_mock()
            mock_mark_not_duplicate.reset_mock()
            
            # Test rejecting as duplicate
            deduplicator.confirm_duplicate('1', '2', 0.95, False)
            mock_mark_duplicate.assert_not_called()
            mock_mark_not_duplicate.assert_called_once()