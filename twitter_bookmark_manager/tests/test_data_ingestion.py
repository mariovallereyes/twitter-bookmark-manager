import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from datetime import datetime
from core.data_ingestion import BookmarkIngester, MediaHandler

@pytest.fixture
def sample_bookmark():
    return {
        "id_str": "123456789",
        "profile_image_url_https": "https://example.com/profile.jpg",
        "screen_name": "test_user",
        "name": "Test User",
        "full_text": "Test bookmark with #AI and media",
        "tweeted_at": "2025-01-27T17:50:29.000Z",
        "extended_media": [{
            "media_url_https": "https://example.com/image.jpg",
            "type": "photo",
            "display_url": "pic.twitter.com/123"
        }]
    }

@pytest.fixture
def mock_vector_store():
    mock = Mock()
    mock.add_embeddings.return_value = True  # Changed from add_bookmark
    mock.get_collection_stats.return_value = {'count': 1, 'name': 'test'}
    return mock

@pytest.fixture
def ingester(mock_vector_store):
    with patch('core.data_ingestion.ChromaStore', return_value=mock_vector_store):
        return BookmarkIngester('database/twitter_bookmarks.json', Path("test_media"))

@pytest.fixture
def media_handler():
    return MediaHandler(Path("test_media"))

def test_process_bookmark_basic(ingester, sample_bookmark):
    """Test basic bookmark processing"""
    with patch('core.data_ingestion.get_db_session'):
        result = ingester.process_bookmark(sample_bookmark)
        assert result['bookmark_id'] == sample_bookmark['id_str']

def test_media_handling(media_handler, sample_bookmark):
    """Test media download and processing"""
    with patch('requests.get') as mock_get:
        with patch('requests.head') as mock_head:
            # Setup mocks
            mock_head.return_value.headers = {'content-type': 'image/jpeg'}
            mock_get.return_value.content = b"fake_image_data"
            
            # Process media
            media_info = media_handler.process_bookmark_media(
                "test_id",
                [sample_bookmark['extended_media'][0]['media_url_https']]
            )
            
            assert len(media_info) == 1
            assert media_info[0]['type'] == 'image/jpeg'

def test_json_loading(ingester):
    """Test JSON file loading"""
    test_data = '{"data": [{"id": "1", "full_text": "test"}]}'  # Use proper JSON string
    with patch('builtins.open', mock_open(read_data=test_data)):
        bookmarks = ingester.fetch_bookmarks()
        assert len(bookmarks) > 0

def test_database_storage(ingester, sample_bookmark):
    """Test database storage"""
    with patch('core.data_ingestion.get_db_session') as mock_session:
        mock_session.return_value.__enter__.return_value = Mock()
        
        result = ingester.process_bookmark(sample_bookmark)
        
        # Verify database interaction
        mock_session.return_value.__enter__.return_value.merge.assert_called_once()
        mock_session.return_value.__enter__.return_value.commit.assert_called_once()

def test_vector_store_integration(ingester, sample_bookmark):
    """Test vector store integration"""
    with patch.object(ingester.vector_store, 'add_bookmark') as mock_add:
        with patch('core.data_ingestion.get_db_session'):
            result = ingester.process_bookmark(sample_bookmark)
            
            # Verify vector store interaction
            mock_add.assert_called_once()
            assert 'vector_store' in result

def test_error_handling(ingester):
    """Test error handling for invalid data"""
    invalid_bookmark = {}  # Empty bookmark should raise error
    
    with pytest.raises(Exception):
        ingester.process_bookmark(invalid_bookmark)

def test_date_parsing(ingester, sample_bookmark):
    """Test date parsing from Twitter format"""
    with patch('core.data_ingestion.get_db_session') as mock_session:
        result = ingester.process_bookmark(sample_bookmark)
        
        # Verify date was parsed correctly
        mock_session.return_value.__enter__.return_value.merge.assert_called_once()
        args = mock_session.return_value.__enter__.return_value.merge.call_args[0][0]
        assert isinstance(args.created_at, datetime)

def test_batch_processing(ingester):
    """Test processing multiple bookmarks"""
    test_bookmarks = [
        {'id': '1', 'full_text': 'test1'},
        {'id': '2', 'full_text': 'test2'}
    ]
    
    with patch.object(ingester, 'process_bookmark') as mock_process:
        mock_process.return_value = {'status': 'success'}
        
        results = ingester.process_all_bookmarks()
        
        assert len(results) > 0

def test_deduplication(ingester, sample_bookmark):
    """Test that duplicates are properly handled"""
    with patch.object(ingester.deduplicator, 'find_potential_duplicates') as mock_find:
        # First run - no duplicates
        mock_find.return_value = []
        result = ingester.process_bookmark(sample_bookmark)
        assert result['bookmark_id'] == sample_bookmark['id_str']
        
        # Second run - with duplicate
        mock_find.return_value = [{'id': '123', 'similarity': 0.98}]
        results = ingester.process_all_bookmarks()
        assert any(r.get('skipped') for r in results)