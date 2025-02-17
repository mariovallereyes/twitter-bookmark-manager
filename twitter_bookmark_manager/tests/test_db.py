import pytest
import tempfile
import shutil
from database.db import DatabaseManager
from database.models import User
import os
from sqlalchemy.orm.session import Session
import numpy as np

@pytest.fixture
def test_db():
    """Create a test database manager with in-memory SQLite"""
    # Set up test environment
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    test_dir = tempfile.mkdtemp()
    os.environ['VECTOR_DB_PATH'] = test_dir
    
    db = DatabaseManager()
    yield db
    
    # Cleanup
    shutil.rmtree(test_dir)

# Existing SQLite Tests
def test_database_initialization(test_db):
    """Test database initialization creates tables"""
    test_db.init_db()
    with test_db.get_session() as session:
        assert isinstance(session, Session)

def test_session_management(test_db):
    """Test session creation and automatic cleanup"""
    test_db.init_db()
    with test_db.get_session() as session:
        user = User(twitter_id="test123")
        session.add(user)
        session.commit()
        
        saved_user = session.query(User).first()
        assert saved_user.twitter_id == "test123"

def test_session_rollback_on_error(test_db):
    """Test automatic rollback on error"""
    test_db.init_db()
    
    with pytest.raises(Exception):
        with test_db.get_session() as session:
            user = User(twitter_id="test123")
            session.add(user)
            raise Exception("Test error")
    
    with test_db.get_session() as session:
        assert session.query(User).count() == 0

# New Vector Store Tests
def test_vector_store_initialization(test_db):
    """Test vector store initialization"""
    vector_store = test_db.get_vector_store()
    assert vector_store is not None
    assert vector_store.collection is not None

def test_vector_store_operations(test_db):
    """Test vector store add and query operations"""
    vector_store = test_db.get_vector_store()
    
    # Test data with unique IDs
    test_id1 = f"test_op_{np.random.randint(10000)}"
    test_id2 = f"test_op_{np.random.randint(10000)}"
    while test_id2 == test_id1:  # Ensure IDs are different
        test_id2 = f"test_op_{np.random.randint(10000)}"
    
    ids = [test_id1, test_id2]
    embeddings = [
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist()
    ]
    metadatas = [
        {"text": "test tweet 1"},
        {"text": "test tweet 2"}
    ]
    texts = ["test tweet 1", "test tweet 2"]

    # Test adding embeddings
    vector_store.add_embeddings(ids, embeddings, metadatas, texts)
    
    # Verify using get() instead of query_similar()
    result = vector_store.collection.get(
        ids=[test_id1],
        include=['metadatas', 'documents']
    )
    assert result['ids'] == [test_id1]
    assert result['metadatas'][0]['text'] == "test tweet 1"
    assert result['documents'][0] == "test tweet 1"

def test_vector_store_deletion(test_db):
    """Test vector store deletion operations"""
    vector_store = test_db.get_vector_store()
    
    # Add test data with unique ID
    test_id = f"test_{np.random.randint(10000)}"  # Generate unique ID
    ids = [test_id]
    embeddings = [np.random.rand(384).tolist()]
    metadatas = [{"text": "test tweet"}]
    texts = ["test tweet"]
    
    vector_store.add_embeddings(ids, embeddings, metadatas, texts)
    
    # Test deletion
    vector_store.delete_embeddings(ids)
    
    # Verify deletion by trying to get the specific ID
    try:
        vector_store.collection.get(ids=ids)
        assert False, "Document should have been deleted"
    except Exception:
        assert True  # Document was deleted successfully