import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Conversation
from database.db import get_db_session
from unittest.mock import patch
import os

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test database and tables"""
    # Use SQLite in-memory database for testing
    engine = create_engine('sqlite:///:memory:')
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    TestingSessionLocal = sessionmaker(bind=engine)
    
    # Patch the database session to use our test database
    with patch('database.db.SessionLocal', TestingSessionLocal):
        yield engine
    
    # Clean up
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(setup_test_db):
    """Provide a transactional scope around each test"""
    Session = sessionmaker(bind=setup_test_db)
    session = Session()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()