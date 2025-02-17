import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, User, Bookmark, Media, Category
from datetime import datetime

@pytest.fixture
def engine():
    """Create a test database in memory"""
    return create_engine('sqlite:///:memory:')

@pytest.fixture
def session(engine):
    """Create all tables and return a test session"""
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def test_user_creation(session):
    """Test creating a user"""
    user = User(twitter_id="123456")
    session.add(user)
    session.commit()
    
    assert user.id is not None
    assert user.twitter_id == "123456"
    assert user.created_at is not None

def test_bookmark_creation(session):
    """Test creating a bookmark with relationships"""
    user = User(twitter_id="123456")
    bookmark = Bookmark(
        tweet_id="789",
        text="Test tweet",
        author_id="author123",
        created_at=datetime.utcnow(),
        user=user
    )
    
    session.add(bookmark)
    session.commit()
    
    assert bookmark.id is not None
    assert bookmark.user.twitter_id == "123456"

def test_media_creation(session):
    """Test creating media attached to a bookmark"""
    bookmark = Bookmark(tweet_id="789", text="Test tweet")
    media = Media(
        type="image",
        url="http://example.com/image.jpg",
        local_path="/path/to/image.jpg",
        hash="abcd1234",
        bookmark=bookmark
    )
    
    session.add(media)
    session.commit()
    
    assert media.id is not None
    assert media.bookmark.tweet_id == "789"

def test_category_relationships(session):
    """Test category and bookmark relationships"""
    category = Category(name="Technology", description="Tech tweets")
    bookmark = Bookmark(tweet_id="789", text="Test tweet")
    bookmark.categories.append(category)
    
    session.add(bookmark)
    session.commit()
    
    assert len(bookmark.categories) == 1
    assert bookmark.categories[0].name == "Technology"
    assert len(category.bookmarks) == 1