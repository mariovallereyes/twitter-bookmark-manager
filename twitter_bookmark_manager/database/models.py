from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table, Text, JSON, Boolean, Float
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from typing import List
import datetime

Base = declarative_base()

# Association table for bookmark categories
bookmark_categories = Table(
    'bookmark_categories',
    Base.metadata,
    Column('bookmark_id', Integer, ForeignKey('bookmarks.id')),
    Column('category_id', Integer, ForeignKey('categories.id'))
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    twitter_id = Column(String(255), unique=True, nullable=False)
    access_token = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    bookmarks = relationship("Bookmark", back_populates="user")

class Bookmark(Base):
    __tablename__ = 'bookmarks'
    
    id = Column(String, primary_key=True)
    text = Column(Text)
    created_at = Column(DateTime)
    author_name = Column(String)
    author_username = Column(String)
    media_files = Column(JSON)
    raw_data = Column(JSON)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    user = relationship("User", back_populates="bookmarks")
    media = relationship("Media", back_populates="bookmark")
    categories = relationship("Category", secondary=bookmark_categories, back_populates="bookmarks")

class Media(Base):
    __tablename__ = 'media'
    
    id = Column(Integer, primary_key=True)
    bookmark_id = Column(Integer, ForeignKey('bookmarks.id'))
    type = Column(String(50))  # image, video, etc.
    url = Column(String(512))
    local_path = Column(String(512))
    hash = Column(String(255))  # For deduplication
    created_at = Column(DateTime, default=func.now())
    
    bookmark = relationship("Bookmark", back_populates="media")

class Category(Base):
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    bookmarks = relationship("Bookmark", secondary=bookmark_categories, back_populates="categories")

# New Conversation model
class Conversation(Base):
    """Model for storing conversation history with bookmarks"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(255), nullable=False)  # UUID for grouping exchanges
    timestamp = Column(DateTime, default=func.now())
    user_input = Column(Text, nullable=False)
    system_response = Column(JSON, nullable=False)  # Full response with metadata
    bookmarks_used = Column(JSON)  # IDs and relevance scores of used bookmarks
    
    # Memory management
    is_archived = Column(Boolean, default=False)
    last_accessed = Column(DateTime, default=func.now(), onupdate=func.now())
    importance_score = Column(Float)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, timestamp={self.timestamp})>"