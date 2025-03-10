"""
Models for the final multi-user version of Twitter Bookmark Manager.
Contains simplified database models without using SQLAlchemy's declarative base.
"""

from sqlalchemy import text
import json
import datetime

class Bookmark:
    """Bookmark model for the Twitter Bookmark Manager"""

    def __init__(self, id=None, text=None, created_at=None, author_name=None,
                 author_username=None, media_files=None, raw_data=None, user_id=None):
        self.id = id
        self.text = text
        self.created_at = created_at
        self.author_name = author_name
        self.author_username = author_username
        self.media_files = media_files or {}
        self.raw_data = raw_data or {}
        self.user_id = user_id

    @classmethod
    def from_row(cls, row):
        """Create a Bookmark object from database row tuple"""
        if not row:
            return None
            
        # Assume row order: id, text, created_at, author_name, author_username, media_files, raw_data, user_id
        return cls(
            id=row[0],
            text=row[1],
            created_at=row[2],
            author_name=row[3],
            author_username=row[4],
            media_files=json.loads(row[5]) if row[5] else {},
            raw_data=json.loads(row[6]) if row[6] else {},
            user_id=row[7]
        )

    def to_dict(self):
        """Convert bookmark to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime.datetime) else self.created_at,
            'author_name': self.author_name,
            'author_username': self.author_username,
            'media_files': self.media_files,
            'raw_data': self.raw_data,
            'user_id': self.user_id
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Bookmark from a dictionary"""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.datetime.fromisoformat(created_at)
            except ValueError:
                # Handle other possible date formats here if needed
                pass
        
        return cls(
            id=data.get('id'),
            text=data.get('text'),
            created_at=created_at,
            author_name=data.get('author_name'),
            author_username=data.get('author_username'),
            media_files=data.get('media_files', {}),
            raw_data=data.get('raw_data', {}),
            user_id=data.get('user_id')
        )

# Database functions for bookmarks
def get_bookmark_by_id(conn, bookmark_id, user_id=None):
    """Get a bookmark by ID"""
    query = """
        SELECT id, text, created_at, author_name, author_username, media_files, raw_data, user_id
        FROM bookmarks
        WHERE id = :bookmark_id
    """
    
    params = {"bookmark_id": bookmark_id}
    
    # Add user_id filter if provided
    if user_id:
        query += " AND user_id = :user_id"
        params["user_id"] = user_id
    
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        row = cursor.fetchone()
        return Bookmark.from_row(row) if row else None

def create_bookmarks_table(conn):
    """Create the bookmarks table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS bookmarks (
        id VARCHAR(255) PRIMARY KEY,
        text TEXT,
        created_at TIMESTAMP,
        author_name VARCHAR(255),
        author_username VARCHAR(255),
        media_files JSONB,
        raw_data JSONB,
        user_id INTEGER REFERENCES users(id),
        embedding_id VARCHAR(255) NULL,
        categories JSONB NULL
    );
    """
    
    with conn.cursor() as cursor:
        cursor.execute(create_table_sql)
    conn.commit() 