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
            
        # Row order: bookmark_id, text, created_at, author_name, author_username, media_files, raw_data, user_id
        try:
            # Handle media_files - could be JSON string or already a dict
            if row[5]:
                try:
                    if isinstance(row[5], dict):
                        media_files = row[5]
                    else:
                        media_files = json.loads(row[5])
                except:
                    media_files = {}
            else:
                media_files = {}
                
            # Handle raw_data - could be JSON string or already a dict
            if row[6]:
                try:
                    if isinstance(row[6], dict):
                        raw_data = row[6]
                    else:
                        raw_data = json.loads(row[6])
                except:
                    raw_data = {}
            else:
                raw_data = {}
                
            return cls(
                id=row[0],                # bookmark_id
                text=row[1],              # text
                created_at=row[2],        # created_at
                author_name=row[3],       # author_name
                author_username=row[4],   # author_username
                media_files=media_files,  # media_files (handled above)
                raw_data=raw_data,        # raw_data (handled above)
                user_id=row[7]            # user_id
            )
        except Exception as e:
            import traceback
            print(f"Error creating Bookmark from row: {e}")
            print(f"Row: {row}")
            traceback.print_exc()
            # Return a minimal bookmark instead of failing
            return cls(
                id=row[0] if len(row) > 0 else None,
                text=row[1] if len(row) > 1 else None,
                user_id=row[7] if len(row) > 7 else None
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