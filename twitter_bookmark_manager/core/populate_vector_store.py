import sys
import os

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database.db import get_db_session
from database.models import Bookmark
from database.vector_store import ChromaStore
import logging

logging.disable(logging.CRITICAL)

def populate_chroma():
    try:
        vector_store = ChromaStore()
        
        with get_db_session() as session:
            bookmarks = session.query(Bookmark).all()
            print(f"Found {len(bookmarks)} bookmarks in SQLite")
            
            for bookmark in bookmarks:
                try:
                    vector_store.add_bookmark(
                        bookmark_id=str(bookmark.id),
                        text=bookmark.text,
                        metadata={
                            'created_at': bookmark.created_at.isoformat() if bookmark.created_at else '',
                            'author': bookmark.author_username or '',
                            'source': 'twitter'
                        }
                    )
                except Exception as e:
                    print(f"Error adding bookmark {bookmark.id}: {e}")
                    
            print("Finished populating ChromaDB")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_chroma()