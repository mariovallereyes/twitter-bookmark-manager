import os
import json
import logging
import requests
from pathlib import Path
import imagehash
from PIL import Image
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from database.db import get_db_session
from database.models import Bookmark
from database.vector_store import ChromaStore
from core.ai_categorization import BookmarkCategorizer
from core.deduplication import BookmarkDeduplicator
import uuid  # Add this import at the top

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaHandler:
    def __init__(self, media_dir: Path):
        self.media_dir = media_dir
        self.media_dir.mkdir(parents=True, exist_ok=True)
        
    def get_content_type(self, url: str) -> str:
        try:
            response = requests.head(url)
            return response.headers.get('content-type', '')
        except Exception as e:
            logger.error(f"Error getting content type for {url}: {e}")
            return ''
            
    def download_media(self, tweet_id: str, url: str) -> tuple[str, str]:
        try:
            response = requests.head(url)
            content_type = response.headers.get('content-type', '')
            
            if not content_type.startswith(('image/', 'video/')):
                return None, None
                
            response = requests.get(url)
            extension = content_type.split('/')[-1]
            file_path = self.media_dir / f"{tweet_id}.{extension}"
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return str(file_path), content_type
            
        except Exception as e:
            logger.error(f"Error downloading media from {url}: {e}")
            return None, None

    def process_bookmark_media(self, tweet_id: str, media_urls: List[str]) -> List[Dict[str, Any]]:
        media_info = []
        for url in media_urls:
            file_path, content_type = self.download_media(tweet_id, url)
            if file_path:
                media_info.append({
                    'url': url,
                    'file_path': file_path,
                    'type': content_type
                })
        return media_info

class BookmarkIngester:
    def __init__(self, json_path: str, media_dir: Path):
        self.json_path = json_path
        self.media_dir = media_dir
        self.media_handler = MediaHandler(media_dir)
        self.vector_store = ChromaStore()
        self.deduplicator = BookmarkDeduplicator()
        self.categorizer = BookmarkCategorizer()
        
    def fetch_bookmarks(self) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Debug: Print structure of first bookmark
                if data and len(data) > 0:
                    first_bookmark = data[0]
                    logger.info("First bookmark keys: %s", first_bookmark.keys())
                    logger.info("First bookmark content: %s", first_bookmark)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    bookmarks = data
                elif isinstance(data, dict):
                    bookmarks = data.get('data', [])
                else:
                    raise ValueError(f"Unexpected JSON structure: {type(data)}")
                    
                # Validate bookmarks - less strict
                valid_bookmarks = [
                    b for b in bookmarks 
                    if isinstance(b, dict)  # Remove id_str check for now
                ]
                
                logger.info(f"Found {len(valid_bookmarks)} valid bookmarks")
                return valid_bookmarks
                
        except Exception as e:
            logger.error(f"Error loading bookmarks: {e}")
            raise

    def process_bookmark(self, bookmark: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extract ID from tweet URL
            tweet_url = bookmark.get('tweet_url')
            if not tweet_url:
                return {'error': 'Missing tweet_url'}
            
            # Ensure bookmark_id is a string
            bookmark_id = str(tweet_url.split('/')[-1])
            if not bookmark_id.isdigit():
                return {'error': f'Invalid ID: {bookmark_id}'}
            
            # Store in database
            with get_db_session() as session:
                db_bookmark = Bookmark(
                    id=bookmark_id,
                    text=bookmark.get('full_text', ''),
                    created_at=datetime.fromisoformat(bookmark.get('tweeted_at', '').replace('Z', '+00:00')),
                    author_name=bookmark.get('name'),
                    author_username=bookmark.get('screen_name'),
                    raw_data=bookmark
                )
                session.merge(db_bookmark)
                session.commit()
            
            # Add to vector store with explicit string ID
            self.vector_store.add_bookmark(
                bookmark_id=bookmark_id,  # Already a string
                text=bookmark.get('full_text', ''),
                metadata={}  # Empty dict instead of None
            )
            
            return {
                'bookmark_id': bookmark_id,
                'stored': True,
                'vector_store': {'added': True}
            }
            
        except Exception as e:
            return {'error': str(e)}

    def process_all_bookmarks(self) -> List[Dict[str, Any]]:
        """Process all bookmarks from JSON file"""
        try:
            bookmarks = self.fetch_bookmarks()
            results = []
            
            for bookmark in bookmarks:
                try:
                    # Process bookmark directly without deduplication
                    result = self.process_bookmark(bookmark)
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            return [{'error': str(e)}]

def main():
    try:
        media_dir = Path('data/media')
        json_path = 'database/twitter_bookmarks.json'
        
        ingester = BookmarkIngester(json_path, media_dir)
        results = ingester.process_all_bookmarks()
        
        logger.info(f"Successfully processed {len(results)} bookmarks")
        
    except Exception as e:
        logger.error(f"Error during bookmark processing: {e}")
        raise

if __name__ == "__main__":
    main()