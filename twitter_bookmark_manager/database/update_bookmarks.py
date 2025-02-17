import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import json
import logging
import argparse
from datetime import datetime
from database.db import get_db_session, get_vector_store
from database.models import Bookmark
from core.data_ingestion import BookmarkIngester
from pathlib import Path

# Set up logging BEFORE any function definitions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'update_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    ]
)
logger = logging.getLogger(__name__)

def rebuild_vector_store(vector_store):
    """Rebuild vector store from SQL database"""
    logger.info("Starting vector store rebuild...")
    try:
        # Get all existing IDs first
        results = vector_store.collection.get()
        if results and results['ids']:
            logger.info(f"Found {len(results['ids'])} existing entries in vector store")
            vector_store.collection.delete(ids=results['ids'])
            logger.info("Cleared existing vector store")
        
        with get_db_session() as session:
            bookmarks = session.query(Bookmark).all()
            total = len(bookmarks)
            logger.info(f"Found {total} bookmarks in SQL database")
            
            for i, bookmark in enumerate(bookmarks, 1):
                try:
                    tweet_url = bookmark.raw_data.get('tweet_url', '') if bookmark.raw_data else ''
                    vector_store.add_bookmark(
                        bookmark_id=str(bookmark.id),
                        text=bookmark.text or '',
                        metadata={
                            'tweet_url': tweet_url,
                            'screen_name': bookmark.author_username or '',
                            'author_name': bookmark.author_name or ''
                        }
                    )
                    if i % 50 == 0:
                        logger.info(f"Processed {i}/{total} bookmarks")
                except Exception as e:
                    logger.error(f"Error processing bookmark {bookmark.id}: {str(e)}")
            
            final_count = vector_store.collection.count()
            logger.info(f"\nRebuild Summary:")
            logger.info(f"- SQL database count: {total}")
            logger.info(f"- Vector store count: {final_count}")
            
            if total != final_count:
                logger.warning(f"Count mismatch! SQL: {total}, Vector: {final_count}")
            else:
                logger.info("✓ Databases are in sync")
            
    except Exception as e:
        logger.error(f"❌ Error rebuilding vector store: {str(e)}")
        raise

def update_bookmarks(rebuild_vectors=False):
    """Update bookmarks from new JSON file"""
    try:
        # Initialize vector store
        vector_store = get_vector_store()
        
        # If rebuild flag is set, only rebuild vector store and return
        if rebuild_vectors:
            rebuild_vector_store(vector_store)
            return
        
        # Regular update process
        json_path = 'database/twitter_bookmarks.json'
        logger.info(f"Loading new bookmarks from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            new_bookmarks = json.load(f)
        
        # Get existing bookmark URLs from raw_data
        with get_db_session() as session:
            existing_urls = {
                bookmark.raw_data.get('tweet_url') 
                for bookmark in session.query(Bookmark).all() 
                if bookmark.raw_data and 'tweet_url' in bookmark.raw_data
            }
            
            # Identify new bookmarks using tweet_url
            new_urls = {bookmark['tweet_url'] for bookmark in new_bookmarks}
            to_add = new_urls - existing_urls
            
            logger.info(f"Found {len(new_bookmarks)} total bookmarks")
            logger.info(f"Found {len(to_add)} new bookmarks to add")
            
            if not to_add:
                logger.info("No new bookmarks to add")
                return
            
            # Process only new bookmarks
            ingester = BookmarkIngester(
                json_path=json_path,
                media_dir=Path('media')
            )
            
            # Process new bookmarks
            results = []
            for bookmark in new_bookmarks:
                if bookmark['tweet_url'] in to_add:
                    result = ingester.process_bookmark(bookmark)
                    
                    # If bookmark was successfully added to SQL DB
                    if 'error' not in result:
                        try:
                            # Get the database-generated ID using raw_data
                            with get_db_session() as session:
                                db_bookmark = session.query(Bookmark).filter(
                                    Bookmark.raw_data['tweet_url'].astext == bookmark['tweet_url']
                                ).first()
                                
                                if db_bookmark:
                                    # Debug log the metadata
                                    metadata = {
                                        'tweet_url': bookmark['tweet_url'],
                                        'screen_name': bookmark.get('screen_name', 'unknown'),  # Using .get() with default
                                        'author_name': bookmark.get('name', 'unknown'),        # Using .get() with default
                                        'id': str(db_bookmark.id)                             # Adding ID explicitly
                                    }
                                    logger.info(f"Adding bookmark with metadata: {metadata}")
                                    
                                    # Add to vector store using DB ID
                                    vector_store.add_bookmark(
                                        bookmark_id=str(db_bookmark.id),
                                        text=bookmark['full_text'],
                                        metadata=metadata
                                    )
                        except Exception as e:
                            logger.error(f"Error updating vector store: {str(e)}")
                    
                    results.append(result)
                    logger.info(f"Processed bookmark: {bookmark['tweet_url']}")
            
            # Final stats
            success = len([r for r in results if 'error' not in r])
            logger.info(f"\nUpdate Summary:")
            logger.info(f"- Total new bookmarks processed: {len(results)}")
            logger.info(f"- Successfully added: {success}")
            logger.info(f"- Errors: {len(results) - success}")
            
            if len(results) - success > 0:
                logger.info("\nErrors encountered:")
                for result in results:
                    if 'error' in result:
                        logger.error(f"- {result['error']}")
            
    except Exception as e:
        logger.error(f"❌ Error updating bookmarks: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store from SQL database')
    args = parser.parse_args()
    
    update_bookmarks(rebuild_vectors=args.rebuild)