"""
PythonAnywhere-specific bookmark update script.
"""
import sys
import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from sqlalchemy import String, cast, create_engine
from sqlalchemy.orm import sessionmaker

# Set up base directory for PythonAnywhere
PA_BASE_DIR = '/home/mariovallereyes/twitter_bookmark_manager'
DATABASE_DIR = os.path.join(PA_BASE_DIR, 'database')
MEDIA_DIR = os.path.join(PA_BASE_DIR, 'media')
VECTOR_DB_DIR = os.path.join(DATABASE_DIR, 'vector_db')

# Add application directory to Python path
if PA_BASE_DIR not in sys.path:
    sys.path.insert(0, PA_BASE_DIR)

# Import required modules
try:
    from database.models import Bookmark
    from deployment.pythonanywhere.postgres.config import get_database_url
    from deployment.pythonanywhere.database.vector_store_pa import VectorStore
    logger = logging.getLogger('pa_update')
    logger.info("Successfully imported required modules")
except Exception as e:
    print(f"Error importing modules: {str(e)}")
    print(traceback.format_exc())
    print(f"Current sys.path: {sys.path}")
    raise

# Set up logging with absolute paths
LOG_DIR = os.path.join(PA_BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'pa_update_log.txt')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file):
    """Set up a logger with both file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger('pa_update', LOG_FILE)

def map_bookmark_data(data):
    """Map JSON data to Bookmark model fields"""
    try:
        # Extract tweet ID from URL
        tweet_url = data.get('tweet_url', '')
        if not tweet_url:
            raise ValueError("No tweet_url found in data")
            
        tweet_id = tweet_url.split('/')[-1] if tweet_url else None
        if not tweet_id:
            raise ValueError("Could not extract tweet ID from URL")
        
        # Parse datetime
        created_at = None
        if tweeted_at := data.get('tweeted_at'):
            try:
                created_at = datetime.strptime(tweeted_at, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                try:
                    created_at = datetime.strptime(tweeted_at, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    logger.error(f"Could not parse datetime: {tweeted_at}")

        # Extract text, preferring full_text over text
        text = data.get('full_text') or data.get('text')
        
        # Map the data
        mapped_data = {
            'id': tweet_id,
            'text': text,
            'created_at': created_at,
            'author_name': data.get('name'),
            'author_username': data.get('screen_name'),
            'media_files': data.get('media_files', {}),
            'raw_data': data
        }
        
        # Validate required fields
        if not mapped_data['id']:
            raise ValueError("No valid ID could be extracted")
            
        return mapped_data
        
    except Exception as e:
        logger.error(f"Error mapping bookmark data: {e}")
        raise

def pa_update_bookmarks():
    """PythonAnywhere-specific function to update bookmarks from JSON file"""
    try:
        logger.info("="*50)
        logger.info("Starting PythonAnywhere bookmark update process")
        
        # Define paths
        json_file = os.path.join(DATABASE_DIR, 'twitter_bookmarks.json')
        logger.info(f"Looking for JSON file at: {json_file}")
        
        # Ensure directories exist
        os.makedirs(DATABASE_DIR, exist_ok=True)
        os.makedirs(MEDIA_DIR, exist_ok=True)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        
        if not os.path.exists(json_file):
            error_msg = f"Bookmarks JSON file not found at {json_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        # Load and parse JSON
        logger.info(f"Reading bookmarks from {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                new_bookmarks = json.load(f)
            total_bookmarks = len(new_bookmarks)
            logger.info(f"Successfully loaded {total_bookmarks} bookmarks from JSON")
        except Exception as e:
            error_msg = f"Error reading JSON file: {str(e)}"
            logger.error(error_msg)
            raise
            
        # Initialize vector store
        try:
            vector_store = VectorStore()
            logger.info("Successfully initialized vector store")
        except Exception as e:
            error_msg = f"Failed to initialize vector store: {str(e)}"
            logger.error(error_msg)
            raise

        # Initialize PostgreSQL engine and session
        engine = create_engine(get_database_url())
        Session = sessionmaker(bind=engine)
        session = Session()

        # Track statistics
        new_count = 0
        updated_count = 0
        errors = 0
        processed_ids = set()  # Track processed IDs to avoid duplicates

        try:
            # Get existing bookmark URLs
            existing_bookmarks = {}
            for bookmark in session.query(Bookmark).all():
                if bookmark.raw_data and 'tweet_url' in bookmark.raw_data:
                    existing_bookmarks[bookmark.raw_data['tweet_url']] = bookmark
            
            # Identify new bookmarks using tweet_url
            new_urls = {bookmark['tweet_url'] for bookmark in new_bookmarks}
            to_add = new_urls - set(existing_bookmarks.keys())
            
            logger.info(f"Found {len(to_add)} new bookmarks to add")
            logger.info(f"Found {len(new_urls - to_add)} existing bookmarks")

            # Process bookmarks in batches
            BATCH_SIZE = 50
            for i, bookmark_data in enumerate(new_bookmarks, 1):
                try:
                    tweet_url = bookmark_data.get('tweet_url')
                    if not tweet_url:
                        logger.warning(f"Skipping bookmark without tweet_url at index {i}")
                        continue

                    # Map the bookmark data
                    try:
                        mapped_data = map_bookmark_data(bookmark_data)
                    except Exception as e:
                        logger.error(f"Error mapping bookmark data for {tweet_url}: {e}")
                        errors += 1
                        continue
                        
                    # Skip if we've already processed this ID
                    if mapped_data['id'] in processed_ids:
                        logger.warning(f"Skipping duplicate bookmark ID: {mapped_data['id']}")
                        continue
                    processed_ids.add(mapped_data['id'])
                    
                    # If this is a new bookmark
                    if tweet_url in to_add:
                        try:
                            # Create new bookmark with mapped data
                            new_bookmark = Bookmark(**mapped_data)
                            session.merge(new_bookmark)  # Use merge instead of add
                            session.flush()  # Get the ID
                            
                            # Add to vector store
                            metadata = {
                                'tweet_url': tweet_url,
                                'screen_name': bookmark_data.get('screen_name', 'unknown'),
                                'author_name': bookmark_data.get('name', 'unknown'),
                                'id': str(new_bookmark.id)
                            }
                            
                            vector_store.add_bookmark(
                                bookmark_id=metadata['id'],
                                text=mapped_data['text'] or '',
                                metadata=metadata
                            )
                            
                            new_count += 1
                            logger.debug(f"Added new bookmark: {tweet_url}")
                        except Exception as e:
                            logger.error(f"Error adding new bookmark {tweet_url}: {e}")
                            session.rollback()
                            errors += 1
                            continue
                    else:
                        try:
                            # Update existing bookmark
                            existing = existing_bookmarks[tweet_url]
                            for key, value in mapped_data.items():
                                setattr(existing, key, value)
                            updated_count += 1
                            logger.debug(f"Updated bookmark: {tweet_url}")
                        except Exception as e:
                            logger.error(f"Error updating bookmark {tweet_url}: {e}")
                            session.rollback()
                            errors += 1
                            continue

                    # Commit in batches
                    if i % BATCH_SIZE == 0:
                        try:
                            session.commit()
                            logger.info(f"Progress: {i}/{total_bookmarks} bookmarks processed ({(i/total_bookmarks)*100:.1f}%)")
                        except Exception as e:
                            logger.error(f"Error committing batch: {str(e)}")
                            session.rollback()
                            errors += len(processed_ids)
                            processed_ids.clear()  # Reset processed IDs after rollback
                            continue

                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing bookmark {bookmark_data.get('tweet_url', 'unknown')}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # Final commit
            try:
                session.commit()
                logger.info("Successfully committed all database changes")
            except Exception as e:
                logger.error(f"Error in final commit: {str(e)}")
                session.rollback()
                raise

        finally:
            session.close()

        # Log summary
        logger.info("="*50)
        logger.info(f"Update Summary:")
        logger.info(f"Total bookmarks in JSON: {total_bookmarks}")
        logger.info(f"New bookmarks added: {new_count}")
        logger.info(f"Bookmarks updated: {updated_count}")
        logger.info(f"Errors encountered: {errors}")
        logger.info(f"Unique IDs processed: {len(processed_ids)}")
        logger.info("="*50)

        return {
            'success': True,
            'new_bookmarks': new_count,
            'updated_bookmarks': updated_count,
            'errors': errors,
            'total_processed': total_bookmarks,
            'unique_ids': len(processed_ids)
        }

    except Exception as e:
        logger.error(f"Fatal error in pa_update_bookmarks: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

if __name__ == "__main__":
    pa_update_bookmarks() 