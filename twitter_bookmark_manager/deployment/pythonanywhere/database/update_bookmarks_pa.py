"""
PythonAnywhere-specific bookmark update script.
"""
import sys
import os
import json
import logging
import traceback
import uuid  # Added for session_id generation
from datetime import datetime
from pathlib import Path
from sqlalchemy import String, cast, create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Tuple, Generator

# Import the Bookmark model and db_session
from .db_pa import db_session, get_db_session
from twitter_bookmark_manager.database.models import Bookmark

# Set up base directory for PythonAnywhere
PA_BASE_DIR = '/home/mariovallereyes/twitter_bookmark_manager'
DATABASE_DIR = os.path.join(PA_BASE_DIR, 'database')
MEDIA_DIR = os.path.join(PA_BASE_DIR, 'media')
VECTOR_DB_DIR = os.path.join(DATABASE_DIR, 'vector_db')

# Add application directory to Python path if not already there
if PA_BASE_DIR not in sys.path:
    sys.path.insert(0, PA_BASE_DIR)

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
            logger.error("No tweet_url found in data")
            return None
            
        tweet_id = tweet_url.split('/')[-1] if tweet_url else None
        if not tweet_id:
            logger.error("Could not extract tweet ID from URL")
            return None
        
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
            logger.error("No valid ID could be extracted")
            return None
            
        return mapped_data
        
    except Exception as e:
        logger.error(f"Error mapping bookmark data: {e}")
        return None

def pa_update_bookmarks(session_id=None, start_index=0):
    """PythonAnywhere-specific function to update bookmarks from JSON file
    
    Args:
        session_id (str, optional): Unique identifier for this update session. If not provided, one will be generated.
        start_index (int, optional): Index to start/resume processing from. Defaults to 0.
        
    Returns:
        dict: Result dictionary containing progress information and success status
    """
    try:
        # Generate a session ID if none provided
        if not session_id:
            session_id = str(uuid.uuid4())
            
        logger.info("="*50)
        logger.info(f"Starting PythonAnywhere bookmark update process for session {session_id} from index {start_index}")
        
        # Define paths
        json_file = os.path.join(DATABASE_DIR, 'twitter_bookmarks.json')
        progress_file = os.path.join(DATABASE_DIR, 'update_progress.json')
        logger.info(f"Looking for JSON file at: {json_file}")
        
        # Load progress if exists
        current_progress = {}
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    current_progress = json.load(f)
                logger.info(f"Loaded progress: {current_progress}")
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
        
        # Use provided start_index or resume from saved progress if start_index is 0
        if start_index == 0 and current_progress.get('last_processed_index'):
            logger.info(f"No start_index provided, resuming from saved progress: {current_progress.get('last_processed_index')}")
            start_index = current_progress.get('last_processed_index', 0)
            
        processed_ids = set(current_progress.get('processed_ids', []))
        
        # Initialize statistics from progress or start fresh
        stats = current_progress.get('stats', {
            'new_count': 0,
            'updated_count': 0,
            'errors': 0,
            'total_processed': 0
        })
        
        # Load and parse JSON
        logger.info(f"Reading bookmarks from {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                new_bookmarks = json.load(f)
            total_bookmarks = len(new_bookmarks)
            logger.info(f"Successfully loaded {total_bookmarks} bookmarks from JSON")
            
            # Skip already processed bookmarks based on start_index
            if start_index > 0:
                logger.info(f"Resuming from index {start_index} of {total_bookmarks}")
                new_bookmarks = new_bookmarks[start_index:]
            
            logger.info(f"Processing remaining {len(new_bookmarks)} bookmarks")
        except Exception as e:
            error_msg = f"Error reading JSON file: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # Process bookmarks in smaller batches
        BATCH_SIZE = 25  # Reduced batch size
        current_batch = []
        
        try:
            with get_db_session() as session:
                # Get existing bookmarks once
                existing_bookmarks = {}
                for bookmark in session.query(Bookmark).all():
                    if bookmark.raw_data and 'tweet_url' in bookmark.raw_data:
                        existing_bookmarks[bookmark.raw_data['tweet_url']] = bookmark
                
                # Process each bookmark
                for i, bookmark_data in enumerate(new_bookmarks, start_index):
                    try:
                        tweet_url = bookmark_data.get('tweet_url')
                        if not tweet_url or tweet_url in processed_ids:
                            continue
                            
                        mapped_data = map_bookmark_data(bookmark_data)
                        if not mapped_data:
                            stats['errors'] += 1
                            continue
                        
                        # Add to current batch
                        current_batch.append((tweet_url, mapped_data, bookmark_data))
                        
                        # Process batch if full or last item
                        if len(current_batch) >= BATCH_SIZE or i == len(new_bookmarks) + start_index - 1:
                            # Process the batch
                            for url, data, raw in current_batch:
                                try:
                                    if url not in existing_bookmarks:
                                        new_bookmark = Bookmark(**data)
                                        session.add(new_bookmark)
                                        session.flush()
                                        stats['new_count'] += 1
                                    else:
                                        existing = existing_bookmarks[url]
                                        for key, value in data.items():
                                            setattr(existing, key, value)
                                        stats['updated_count'] += 1
                                    
                                    processed_ids.add(url)
                                except Exception as e:
                                    logger.error(f"Error processing bookmark {url}: {e}")
                                    stats['errors'] += 1
                            
                            # Commit the batch
                            session.commit()
                            
                            # Update progress file with session_id
                            stats['total_processed'] = i + 1
                            progress_data = {
                                'session_id': session_id,
                                'last_processed_index': i + 1,
                                'processed_ids': list(processed_ids),
                                'stats': stats,
                                'last_update': datetime.now().isoformat()
                            }
                            
                            with open(progress_file, 'w') as f:
                                json.dump(progress_data, f)
                            
                            # Clear batch
                            current_batch = []
                            
                            # Log progress
                            progress_percent = ((i + 1) / total_bookmarks) * 100
                            logger.info(f"Progress: {progress_percent:.1f}% ({i + 1}/{total_bookmarks})")
                            
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                        stats['errors'] += 1
                        continue
                
                # Return progress information
                is_complete = stats['total_processed'] >= total_bookmarks
                logger.info(f"Update {'completed' if is_complete else 'in progress'}: {stats['total_processed']}/{total_bookmarks} bookmarks processed")
                
                return {
                    'success': True,
                    'session_id': session_id,
                    'progress': {
                        'total': total_bookmarks,
                        'processed': stats['total_processed'],
                        'new': stats['new_count'],
                        'updated': stats['updated_count'],
                        'errors': stats['errors'],
                        'percent_complete': (stats['total_processed'] / total_bookmarks) * 100
                    },
                    'is_complete': is_complete,
                    'next_index': stats['total_processed'] if not is_complete else None
                }
                
        except Exception as e:
            error_msg = f"Database error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'session_id': session_id,
                'progress': {
                    'last_index': start_index,
                    'stats': stats
                }
            }
            
    except Exception as e:
        logger.error(f"Update error: {str(e)}")
        return {'success': False, 'error': str(e), 'session_id': session_id if session_id else None}

def test_resumable_update():
    """
    Test function to validate the resumable update functionality.
    This can be run with:
        python -c "from twitter_bookmark_manager.deployment.pythonanywhere.database.update_bookmarks_pa import test_resumable_update; test_resumable_update()"
    """
    try:
        # First run - start from beginning
        logger.info("STARTING TEST: First run (from beginning)")
        result1 = pa_update_bookmarks(start_index=0)
        logger.info(f"First run result: {result1}")
        
        if not result1['success']:
            logger.error(f"First run failed: {result1.get('error')}")
            return
            
        if result1['is_complete']:
            logger.info("First run completed all bookmarks - nothing to resume")
            return
            
        # Second run - resume from where we left off
        next_index = result1['next_index']
        session_id = result1['session_id']
        logger.info(f"STARTING TEST: Second run (resuming from index {next_index})")
        
        result2 = pa_update_bookmarks(session_id=session_id, start_index=next_index)
        logger.info(f"Second run result: {result2}")
        
        if not result2['success']:
            logger.error(f"Second run failed: {result2.get('error')}")
            return
            
        logger.info("TEST COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    pa_update_bookmarks() 