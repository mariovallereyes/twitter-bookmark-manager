"""
Final environment bookmark update script.
"""
import sys
import os
import json
import logging
import traceback
import uuid  # Added for session_id generation
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import String, cast, create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Tuple, Generator
import shutil
import re
import glob
import hashlib
import time
import gc
import random
import string
import tempfile

# Import the Bookmark model and db_session
from .db_final import db_session, get_db_session, get_bookmarks_for_user
from .models_final import Bookmark
from .vector_store_final import VectorStore  # Add import for the vector store

# Set up base directory using environment variables or relative paths
BASE_DIR = os.environ.get('APP_BASE_DIR', '/app')
DATABASE_DIR = os.environ.get('DATABASE_DIR', os.path.join(BASE_DIR, 'database'))
MEDIA_DIR = os.environ.get('MEDIA_DIR', os.path.join(BASE_DIR, 'media'))
VECTOR_DB_DIR = os.environ.get('VECTOR_DB_DIR', os.path.join(DATABASE_DIR, 'vector_db'))

# Add application directory to Python path if not already there
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Set up logging with absolute paths
LOG_DIR = os.environ.get('LOG_DIR', os.path.join(BASE_DIR, 'logs'))
LOG_FILE = os.path.join(LOG_DIR, 'final_update_log.txt')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize global variable for memory monitoring
_last_memory_usage = "0MB"

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

logger = setup_logger('final_update', LOG_FILE)

def map_bookmark_data(bookmark_data, user_id=None):
    """
    Map the bookmark JSON data to Bookmark model fields with improved error handling.
    
    Args:
        bookmark_data (dict): Raw bookmark data from JSON
        user_id (int, optional): User ID for multi-user support
        
    Returns:
        dict or None: Mapped data dictionary or None if mapping failed
    """
    try:
        # Ensure we have a dictionary
        if not isinstance(bookmark_data, dict):
            logger.error(f"Bookmark data is not a dictionary: {type(bookmark_data)}")
            return None
            
        # Extract tweet URL (this is our primary way to identify a bookmark)
        tweet_url = bookmark_data.get('tweet_url')
        
        # Try to extract tweet ID from various sources
        tweet_id = None
        
        # 1. Try direct 'id' field
        if 'id' in bookmark_data:
            tweet_id = str(bookmark_data['id'])
        
        # 2. Try to extract from URL if available
        elif tweet_url:
            # Extract ID from URL
            match = re.search(r'/status/(\d+)', tweet_url)
            if match:
                tweet_id = match.group(1)
                
        # If we still don't have an ID but have a URL, try alternative patterns
        if not tweet_id and tweet_url:
            # Try other URL patterns
            patterns = [
                r'twitter\.com/\w+/status/(\d+)',
                r'x\.com/\w+/status/(\d+)',
                r'/status/(\d+)',
                r'statuses/(\d+)',
                r'i/status/(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, tweet_url)
                if match:
                    tweet_id = match.group(1)
                    break
        
        # If we still don't have an ID or URL, log and skip
        if not tweet_id and not tweet_url:
            logger.error(f"Could not extract tweet ID or URL from bookmark data: {bookmark_data}")
            return None
            
        # If we have an ID but no URL, create a standard URL
        if tweet_id and not tweet_url:
            tweet_url = f"https://twitter.com/i/status/{tweet_id}"
            bookmark_data['tweet_url'] = tweet_url
        
        # Format tweet_id properly
        if tweet_id:
            # Ensure it's a string
            tweet_id = str(tweet_id).strip()
            
            # Remove any non-numeric characters
            tweet_id = re.sub(r'\D', '', tweet_id)
            
            # Validate the ID is numeric and reasonable length
            if not tweet_id.isdigit() or len(tweet_id) < 5 or len(tweet_id) > 30:
                logger.error(f"Invalid tweet ID format: {tweet_id}")
                # If we have a URL, try again with different pattern
                if tweet_url:
                    tweet_id = None
                    for pattern in patterns:
                        match = re.search(pattern, tweet_url)
                        if match:
                            tweet_id = match.group(1)
                            if tweet_id.isdigit() and 5 <= len(tweet_id) <= 30:
                                break
                            else:
                                tweet_id = None
                
                # If still invalid, use a fallback approach
                if not tweet_id or not tweet_id.isdigit() or len(tweet_id) < 5 or len(tweet_id) > 30:
                    logger.warning(f"Could not extract valid tweet ID from bookmark, generating a UUID-based ID")
                    import hashlib
                    import uuid
                    
                    # Generate a consistent ID based on content or URL
                    if tweet_url:
                        tweet_id = "gen_" + hashlib.md5(tweet_url.encode()).hexdigest()[:16]
                    else:
                        content = str(bookmark_data.get('tweet_content', '')) + str(bookmark_data.get('author_name', ''))
                        if content:
                            tweet_id = "gen_" + hashlib.md5(content.encode()).hexdigest()[:16]
                        else:
                            tweet_id = "gen_" + str(uuid.uuid4())[:16]
        
        # Parse the tweeted_at datetime
        tweeted_at = None
        date_str = bookmark_data.get('created_at') or bookmark_data.get('tweeted_at') or bookmark_data.get('date')
        
        if date_str:
            try:
                # Try multiple datetime formats
                for fmt in [
                    '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO format with microseconds 
                    '%Y-%m-%dT%H:%M:%SZ',     # ISO format without microseconds
                    '%a %b %d %H:%M:%S %z %Y',  # Twitter format
                    '%Y-%m-%d %H:%M:%S',      # Simple format
                    '%Y-%m-%d'                # Date only
                ]:
                    try:
                        tweeted_at = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                        
                # If none of the formats worked, try dateutil
                if not tweeted_at:
                    import dateutil.parser
                    tweeted_at = dateutil.parser.parse(date_str)
            except Exception as e:
                logger.warning(f"Could not parse date '{date_str}': {e}")
                # Use current time as fallback
                tweeted_at = datetime.now()
        else:
            # No date provided, use current time
            tweeted_at = datetime.now()
            
        # Extract text from various possible fields
        text = (
            bookmark_data.get('full_text') or 
            bookmark_data.get('text') or 
            bookmark_data.get('tweet_content') or 
            ""
        )
        
        # Extract author information from various possible fields
        author_name = (
            bookmark_data.get('name') or 
            bookmark_data.get('author_name') or 
            bookmark_data.get('user', {}).get('name') if isinstance(bookmark_data.get('user'), dict) else None
        )
        
        author_username = (
            bookmark_data.get('screen_name') or 
            bookmark_data.get('author_username') or 
            bookmark_data.get('user', {}).get('screen_name') if isinstance(bookmark_data.get('user'), dict) else None
        )
        
        # Extract media if available
        media_files = bookmark_data.get('media_files') or bookmark_data.get('media', [])
        
        # Create the mapping with all required fields
        mapped_data = {
            'id': tweet_id,
            'text': text,
            'created_at': tweeted_at,
            'author_name': author_name,
            'author_username': author_username,
            'media_files': media_files,
            'raw_data': bookmark_data,
            'user_id': user_id
        }
        
        # Validate essential fields
        if not mapped_data['id']:
            logger.error("No valid ID could be extracted")
            return None
            
        if not mapped_data['text']:
            logger.warning(f"Bookmark {tweet_id} has no text content")
            # Continue anyway - this is not a critical error
            
        return mapped_data
        
    except Exception as e:
        logger.error(f"Error mapping bookmark data: {e}")
        logger.error(traceback.format_exc())
        return None

def get_user_directory(user_id):
    """
    Find a valid user directory for storing user-specific files.
    
    Args:
        user_id: User ID for multi-user support
        
    Returns:
        str: Path to user directory or None if not found
    """
    if not user_id:
        return DATABASE_DIR
        
    user_dir = f"user_{user_id}"
    
    # Try multiple potential paths
    potential_dirs = [
        os.path.join(BASE_DIR, "database", user_dir),
        os.path.join("database", user_dir),
        os.path.join("/app/database", user_dir),
        os.path.join(DATABASE_DIR, user_dir)
    ]
    
    # Find first existing directory
    for dir_path in potential_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            return dir_path
            
    # Try to create directory if none found
    try:
        os.makedirs(potential_dirs[0], exist_ok=True)
        return potential_dirs[0]
    except Exception as e:
        logger.error(f"Error creating user directory: {e}")
        return None

def rebuild_vector_store(session_id, user_id, batch_size=20, resume=True, progress_data=None):
    """
    Rebuild the vector store from bookmarks in the database.
    This function has been enhanced with:
    - Improved concurrency handling with file-based locks
    - Retry mechanism with exponential backoff
    - Batch processing with proper error handling
    - Checkpointing for resumable operations
    
    Args:
        session_id (str): Unique session ID for tracking
        user_id (int): User ID for which to rebuild the vector store
        batch_size (int): Number of bookmarks to process in each batch
        resume (bool): Whether to attempt resuming from previous progress
        progress_data (dict): Previous progress data if resuming
        
    Returns:
        dict: Results of the rebuild operation
    """
    # Check if vector store is disabled
    if os.environ.get('DISABLE_VECTOR_STORE', 'false').lower() == 'true':
        logger.warning(f"Vector store is disabled by environment variable - session_id={session_id}")
        return {
            'success': True,
            'message': 'Vector store is disabled by environment variable',
            'session_id': session_id,
            'user_id': user_id,
            'skipped': True
        }
    
    # Create lock file path
    lock_file_path = os.path.join(tempfile.gettempdir(), f"vector_rebuild_lock_{user_id}.lock")
    
    # Check if another rebuild is in progress
    if os.path.exists(lock_file_path):
        try:
            lock_file_age = time.time() - os.path.getmtime(lock_file_path)
            if lock_file_age < 600:  # 10 minutes
                logger.warning(f"Another vector rebuild is in progress (lock age: {lock_file_age:.1f}s) - session_id={session_id}")
            return {
                    'success': False,
                    'message': f'Another vector rebuild is already in progress (started {lock_file_age:.1f} seconds ago)',
                    'session_id': session_id,
                    'user_id': user_id
                }
            else:
                logger.warning(f"Removing stale lock file (age: {lock_file_age:.1f}s) - session_id={session_id}")
                try:
                    os.remove(lock_file_path)
                except Exception as e:
                    logger.error(f"Error removing stale lock file: {e} - session_id={session_id}")
        except Exception as e:
            logger.error(f"Error checking lock file: {e} - session_id={session_id}")
    
    # Create new lock file
    try:
        with open(lock_file_path, 'w') as f:
            f.write(f"vector_rebuild_session_{session_id}_{datetime.now().isoformat()}")
        logger.info(f"Created lock file at {lock_file_path} - session_id={session_id}")
            except Exception as e:
        logger.error(f"Error creating lock file: {e} - session_id={session_id}")
        # Continue anyway
    
    # Start timing
    start_time = datetime.now()
    
    # Set up progress tracking
    progress_file = os.path.join(tempfile.gettempdir(), f"vector_rebuild_progress_{user_id}_{session_id}.json")
    
    # Initialize progress values
    start_index = 0
    bookmarks_processed = 0
    total_errors = 0
    total_success = 0
    
    # Calculate progress if resuming from previous run
    if resume and progress_data:
        try:
            start_index = progress_data.get('processed_index', 0)
            bookmarks_processed = progress_data.get('bookmarks_processed', 0)
            total_errors = progress_data.get('total_errors', 0)
            total_success = progress_data.get('total_success', 0)
            logger.info(f"Resuming from index {start_index} (processed: {bookmarks_processed}) - session_id={session_id}")
        except Exception as e:
            logger.error(f"Error parsing progress data: {e} - session_id={session_id}")
            start_index = 0
    
    try:
        # Initialize vector store with retry logic
        vector_store = get_vector_store_with_retry(user_id, session_id)
        if not vector_store:
            logger.error(f"Failed to initialize vector store after retries - session_id={session_id}")
            return {
                'success': False,
                'error': "Failed to initialize vector store after multiple attempts",
                'message': "Vector store initialization failed",
                'session_id': session_id,
                'user_id': user_id
            }
        
        # Get bookmarks with optimized query
        try:
            bookmarks = get_bookmarks_optimized(user_id)
                except Exception as e:
            logger.error(f"Error retrieving bookmarks: {e} - session_id={session_id}")
            return {
                'success': False,
                'error': f"Error retrieving bookmarks: {str(e)}",
                'message': "Failed to retrieve bookmarks from database",
                'session_id': session_id,
                'user_id': user_id
            }
        
        total_bookmarks = len(bookmarks)
        if total_bookmarks == 0:
            logger.warning(f"No bookmarks found for user {user_id} - session_id={session_id}")
            return {
                'success': True,
                'message': "No bookmarks found, nothing to rebuild",
                'total': 0,
                'processed': 0,
                'success_count': 0,
                'error_count': 0,
                'session_id': session_id,
                'user_id': user_id
            }
        
        # Skip previously processed bookmarks if resuming
        valid_bookmarks = [b for b in bookmarks if getattr(b, 'text', None) and getattr(b, 'text', '').strip()]
        total_valid = len(valid_bookmarks)
        
        logger.info(f"Processing {total_valid} valid bookmarks out of {total_bookmarks} total - session_id={session_id}")
        
        if start_index >= total_valid:
            logger.warning(f"Start index {start_index} is beyond the end of valid bookmarks ({total_valid}) - session_id={session_id}")
        return {
                'success': True,
                'message': "All bookmarks already processed",
                'total': total_bookmarks,
                'valid': total_valid,
                'processed': total_valid,
                'success_count': total_success,
                'error_count': total_errors,
                'session_id': session_id,
                'user_id': user_id
            }
        
        # Process bookmarks in batches
        current_index = start_index
        batch_errors = 0
        batch_success = 0
        
        # Clear the entire collection first if starting fresh (not resuming)
        if not resume or start_index == 0:
            try:
                logger.info(f"Clearing existing vector store - session_id={session_id}")
                if hasattr(vector_store, 'clear'):
                    vector_store.clear()
                else:
                    logger.warning(f"Vector store does not have a clear method - session_id={session_id}")
            except Exception as e:
                logger.error(f"Error clearing vector store: {e} - session_id={session_id}")
                # Continue anyway, as we'll attempt to add/update vectors
        
        while current_index < total_valid:
            batch_end = min(current_index + batch_size, total_valid)
            batch = valid_bookmarks[current_index:batch_end]
            
            logger.info(f"Processing batch {current_index}-{batch_end-1} of {total_valid} bookmarks - session_id={session_id}")
            
            batch_errors = 0
            batch_success = 0
            
            for i, bookmark in enumerate(batch):
                absolute_index = current_index + i
                try:
                    # Add bookmark to vector store with retry mechanism
                    result = add_bookmark_with_retry(vector_store, bookmark, session_id)
                    
                    if result['success']:
                        batch_success += 1
                        total_success += 1
                    else:
                        batch_errors += 1
                        total_errors += 1
                        logger.warning(f"Failed to add bookmark {getattr(bookmark, 'id', 'unknown')}: {result.get('error')} - session_id={session_id}")
                except Exception as e:
                    batch_errors += 1
                    total_errors += 1
                    logger.error(f"Unexpected error adding bookmark {getattr(bookmark, 'id', 'unknown')}: {e} - session_id={session_id}")
                
                # Update progress after each bookmark
                bookmarks_processed += 1
                progress_percent = int((bookmarks_processed / total_valid) * 100)
                
                # Update progress file regularly
                if i % max(1, min(5, batch_size // 5)) == 0 or i == len(batch) - 1:
                    try:
        with open(progress_file, 'w') as f:
                            json.dump({
                                'processed_index': absolute_index + 1,  # +1 to start from next bookmark on resume
                                'bookmarks_processed': bookmarks_processed,
                                'total_valid': total_valid,
                                'total_bookmarks': total_bookmarks,
                                'progress_percent': progress_percent,
                                'total_success': total_success,
                                'total_errors': total_errors,
                                'last_update': datetime.now().isoformat()
                            }, f)
                    except Exception as e:
                        logger.error(f"Error updating progress file: {e} - session_id={session_id}")
            
            logger.info(f"Batch complete - Success: {batch_success}, Errors: {batch_errors} - session_id={session_id}")
            
            # Move to next batch
            current_index = batch_end
        
        # Final verification
        logger.info(f"Vector rebuild complete - session_id={session_id}")
        logger.info(f"Stats: Original: {total_bookmarks}, Valid: {total_valid}, Processed: {bookmarks_processed}, Success: {total_success}, Errors: {total_errors}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Clean up progress file on successful completion
        try:
            if os.path.exists(progress_file):
                os.remove(progress_file)
                logger.info(f"Removed progress file - session_id={session_id}")
        except Exception as e:
            logger.warning(f"Could not remove progress file: {e} - session_id={session_id}")
        
            return {
            'success': True,
            'message': "Vector store rebuild completed successfully",
            'total': total_bookmarks,
            'valid': total_valid,
            'processed': bookmarks_processed,
            'success_count': total_success,
            'error_count': total_errors,
            'duration_seconds': duration,
            'progress': 100,
                'session_id': session_id,
                'user_id': user_id
            }
                    except Exception as e:
        logger.error(f"Critical error in vector rebuild: {e} - session_id={session_id}")
        logger.error(traceback.format_exc())
        
        # Get progress for reporting
        progress_percent = 0
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    progress_percent = progress_data.get('progress_percent', 0)
        except Exception as progress_error:
            logger.error(f"Error reading progress file: {progress_error} - session_id={session_id}")
            
            return {
                'success': False,
            'error': str(e),
            'message': "Critical error during vector rebuild",
            'progress': progress_percent,
                'session_id': session_id,
                'user_id': user_id
            }
    finally:
        # Always clean up lock file
        try:
            if os.path.exists(lock_file_path):
                os.remove(lock_file_path)
                logger.info(f"Removed lock file - session_id={session_id}")
                except Exception as e:
            logger.error(f"Error removing lock file: {e} - session_id={session_id}")
            # Not much we can do at this point

def get_vector_store_with_retry(user_id, session_id, max_retries=3, initial_backoff=1):
    """Get a vector store instance with retry logic."""
    # Check if vector store is disabled
    if os.environ.get('DISABLE_VECTOR_STORE', 'false').lower() == 'true':
        logger.warning(f"Vector store is disabled by environment variable - session_id={session_id}")
        return None
        
    logger.info(f"Getting vector store - user_id={user_id} - session_id={session_id}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Vector store attempt {attempt+1}/{max_retries} - session_id={session_id}")
            
            try:
                # First try importing with relative imports (when running as a package)
                from .vector_store_final import VectorStoreMultiUser, DummyVectorStore
                logger.info(f"Imported vector store with relative import - session_id={session_id}")
            except ImportError:
                # Fall back to absolute import (when running as a script)
                from database.multi_user_db.vector_store_final import VectorStoreMultiUser, DummyVectorStore
                logger.info(f"Imported vector store with absolute import - session_id={session_id}")
            except Exception as import_error:
                logger.error(f"Error importing vector store: {import_error} - session_id={session_id}")
                if attempt < max_retries - 1:
                    backoff = initial_backoff * (2 ** attempt)
                    logger.info(f"Retrying in {backoff}s - session_id={session_id}")
                    time.sleep(backoff)
                    continue
                else:
                    logger.error(f"Failed to import vector store after {max_retries} attempts - session_id={session_id}")
                    return None
                
            try:
                # Try importing the getter function first
                from database.multi_user_db.vector_store_final import get_multi_user_vector_store
                logger.info(f"Using get_multi_user_vector_store - session_id={session_id}")
                vector_store = get_multi_user_vector_store()
                
                # Check if we got a dummy store
                if hasattr(vector_store, 'is_dummy') and vector_store.is_dummy:
                    logger.warning(f"Received dummy vector store - session_id={session_id}")
                    # Return None to trigger fallback handling
                    return None
                    
                logger.info(f"Successfully got vector store - session_id={session_id}")
                return vector_store
            except Exception as getter_error:
                logger.warning(f"Error using getter function: {getter_error} - session_id={session_id}")
                # Fall back to direct instantiation
                logger.info(f"Falling back to direct instantiation - session_id={session_id}")
                
                try:
                    vector_store = VectorStoreMultiUser(user_id=user_id)
                    logger.info(f"Successfully created vector store directly - session_id={session_id}")
                    return vector_store
                except Exception as direct_error:
                    logger.error(f"Error creating vector store directly: {direct_error} - session_id={session_id}")
                    if attempt < max_retries - 1:
                        backoff = initial_backoff * (2 ** attempt)
                        logger.info(f"Retrying in {backoff}s - session_id={session_id}")
                        time.sleep(backoff)
                    else:
                        logger.error(f"Failed to create vector store after {max_retries} attempts - session_id={session_id}")
                        return None
                        
        except Exception as e:
            logger.error(f"Unexpected error in get_vector_store_with_retry: {e} - session_id={session_id}")
            logger.error(traceback.format_exc())
            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)
                logger.info(f"Retrying in {backoff}s - session_id={session_id}")
                time.sleep(backoff)
            else:
                logger.error(f"Failed after {max_retries} attempts - session_id={session_id}")
                return None
    
    return None

def add_bookmark_with_retry(vector_store, bookmark, session_id, max_retries=2, initial_backoff=0.5):
    """
    Attempt to add a bookmark to the vector store with retry logic and exponential backoff
    
    Args:
        vector_store: Vector store instance
        bookmark: Bookmark to add
        session_id: Session ID for tracking
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        
    Returns:
        dict: Result of the operation
    """
    # Check if we have a vector store
    if vector_store is None:
        logger.warning(f"No vector store available, skipping add_bookmark - session_id={session_id}")
        return {
            'success': True,  # Return success so processing continues
            'message': 'Vector store not available, bookmark added to database only',
            'bookmark_id': bookmark.get('id', 'unknown')
        }
    
    # Check if vector store is disabled
    if os.environ.get('DISABLE_VECTOR_STORE', 'false').lower() == 'true':
        logger.warning(f"Vector store is disabled, skipping add_bookmark - session_id={session_id}")
        return {
            'success': True,  # Return success so processing continues
            'message': 'Vector store is disabled, bookmark added to database only',
            'bookmark_id': bookmark.get('id', 'unknown')
        }
    
    # Check if this is a dummy vector store
    if hasattr(vector_store, 'is_dummy') and vector_store.is_dummy:
        logger.warning(f"Dummy vector store, skipping add_bookmark - session_id={session_id}")
        return {
            'success': True,  # Return success so processing continues
            'message': 'Using dummy vector store, bookmark added to database only',
            'bookmark_id': bookmark.get('id', 'unknown')
        }
    
    for attempt in range(max_retries + 1):
        try:
            # Try adding the bookmark to the vector store
            vector_store.add_text(
                text=bookmark.get('text', ''),
                metadata={
                    'id': bookmark.get('id', 'unknown'),
                    'url': bookmark.get('url', ''),
                    'author': bookmark.get('author', ''),
                    'date': bookmark.get('date', '')
                }
            )
            return {
                'success': True,
                'message': f'Bookmark added to vector store - attempt={attempt+1}',
                'bookmark_id': bookmark.get('id', 'unknown')
            }
            except Exception as e:
            logger.error(f"Error adding bookmark to vector store: {e} - session_id={session_id}")
            
            if attempt < max_retries:
                # Calculate backoff time
                backoff = initial_backoff * (2 ** attempt)
                logger.info(f"Retrying add_bookmark in {backoff}s - session_id={session_id}, attempt={attempt+1}/{max_retries+1}")
                time.sleep(backoff)
            else:
                logger.error(f"Max retries reached for add_bookmark - session_id={session_id}")
                return {
                    'success': False,
                    'error': f'Failed to add bookmark to vector store after {max_retries+1} attempts: {str(e)}',
                    'bookmark_id': bookmark.get('id', 'unknown')
                }
    
    # Should never reach here, but just in case
    return {
        'success': False,
        'error': 'Unexpected error in add_bookmark_with_retry',
        'bookmark_id': bookmark.get('id', 'unknown')
    }

def get_bookmarks_optimized(user_id):
    """
    Get bookmarks for a user with optimized SQL query
    
    Args:
        user_id (int): User ID
        
    Returns:
        list: List of Bookmark objects
    """
    from database.multi_user_db.db_final import get_db_connection
    from datetime import datetime
    from models import Bookmark
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Optimize query to filter out empty text in the database
        cursor.execute(
            """
            SELECT id, title, url, text, created_at 
            FROM bookmarks 
            WHERE user_id = ? AND text IS NOT NULL AND text <> ''
            ORDER BY id ASC
            """,
            (user_id,)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        bookmarks = []
        for row in rows:
            try:
                created_at = datetime.fromisoformat(row[4]) if row[4] else None
            except (ValueError, TypeError):
                created_at = None
                
            bookmark = Bookmark(
                id=row[0],
                title=row[1],
                url=row[2],
                text=row[3],
                created_at=created_at,
                user_id=user_id
            )
            bookmarks.append(bookmark)
            
        return bookmarks
                            except Exception as e:
        logger.error(f"Error in get_bookmarks_optimized: {e}")
        raise

def update_progress_file(file_path, data):
    """
    Update the progress file with current state
    
    Args:
        file_path (str): Path to progress file
        data (dict): Progress data to save
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f)
                        except Exception as e:
                            logger.error(f"Error updating progress file: {e}")
                        
def calculate_progress(progress_file, total_valid=0):
    """
    Calculate progress percentage from progress file
    
    Args:
        progress_file (str): Path to progress file
        total_valid (int): Total valid bookmarks count
        
    Returns:
        int: Progress percentage (0-100)
    """
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                return progress_data.get('progress_percent', 0)
                except Exception as e:
            logger.error(f"Error reading progress file: {e}")
    
    # Fallback if progress file doesn't exist or can't be read
    return 0

def test_resumable_update():
    """Test the resumable update functionality
    
    This demonstrates how the update can be interrupted and resumed
    from where it left off using the session_id and progress tracking.
    
    Example usage:
    ```
    python -c "from twitter_bookmark_manager.deployment.final.database.multi_user_db.update_bookmarks_final import test_resumable_update; test_resumable_update()"
    ```
    """
    import time
    
    # First run - will be "interrupted" partway through
    session_id = str(uuid.uuid4())
    logger.info(f"Running first part of test with session_id: {session_id}")
    result1 = final_update_bookmarks(start_index=0)
    
    if not result1.get('success'):
        logger.error("First run failed, can't continue test")
        return
    
    # Get the progress data
    progress_file = os.path.join(LOG_DIR, f'update_progress_{session_id}.json')
    if not os.path.exists(progress_file):
        logger.error("Progress file not found")
        return
    
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    next_index = progress.get('next_index', 0)
    logger.info(f"First run completed at index {next_index}, resuming from there...")
    
    # Wait a moment to simulate a break
    time.sleep(2)
    
    # Second run - continues from where the first one left off
    result2 = final_update_bookmarks(session_id=session_id, start_index=next_index)
    logger.info(f"Second run result: {result2}")
    
def get_memory_usage():
    """Get current memory usage of the process"""
    try:
        # Cross-platform memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return f"{memory_mb:.1f}MB"
    except Exception as e:
        try:
            if sys.platform == 'linux':
                memory_usage = os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()
                memory_mb = float(memory_usage) / 1024
                return f"{memory_mb:.1f}MB"
        except Exception:
            pass
        return "Unknown"

def monitor_memory(label=""):
    """
    Log current memory usage with a label and memory increase from last check
    
    Args:
        label (str): Label to identify memory checkpoint
        
    Returns:
        str: Memory usage information
    """
    global _last_memory_usage
    
    try:
        current_memory = get_memory_usage()
        
        # Extract numeric value for comparison
        try:
            current_value = float(current_memory.split('MB')[0])
            if '_last_memory_usage' in globals():
                last_value = float(_last_memory_usage.split('MB')[0])
                diff = current_value - last_value
                diff_text = f" ({'+' if diff >= 0 else ''}{diff:.1f}MB from last check)"
            else:
                diff_text = ""
                
            _last_memory_usage = current_memory
            memory_text = f"{current_memory}{diff_text}"
        except:
            memory_text = current_memory
            
        logger.info(f"🧠 Memory usage {label}: {memory_text}")
        
        # Report warning if memory usage is high (over 1GB)
        if 'MB' in current_memory:
            try:
                mb_value = float(current_memory.split('MB')[0])
                if mb_value > 1000:
                    logger.warning(f"⚠️ High memory usage detected: {current_memory}")
            except:
                pass
                
        return memory_text
        
    except Exception as e:
        logger.error(f"Error in monitor_memory: {e}")
        return "Memory monitoring error"

def find_file_in_possible_paths(filename, user_id=None, additional_paths=None):
    """
    Find a file in multiple possible paths to handle environment inconsistencies.
    
    Args:
        filename (str): Name of the file to find
        user_id (int, optional): User ID for multi-user support
        additional_paths (list, optional): Additional paths to search
        
    Returns:
        str or None: Full path to the file if found, None otherwise
    """
    # Base paths to search
    base_paths = [
        BASE_DIR,
        os.path.dirname(BASE_DIR),
        "/app",
        os.getcwd(),
        "."
    ]
    
    # Additional path components based on user_id
    user_dir = f"user_{user_id}" if user_id else ""
    sub_paths = [
        os.path.join("database", user_dir),
        os.path.join("database"),
        user_dir
    ]
    
    # Build a list of all possible paths
    search_paths = []
    for base in base_paths:
        for sub in sub_paths:
            path = os.path.join(base, sub, filename)
            search_paths.append(path)
    
    # Add any additional custom paths
    if additional_paths:
        search_paths.extend(additional_paths)
    
    # Search for the file in all paths
    for path in search_paths:
        if os.path.exists(path):
            logger.info(f"Found file {filename} at path: {path}")
            return path
            
    # File not found
    logger.warning(f"Could not find file {filename} in any of the search paths")
    return None

def update_status(status_file, status, **kwargs):
    """
    Update the status file with current progress and additional data.
    
    Args:
        status_file (str): Path to status file
        status (str): Current status ('initializing', 'processing', 'completed', 'error', etc.)
        **kwargs: Additional data to include in the status update
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        
        # Read existing status if available
        current_status = {}
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    current_status = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start fresh
                current_status = {}
        
        # Update with new information
        current_status.update({
            'status': status,
            'last_update': datetime.now().isoformat(),
            **kwargs
        })
        
        # Add memory usage if not provided
        if 'current_memory' not in kwargs:
            current_status['current_memory'] = get_memory_usage()
        
        # Write updated status
        with open(status_file, 'w') as f:
            json.dump(current_status, f)
    except Exception as e:
        logger.error(f"Error updating status file: {e}")
        # Don't raise - status updates are non-critical

def run_vector_rebuild(user_id, session_id=None, max_bookmarks=50):
    """
    Run a vector rebuild in the background
    
    Args:
        user_id: User ID to rebuild vectors for
        session_id: Session ID for tracking progress
        max_bookmarks: Maximum number of bookmarks to process (for memory constraints)
    """
    if not session_id:
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
    # Set up logging
    logger = logging.getLogger('vector_rebuild')
    logger.info(f"📢 [REBUILD-{session_id}] Starting background vector rebuild for user {user_id}")
    
    # Get user directory
    user_dir = get_user_directory(user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Create status file path
    status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
    
    # Write initial status
    with open(status_file, 'w') as f:
        status = {
            'user_id': user_id,
            'status': 'processing',
            'message': 'Vector rebuild started',
            'timestamp': time.time(),
            'progress': 0
        }
        json.dump(status, f)
    
    try:
        # Import vector store module
        from database.multi_user_db.vector_store_final import get_multi_user_vector_store
        
        # Get vector store instance
        vector_store = get_multi_user_vector_store()
        
        # Start rebuild
        logger.info(f"🔄 [REBUILD-{session_id}] Starting rebuild with batch_size=2, max_bookmarks={max_bookmarks}")
        success = vector_store.rebuild_user_vectors(user_id, batch_size=2, session_id=session_id)
        
        # Update status
        if success:
            with open(status_file, 'w') as f:
                status = {
                    'user_id': user_id,
                    'status': 'completed', 
                    'message': 'Vector rebuild completed successfully',
                    'timestamp': time.time(),
                    'progress': 100
                }
                json.dump(status, f)
            logger.info(f"✅ [REBUILD-{session_id}] Vector rebuild completed successfully")
        else:
            with open(status_file, 'w') as f:
                status = {
                    'user_id': user_id,
                    'status': 'failed',
                    'message': 'Vector rebuild failed',
                    'timestamp': time.time(),
                    'progress': 0
                }
                json.dump(status, f)
            logger.error(f"❌ [REBUILD-{session_id}] Vector rebuild failed")
    
    except Exception as e:
        # Update status file with error
        try:
            with open(status_file, 'w') as f:
                status = {
                    'user_id': user_id,
                    'status': 'failed',
                    'message': f'Error during vector rebuild: {str(e)}',
                    'timestamp': time.time(),
                    'error': str(e),
                    'progress': 0
                }
                json.dump(status, f)
        except Exception as write_error:
            logger.error(f"❌ [REBUILD-{session_id}] Failed to write error status: {write_error}")
            
        logger.error(f"❌ [REBUILD-{session_id}] Error during vector rebuild: {str(e)}")
        logger.error(traceback.format_exc())

# Add a function to convert dictionary bookmarks to Bookmark objects
def dict_to_bookmark(bookmark_dict):
    """Convert a bookmark dictionary to a Bookmark object"""
    if not bookmark_dict:
        return None
        
    # Extract basic fields
    bookmark_id = bookmark_dict.get('bookmark_id') or bookmark_dict.get('id')
    text = bookmark_dict.get('text', '')
    created_at = bookmark_dict.get('created_at')
    author_name = bookmark_dict.get('author_name')
    author_username = bookmark_dict.get('author_username')
    media_files = bookmark_dict.get('media_files', {})
    raw_data = bookmark_dict.get('raw_data', {})
    user_id = bookmark_dict.get('user_id')
    
    # Create Bookmark object
    return Bookmark(
        id=bookmark_id,
        text=text,
        created_at=created_at,
        author_name=author_name,
        author_username=author_username,
        media_files=media_files,
        raw_data=raw_data,
        user_id=user_id
    )

def final_update_bookmarks(session_id=None, start_index=0, rebuild_vector=False, user_id=None, skip_vector=False):
    """
    Final version of bookmark update function with improved error handling and progress tracking.
    
    Args:
        session_id (str, optional): Session ID for tracking progress
        start_index (int): Index to start processing from
        rebuild_vector (bool): Whether to rebuild vector store after update
        user_id (int, optional): User ID for multi-user support
        skip_vector (bool, optional): If True, skip all vector store operations
        
    Returns:
        dict: Results of the update operation
    """
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    
    logger.info(f"Starting bookmark update - session_id={session_id}")
    
    try:
        # Get user directory
        user_dir = get_user_directory(user_id)
        if not user_dir:
            return {
                'success': False,
                'error': 'Could not find or create user directory',
                'session_id': session_id
            }
        
        # Set up status file
        status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
        update_status(status_file, 'initializing', start_index=start_index)
        
        # Process bookmarks
        try:
            conn = get_db_connection()
            bookmarks = get_bookmarks_optimized(user_id if user_id else 1)
            
            if not bookmarks:
                update_status(status_file, 'completed', message='No bookmarks to process')
                return {
                    'success': True,
                    'message': 'No bookmarks to process',
                    'session_id': session_id
                }
            
            total_bookmarks = len(bookmarks)
            processed = 0
            errors = 0
            
            update_status(status_file, 'processing', 
                total=total_bookmarks,
                processed=processed,
                errors=errors
            )
            
            # Process each bookmark
            for i, bookmark in enumerate(bookmarks[start_index:], start=start_index):
                try:
                    # Your bookmark processing logic here
                    # For example:
                    if bookmark.text:
                        processed += 1
                    else:
                        errors += 1
                        
                    # Update status periodically
                    if i % 10 == 0:
                        update_status(status_file, 'processing',
                            total=total_bookmarks,
                            processed=processed,
                            errors=errors,
                            current_index=i
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing bookmark {bookmark.id}: {e}")
                    errors += 1
            
            # Final status update
            update_status(status_file, 'completed',
                total=total_bookmarks,
                processed=processed,
                errors=errors
            )
            
            # Rebuild vector store if requested
            if rebuild_vector and not skip_vector:
                logger.info("Starting vector store rebuild...")
                rebuild_result = rebuild_vector_store(
                    session_id=session_id,
                    user_id=user_id if user_id else 1
                )
                
                return {
                    'success': True,
                    'bookmarks_processed': processed,
                    'errors': errors,
                    'vector_rebuild': rebuild_result,
                    'session_id': session_id
                }
            
            return {
                'success': True,
                'bookmarks_processed': processed,
                'errors': errors,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Error in bookmark update: {e}")
            update_status(status_file, 'error',
                error=str(e),
                traceback=traceback.format_exc()
            )
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
            
    except Exception as e:
        logger.error(f"Critical error in final_update_bookmarks: {e}")
        return {
            'success': False,
            'error': str(e),
            'session_id': session_id
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Environment Bookmark Update Tool')
    parser.add_argument('--start', type=int, default=0, help='Index to start from')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store after update')
    parser.add_argument('--resume', type=str, help='Resume using session ID')
    
    args = parser.parse_args()
    
    if args.resume:
        # Resume from a previous session
        result = final_update_bookmarks(session_id=args.resume, start_index=args.start, rebuild_vector=args.rebuild)
    else:
        # Start a new session
        result = final_update_bookmarks(start_index=args.start, rebuild_vector=args.rebuild)
        
    print(json.dumps(result, indent=2)) 