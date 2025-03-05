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

# Import the Bookmark model and db_session
from .db_final import db_session, get_db_session
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

def map_bookmark_data(data, user_id=None):
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
            'raw_data': data,
            'user_id': user_id
        }
        
        # Validate required fields
        if not mapped_data['id']:
            logger.error("No valid ID could be extracted")
            return None
            
        return mapped_data
        
    except Exception as e:
        logger.error(f"Error mapping bookmark data: {e}")
        return None

def rebuild_vector_store(session_id=None, user_id=None, batch_size=25, force_full_rebuild=False, max_duration_minutes=30):
    """Rebuild the vector store from the database
    
    This function rebuilds the vector store by adding all bookmarks from the 
    PostgreSQL database to the Qdrant vector store.
    
    Args:
        session_id: Optional session identifier for logging purposes
        user_id: Optional user ID for multi-user support
        batch_size: Number of bookmarks to process in a batch (smaller = less memory)
        force_full_rebuild: Force a full rebuild even if incremental is possible
        max_duration_minutes: Maximum duration in minutes before timing out
        
    Returns:
        Dict with rebuild results and status
    """
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        
    logger.info(f"🔄 [REBUILD-{session_id}] Starting vector store (Qdrant) rebuild process")
    monitor_memory("at start of vector store rebuild")
    
    if user_id:
        logger.info(f"👤 [REBUILD-{session_id}] Processing for user_id: {user_id}")
    start_time = datetime.now()
    
    # Create progress file path
    rebuild_progress_file = os.path.join(LOG_DIR, f'rebuild_progress_{session_id}.json')
    if isinstance(rebuild_progress_file, Path):
        rebuild_progress_exists = rebuild_progress_file.exists()
    else:
        rebuild_progress_exists = os.path.exists(rebuild_progress_file)
    
    # Load previous progress if exists
    processed_ids = set()
    last_processed_index = 0
    
    if rebuild_progress_exists and not force_full_rebuild:
        try:
            with open(rebuild_progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_ids = set(progress_data.get('processed_ids', []))
                last_processed_index = progress_data.get('last_processed_index', 0)
                logger.info(f"📊 [REBUILD-{session_id}] Resuming from index {last_processed_index} with {len(processed_ids)} already processed")
        except Exception as e:
            logger.warning(f"⚠️ [REBUILD-{session_id}] Could not load progress file: {e}. Starting fresh.")
            processed_ids = set()
            last_processed_index = 0
    
    try:
        # Initialize a new vector store with user_id if provided
        monitor_memory("before vector store initialization")
        vector_store = VectorStore(user_id=user_id)
        logger.info(f"✅ [REBUILD-{session_id}] Vector store initialized with collection: {vector_store.collection_name}")
        monitor_memory("after vector store initialization")
        
        # Get all bookmarks from PostgreSQL, filtered by user_id if provided
        with get_db_session() as session:
            # Create query that filters by user_id if provided
            query = session.query(Bookmark)
            if user_id:
                query = query.filter(Bookmark.user_id == user_id)
                
            # Count total bookmarks first for logging
            total_count = query.count()
            logger.info(f"📊 [REBUILD-{session_id}] Found {total_count} bookmarks in PostgreSQL database")
            
            if total_count == 0:
                logger.warning(f"⚠️ [REBUILD-{session_id}] No bookmarks found in database, nothing to rebuild")
                return {
                    "success": True,
                    "message": "No bookmarks found in database",
                    "postgres_count": 0,
                    "vector_count": 0,
                    "duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "user_id": user_id
                }
            
            # Process bookmarks in smaller batches to avoid memory issues
            BATCH_SIZE = min(batch_size, 25)  # Cap at 25 for safety
            processed_count = 0
            success_count = len(processed_ids)  # Count previously processed
            error_count = 0
            processed_this_session = 0
            
            # Calculate timeout time
            timeout_time = start_time + timedelta(minutes=max_duration_minutes)
            
            # Use batched query to reduce memory pressure
            for i in range(last_processed_index, total_count, BATCH_SIZE):
                # Check if we've exceeded the maximum allowed time
                current_time = datetime.now()
                if current_time > timeout_time:
                    logger.warning(f"⚠️ [REBUILD-{session_id}] Reached maximum allowed time ({max_duration_minutes} minutes). Stopping with progress saved.")
                    # Save progress and return partial success
                    duration = (current_time - start_time).total_seconds()
                    return {
                        "success": True,
                        "message": f"Timeout after {duration:.1f} seconds, progress saved",
                        "postgres_count": total_count,
                        "vector_count": success_count,
                        "successful_additions": success_count,
                        "processed_this_session": processed_this_session,
                        "errors": error_count,
                        "duration_seconds": duration,
                        "is_complete": False,
                        "is_timeout": True,
                        "next_index": i,
                        "user_id": user_id
                    }
                
                # Clear any previous batch data from memory
                import gc
                gc.collect()
                
                monitor_memory(f"before loading batch {i // BATCH_SIZE + 1}")
                batch = query.limit(BATCH_SIZE).offset(i).all()
                batch_size = len(batch)
                
                logger.info(f"🔄 [REBUILD-{session_id}] Processing batch {i // BATCH_SIZE + 1}: {batch_size} bookmarks")
                
                batch_success = 0
                batch_errors = 0
                batch_skipped = 0
                
                # Process each bookmark in the batch
                for j, bookmark in enumerate(batch):
                    try:
                        # Extract the tweet URL and ID
                        tweet_url = bookmark.raw_data.get('tweet_url', 'unknown') if bookmark.raw_data else 'unknown'
                        tweet_id = tweet_url.split('/')[-1] if tweet_url else None
                        
                        # Skip if already processed (for incremental rebuilds)
                        if tweet_id in processed_ids or tweet_url in processed_ids:
                            batch_skipped += 1
                            continue
                        
                        # Prepare metadata for the vector store
                        metadata = {
                            'tweet_url': tweet_url,
                            'screen_name': bookmark.author_username or '',
                            'author_name': bookmark.author_name or '',
                            'user_id': user_id if user_id else bookmark.user_id
                        }
                        
                        # Skip empty text
                        if not bookmark.text or len(bookmark.text.strip()) == 0:
                            logger.warning(f"⚠️ [REBUILD-{session_id}] Skipping bookmark {tweet_id} with empty text")
                            continue
                        
                        # Add bookmark to vector store using its ID
                        vector_store.add_bookmark(
                            bookmark_id=tweet_id,
                            text=bookmark.text or '',
                            metadata=metadata
                        )
                        
                        processed_ids.add(tweet_id)
                        processed_ids.add(tweet_url) 
                        success_count += 1
                        batch_success += 1
                        processed_this_session += 1
                        
                    except Exception as e:
                        error_msg = f"Error adding bookmark {tweet_id} to vector store: {str(e)}"
                        logger.error(f"❌ [REBUILD-{session_id}] {error_msg}")
                        error_count += 1
                        batch_errors += 1
                
                processed_count = i + batch_size
                last_processed_index = i + batch_size
                
                # Log batch completion with progress
                progress = (processed_count / total_count) * 100
                logger.info(f"✅ [REBUILD-{session_id}] Completed batch {i // BATCH_SIZE + 1}: {batch_success} success, {batch_errors} errors, {batch_skipped} skipped")
                logger.info(f"✅ [REBUILD-{session_id}] Overall progress: {processed_count}/{total_count} ({progress:.1f}%)")
                
                # Save progress to file for potential resuming
                progress_data = {
                    'session_id': session_id,
                    'last_processed_index': last_processed_index,
                    'processed_ids': list(processed_ids),
                    'success_count': success_count,
                    'error_count': error_count,
                    'last_update': datetime.now().isoformat()
                }
                
                with open(rebuild_progress_file, 'w') as f:
                    json.dump(progress_data, f)
                
                # Memory checkpoint - force garbage collection and log memory usage
                gc.collect()
                monitor_memory(f"after batch {i // BATCH_SIZE + 1}")
            
            # Final verification - get collection info
            collection_info = vector_store.get_collection_info()
            vector_count = collection_info.get('vectors_count', 0)
            
            # Clear progress file after successful completion
            if os.path.exists(rebuild_progress_file):
                try:
                    os.remove(rebuild_progress_file)
                    logger.info(f"🧹 [REBUILD-{session_id}] Removed progress file after successful completion")
                except Exception as e:
                    logger.warning(f"⚠️ [REBUILD-{session_id}] Could not remove progress file: {e}")
            
            # Log summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"🏁 [REBUILD-{session_id}] Vector store rebuild completed in {duration:.2f} seconds")
            logger.info(f"📊 [REBUILD-{session_id}] Summary:")
            logger.info(f"  - User ID: {user_id if user_id else 'All users'}")
            logger.info(f"  - PostgreSQL bookmark count: {total_count}")
            logger.info(f"  - Vector store count: {vector_count}")
            logger.info(f"  - Successful additions: {success_count}")
            logger.info(f"  - Processed this session: {processed_this_session}")
            logger.info(f"  - Errors: {error_count}")
            monitor_memory("at end of vector store rebuild")
            
            # Check for count mismatch
            if success_count != vector_count:
                mismatch_msg = f"Count mismatch! Added: {success_count}, Vector count: {vector_count}"
                logger.warning(f"⚠️ [REBUILD-{session_id}] {mismatch_msg}")
            
            return {
                "success": True,
                "postgres_count": total_count,
                "vector_count": vector_count,
                "successful_additions": success_count,
                "processed_this_session": processed_this_session,
                "errors": error_count,
                "duration_seconds": duration,
                "is_complete": True,
                "is_timeout": False,
                "is_in_sync": success_count == vector_count,
                "user_id": user_id
            }
            
    except Exception as e:
        error_msg = f"Error during vector store rebuild: {str(e)}"
        logger.error(f"❌ [REBUILD-{session_id}] {error_msg}")
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "user_id": user_id
        }

def detect_update_loop(progress_file, max_loop_count=3, user_id=None):
    """
    Detect and prevent infinite update loops by analyzing the progress file
    
    Args:
        progress_file (str): Path to the progress file
        max_loop_count (int): Maximum number of times an index can be repeated
        user_id: Optional user ID for multi-user support
        
    Returns:
        tuple: (is_in_loop, loop_data) where loop_data contains diagnostic information
    """
    try:
        if not os.path.exists(progress_file):
            return False, {"message": "No progress file found"}
            
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            
        # Check if progress file belongs to current user
        if user_id and progress.get('user_id') != user_id:
            # Create new progress file for this user
            logger.warning(f"⚠️ Progress file belongs to different user. Creating new progress for user {user_id}")
            return False, {"message": f"Progress file belongs to different user. Creating new for {user_id}"}
            
        # Check if we have a 'loop_detection' field already
        loop_detection = progress.get('loop_detection', {})
        current_index = progress.get('last_processed_index')
        
        if not current_index:
            return False, {"message": "No current index in progress file"}
            
        # Initialize loop detection if not present
        if 'indices' not in loop_detection:
            loop_detection['indices'] = []
            loop_detection['timestamps'] = []
            loop_detection['count'] = {}
            
        # Add current state to loop detection
        loop_detection['indices'].append(current_index)
        loop_detection['timestamps'].append(datetime.now().isoformat())
        
        # Only keep last 10 entries
        if len(loop_detection['indices']) > 10:
            loop_detection['indices'] = loop_detection['indices'][-10:]
            loop_detection['timestamps'] = loop_detection['timestamps'][-10:]
            
        # Count occurrences of each index
        loop_detection['count'][str(current_index)] = loop_detection['count'].get(str(current_index), 0) + 1
        
        # Check if we're in a loop
        is_in_loop = loop_detection['count'].get(str(current_index), 0) >= max_loop_count
        
        # Update the progress file with loop detection data
        progress['loop_detection'] = loop_detection
        # Add user_id to progress
        if user_id:
            progress['user_id'] = user_id
            
        with open(progress_file, 'w') as f:
            json.dump(progress, f)
            
        if is_in_loop:
            logger.warning(f"⚠️ Loop detected! Index {current_index} has been processed {loop_detection['count'][str(current_index)]} times")
            
        return is_in_loop, {
            "current_index": current_index,
            "occurrences": loop_detection['count'].get(str(current_index), 0),
            "recent_indices": loop_detection['indices'],
            "timestamps": loop_detection['timestamps'],
            "is_in_loop": is_in_loop,
            "user_id": user_id
        }
            
    except Exception as e:
        logger.error(f"Error in loop detection: {str(e)}")
        return False, {"error": str(e)}

# Enhance the update_bookmarks function with better debugging and vector store updates
def final_update_bookmarks(session_id=None, start_index=0, rebuild_vector=False, user_id=None):
    """
    Update bookmarks database from the JSON file
    
    Args:
        session_id: Optional session identifier for tracking
        start_index: Optional starting index for batch processing
        rebuild_vector: Whether to rebuild the vector store after updating
        user_id: Optional user ID for multi-user support
        
    Returns:
        Dict with update results and status
    """
    # Generate a session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    
    # Monitor initial memory state
    monitor_memory("at start of update_bookmarks")
        
    # Set up user-specific paths
    if user_id:
        logger.info(f"👤 [UPDATE-{session_id}] Processing for user_id: {user_id}")
        
        # Define user-specific directories
        from pathlib import Path
        user_dir = f"user_{user_id}"
        
        # Use absolute paths for Railway deployment
        # There appears to be a path inconsistency where upload uses /database/user_X but update looks in /app/database/user_X
        app_prefix = '/app'
        
        # First try with app prefix (Railway production environment)
        base_dir = Path(os.path.join(app_prefix, DATABASE_DIR.lstrip('/')))
        database_dir = base_dir / user_dir
        
        # Ensure directories exist
        database_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up user-specific files with Path objects
        progress_file = database_dir / 'update_progress.json'
        bookmarks_file = database_dir / 'twitter_bookmarks.json'
        
        # Check if file exists
        if not bookmarks_file.exists():
            # If not found with app prefix, try without app prefix
            base_dir_no_prefix = Path(DATABASE_DIR)
            database_dir_no_prefix = base_dir_no_prefix / user_dir
            bookmarks_file_no_prefix = database_dir_no_prefix / 'twitter_bookmarks.json'
            
            if bookmarks_file_no_prefix.exists():
                # Use the path without prefix if file exists there
                logger.info(f"✅ [UPDATE-{session_id}] Found bookmarks file without app prefix: {bookmarks_file_no_prefix}")
                bookmarks_file = bookmarks_file_no_prefix
                progress_file = database_dir_no_prefix / 'update_progress.json'
                database_dir = database_dir_no_prefix
    else:
        # Use default paths
        progress_file = os.path.join(LOG_DIR, f'update_progress_{session_id}.json')
        bookmarks_file = os.path.join(DATABASE_DIR, 'twitter_bookmarks.json')
    
    logger.info(f"Starting final environment bookmark update process for session {session_id} from index {start_index}")
    logger.info(f"Using bookmarks file: {bookmarks_file}")
    
    # Check if file exists - final check
    if isinstance(bookmarks_file, Path):
        file_exists = bookmarks_file.exists()
    else:
        file_exists = os.path.exists(bookmarks_file)
        
    if not file_exists:
        # Try a few more path combinations as a last resort
        possible_paths = [
            os.path.join('/app', DATABASE_DIR.lstrip('/'), user_dir, 'twitter_bookmarks.json'),
            os.path.join('/app/database', user_dir, 'twitter_bookmarks.json'),
            os.path.join(DATABASE_DIR, user_dir, 'twitter_bookmarks.json'),
            os.path.join('/database', user_dir, 'twitter_bookmarks.json')
        ]
        
        for alt_path in possible_paths:
            if os.path.exists(alt_path):
                logger.info(f"✅ [UPDATE-{session_id}] Found bookmarks file at alternative path: {alt_path}")
                bookmarks_file = alt_path
                file_exists = True
                break
                
    if not file_exists:
        logger.error(f"❌ [UPDATE-{session_id}] Bookmarks file not found: {bookmarks_file}")
        # Log additional debugging info
        logger.error(f"❌ [UPDATE-{session_id}] DATABASE_DIR setting: {DATABASE_DIR}")
        logger.error(f"❌ [UPDATE-{session_id}] Tried multiple path combinations but none worked")
        
        # List files in potential directories to help debug
        for dir_to_check in ['/app/database', '/database', DATABASE_DIR]:
            if os.path.exists(dir_to_check):
                logger.info(f"📁 [UPDATE-{session_id}] Contents of {dir_to_check}:")
                try:
                    for item in os.listdir(dir_to_check):
                        logger.info(f"   - {item}")
                except Exception as e:
                    logger.error(f"❌ [UPDATE-{session_id}] Error listing directory {dir_to_check}: {str(e)}")
                    
            user_specific_dir = os.path.join(dir_to_check, user_dir)
            if os.path.exists(user_specific_dir):
                logger.info(f"📁 [UPDATE-{session_id}] Contents of {user_specific_dir}:")
                try:
                    for item in os.listdir(user_specific_dir):
                        logger.info(f"   - {item}")
                except Exception as e:
                    logger.error(f"❌ [UPDATE-{session_id}] Error listing directory {user_specific_dir}: {str(e)}")
        
        return {
            'success': False,
            'error': f'Bookmarks file not found: {bookmarks_file}',
            'session_id': session_id,
            'user_id': user_id
        }
    
    # If rebuild_vector is True, only rebuild vector store and return
    if rebuild_vector:
        logger.info("Rebuild vector flag is set, rebuilding vector store")
        rebuild_result = rebuild_vector_store(session_id=session_id, user_id=user_id, batch_size=15)
        return {
            'success': rebuild_result["success"],
            'message': 'Vector store rebuild complete',
            'details': rebuild_result,
            'session_id': session_id,
            'user_id': user_id
        }
        
    # Check for update loops
    is_in_loop, loop_data = detect_update_loop(progress_file, user_id=user_id)
    if is_in_loop:
        logger.warning(f"⚠️ [UPDATE-{session_id}] Update loop detected! Breaking out of loop and forcing vector rebuild")
        # Force a vector rebuild to break the loop
        rebuild_result = rebuild_vector_store(session_id=session_id, user_id=user_id, batch_size=15, force_full_rebuild=True)
        
        # Reset progress file to start fresh
        if isinstance(progress_file, Path):
            exists = progress_file.exists()
        else:
            exists = os.path.exists(progress_file)
        
        if exists:
            try:
                # Backup the file first
                backup_file = f"{progress_file}.loop_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(progress_file, backup_file)
                logger.info(f"✅ [UPDATE-{session_id}] Created backup of progress file at {backup_file}")
                
                # Reset the progress file
                os.remove(progress_file)
                logger.info(f"✅ [UPDATE-{session_id}] Removed progress file to break update loop")
            except Exception as e:
                logger.error(f"❌ [UPDATE-{session_id}] Error handling progress file during loop recovery: {e}")
        
        return {
            'success': True,
            'message': 'Update loop detected and resolved with vector rebuild',
            'loop_data': loop_data,
            'session_id': session_id,
            'user_id': user_id
        }
            
    # Track processed IDs to avoid duplicates
    processed_ids = set()
    
    # Statistics for reporting
    stats = {
        'total_processed': 0,
        'new_count': 0,
        'updated_count': 0,
        'errors': 0
    }
    
    # Check for existing progress to resume
    current_progress = {}
    
    if isinstance(progress_file, Path):
        progress_exists = progress_file.exists()
    else:
        progress_exists = os.path.exists(progress_file)
        
    if progress_exists and start_index > 0:
        try:
            with open(progress_file, 'r') as f:
                current_progress = json.load(f)
                # Get processed IDs from progress file
                if 'processed_ids' in current_progress:
                    processed_ids = set(current_progress['processed_ids'])
                    logger.info(f"📊 [UPDATE-{session_id}] Resuming from index {start_index} with {len(processed_ids)} already processed IDs")
        except Exception as e:
            logger.error(f"❌ [UPDATE-{session_id}] Error reading progress: {e}")
    
    logger.info(f"📋 [UPDATE-{session_id}] STEP 3: Initiating database update with pa_update_bookmarks")
    
    # Load the bookmark data from the JSON file
    try:
        # Monitor memory before loading bookmarks
        monitor_memory("before loading bookmarks.json")
        
        with open(bookmarks_file, 'r', encoding='utf-8') as f:
            bookmark_data = json.load(f)
            
        # Get bookmark array and count for logging
        if isinstance(bookmark_data, dict) and 'bookmarks' in bookmark_data:
            new_bookmarks = bookmark_data['bookmarks']
            total_bookmarks = len(new_bookmarks)
        else:
            # Try handling as a direct array
            new_bookmarks = bookmark_data
            total_bookmarks = len(new_bookmarks)
            
        logger.info(f"Loaded {total_bookmarks} bookmarks from JSON file")
        monitor_memory("after loading bookmarks.json")
        
        # Validate and adjust start_index
        if start_index >= total_bookmarks:
            error_msg = f"Start index {start_index} exceeds total bookmarks {total_bookmarks}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # Process bookmarks in smaller batches
        BATCH_SIZE = 10  # Reduced batch size for better error handling
        current_batch = []
        
        # Get existing bookmarks once using raw SQL instead of ORM
        existing_bookmarks = {}
        
        try:
            with get_db_session() as session:
                # First attempt to get just id and data, which are the minimum we need
                bookmark_query = """
                    SELECT id, raw_data, user_id
                    FROM bookmarks
                """
                params = {}
                
                # Add user_id filter if provided
                if user_id:
                    bookmark_query += " WHERE user_id = :user_id"
                    params["user_id"] = user_id
                    
                # Execute the query and create Bookmark objects
                result = session.execute(text(bookmark_query), params)
                for row in result:
                    # Create simplified bookmark object with minimal data
                    bookmark_data = {
                        'id': row[0],
                        'raw_data': json.loads(row[1]) if row[1] else {}
                    }
                    
                    # Only add user_id if it's available
                    if len(row) > 2:
                        bookmark_data['user_id'] = row[2]
                    
                    bookmark = Bookmark(**bookmark_data)
                    if bookmark and bookmark.raw_data and 'tweet_url' in bookmark.raw_data:
                        existing_bookmarks[bookmark.raw_data['tweet_url']] = bookmark
                
                logger.info(f"Found {len(existing_bookmarks)} existing bookmarks in database")
                monitor_memory("after loading existing bookmarks")
        except Exception as e:
            logger.warning(f"Error querying bookmarks table: {e}")
            logger.info("Proceeding with empty existing bookmarks")
            
        # Process each bookmark
        batch_count = 0
        for i, bookmark_data in enumerate(new_bookmarks, start_index):
            try:
                tweet_url = bookmark_data.get('tweet_url')
                if not tweet_url or tweet_url in processed_ids:
                    continue
                    
                mapped_data = map_bookmark_data(bookmark_data, user_id)
                if not mapped_data:
                    stats['errors'] += 1
                    continue
                
                # Add to current batch
                current_batch.append((tweet_url, mapped_data, bookmark_data))
                
                # Process batch if full or last item
                if len(current_batch) >= BATCH_SIZE or i == len(new_bookmarks) + start_index - 1:
                    batch_count += 1
                    
                    # Monitor memory before processing batch
                    monitor_memory(f"before processing batch {batch_count}")
                    
                    batch_success = 0
                    batch_errors = 0
                    
                    # Process each item in batch with separate transactions
                    for url, data, raw in current_batch:
                        # Use a new session for each bookmark to isolate transactions
                        with get_db_session() as item_session:
                            try:
                                if url not in existing_bookmarks:
                                    # Use raw SQL to insert new bookmark
                                    insert_query = """
                                        INSERT INTO bookmarks 
                                        (id, raw_data, user_id)
                                        VALUES (:id, :raw_data, :user_id)
                                    """
                                    # Convert Python data types to SQL-compatible types
                                    insert_params = {
                                        'id': data.get('id'),
                                        'raw_data': json.dumps(raw),
                                        'user_id': data.get('user_id')
                                    }
                                    item_session.execute(text(insert_query), insert_params)
                                    stats['new_count'] += 1
                                else:
                                    # Use raw SQL to update existing bookmark
                                    existing = existing_bookmarks[url]
                                    update_query = """
                                        UPDATE bookmarks
                                        SET raw_data = :raw_data
                                        WHERE id = :id
                                    """
                                    update_params = {
                                        'id': existing.id,
                                        'raw_data': json.dumps(raw)
                                    }
                                    item_session.execute(text(update_query), update_params)
                                    stats['updated_count'] += 1
                                
                                # Commit the individual transaction
                                item_session.commit()
                                processed_ids.add(url)
                                batch_success += 1
                                
                            except Exception as e:
                                # Rollback the individual transaction on error
                                item_session.rollback()
                                logger.error(f"Error processing bookmark {url}: {e}")
                                stats['errors'] += 1
                                batch_errors += 1
                    
                    # Update progress file with session_id
                    stats['total_processed'] = i + 1
                    progress_data = {
                        'session_id': session_id,
                        'last_processed_index': i + 1,
                        'processed_ids': list(processed_ids),
                        'stats': stats,
                        'last_update': datetime.now().isoformat()
                    }
                    
                    # Log batch results
                    logger.info(f"Batch completed: {batch_success} successful, {batch_errors} failed")
                    
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)
                    
                    # Clear batch
                    current_batch = []
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Log progress with memory usage info
                    progress_percent = ((i + 1) / total_bookmarks) * 100
                    monitor_memory(f"after processing batch {batch_count}")
                    logger.info(f"✅ [UPDATE-{session_id}] Progress: {progress_percent:.1f}% ({i + 1}/{total_bookmarks})")
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                stats['errors'] += 1
                continue
            
        # Return progress information
        is_complete = stats['total_processed'] >= total_bookmarks
        logger.info(f"🏁 [UPDATE-{session_id}] Update {'completed' if is_complete else 'in progress'}: {stats['total_processed']}/{total_bookmarks} bookmarks processed")
        monitor_memory("after database updates")
        
        result_data = {
            'success': True,
            'session_id': session_id,
            'progress': {
                'total': total_bookmarks,
                'processed': stats['total_processed'],
                'new': stats['new_count'],
                'updated': stats['updated_count'],
                'errors': stats['errors'],
                'percent_complete': (stats['total_processed'] / total_bookmarks) * 100 if total_bookmarks > 0 else 0
            },
            'is_complete': is_complete,
            'next_index': stats['total_processed'] if not is_complete else None,
            'vector_store_updated': False,
            'user_id': user_id
        }
        
        if is_complete or stats['new_count'] > 0 or stats['updated_count'] > 0:
            # Rebuild vector store if update is complete or if we added/updated any bookmarks
            try:
                # Use smaller batch size (15) for vector store rebuild to reduce memory usage
                # Don't force full rebuild to allow incremental processing
                logger.info(f"Performing memory-optimized vector store rebuild (completed={is_complete}, new={stats['new_count']}, updated={stats['updated_count']})")
                rebuild_result = rebuild_vector_store(session_id=session_id, user_id=user_id, batch_size=15, force_full_rebuild=False)
                result_data['vector_store_updated'] = True
                result_data['vector_rebuild_result'] = rebuild_result
                
                if not rebuild_result["success"]:
                    logger.warning(f"Vector rebuild warning: {rebuild_result.get('error', 'Unknown error')}")
                else:
                    logger.info("Vector store rebuild successful")
            except Exception as vector_error:
                error_msg = f"Vector store rebuild error: {str(vector_error)}"
                logger.error(error_msg)
                result_data['vector_rebuild_error'] = error_msg
        
        return result_data
        
    except Exception as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg)
        
        # Try to perform vector store rebuild even if there was an error in the main process
        vector_rebuild_result = None
        if stats['new_count'] > 0 or stats['updated_count'] > 0:
            try:
                logger.info(f"Attempting memory-optimized vector store rebuild despite errors (new={stats['new_count']}, updated={stats['updated_count']})")
                vector_rebuild_result = rebuild_vector_store(session_id=session_id, user_id=user_id, batch_size=15, force_full_rebuild=False)
                if not vector_rebuild_result["success"]:
                    logger.warning(f"Vector rebuild warning: {vector_rebuild_result.get('error', 'Unknown error')}")
                else:
                    logger.info("Vector store rebuild successful despite errors in main process")
            except Exception as vector_error:
                logger.error(f"Vector store rebuild error: {str(vector_error)}")
        
        return {
            'success': False,
            'error': error_msg,
            'session_id': session_id,
            'progress': {
                'last_index': start_index,
                'stats': stats
            },
            'user_id': user_id,
            'vector_rebuild_result': vector_rebuild_result
        }

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
    """Get current memory usage of the process in MB"""
    try:
        # Fallback to ps command for Linux/Mac systems
        try:
            memory_kb = os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()
            memory_mb = float(memory_kb) / 1024
            return f"{memory_mb:.1f}MB"
        except:
            return "Unknown"
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return "Unknown"

def monitor_memory(label=""):
    """Log current memory usage with a label"""
    memory = get_memory_usage()
    logger.info(f"🧠 Memory usage {label}: {memory}")
    return memory

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