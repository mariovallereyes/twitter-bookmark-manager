"""
Final environment bookmark update script.
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
import shutil

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

def rebuild_vector_store(session_id=None, user_id=None):
    """Rebuild the vector store from the database
    
    This function rebuilds the vector store by adding all bookmarks from the 
    PostgreSQL database to the Qdrant vector store.
    
    Args:
        session_id: Optional session identifier for logging purposes
        user_id: Optional user ID for multi-user support
        
    Returns:
        Dict with rebuild results and status
    """
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        
    logger.info(f"🔄 [REBUILD-{session_id}] Starting vector store (Qdrant) rebuild process")
    if user_id:
        logger.info(f"👤 [REBUILD-{session_id}] Processing for user_id: {user_id}")
    start_time = datetime.now()
    
    try:
        # Initialize a new vector store with user_id if provided
        vector_store = VectorStore(user_id=user_id)
        logger.info(f"✅ [REBUILD-{session_id}] Vector store initialized with collection: {vector_store.collection_name}")
        
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
            
            # Process bookmarks in batches to avoid memory issues
            BATCH_SIZE = 50
            processed_count = 0
            success_count = 0
            error_count = 0
            
            # Use batched query to reduce memory pressure
            for i in range(0, total_count, BATCH_SIZE):
                batch = query.limit(BATCH_SIZE).offset(i).all()
                batch_size = len(batch)
                
                logger.info(f"🔄 [REBUILD-{session_id}] Processing batch {i // BATCH_SIZE + 1}: {batch_size} bookmarks")
                
                # Process each bookmark in the batch
                for bookmark in batch:
                    try:
                        # Extract the tweet URL from raw_data for logging
                        tweet_url = bookmark.raw_data.get('tweet_url', 'unknown') if bookmark.raw_data else 'unknown'
                        
                        # Prepare metadata for the vector store
                        metadata = {
                            'tweet_url': tweet_url,
                            'screen_name': bookmark.author_username or '',
                            'author_name': bookmark.author_name or '',
                            'user_id': user_id if user_id else bookmark.user_id
                        }
                        
                        # Add bookmark to vector store using its ID
                        vector_store.add_bookmark(
                            bookmark_id=str(bookmark.id),
                            text=bookmark.text or '',
                            metadata=metadata
                        )
                        
                        success_count += 1
                        
                        # Detailed logging every 10 bookmarks
                        if success_count % 10 == 0:
                            memory_usage = os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()
                            logger.debug(f"🔍 [REBUILD-{session_id}] Added bookmark {success_count}/{total_count}: {tweet_url} (Memory: {memory_usage}KB)")
                            
                    except Exception as e:
                        error_msg = f"Error adding bookmark {bookmark.id} to vector store: {str(e)}"
                        logger.error(f"❌ [REBUILD-{session_id}] {error_msg}")
                        error_count += 1
                
                processed_count += batch_size
                
                # Log batch completion with progress
                progress = (processed_count / total_count) * 100
                logger.info(f"✅ [REBUILD-{session_id}] Completed batch {i // BATCH_SIZE + 1}: {processed_count}/{total_count} ({progress:.1f}%)")
                
                # Memory checkpoint - force garbage collection
                import gc
                gc.collect()
            
            # Final verification - get collection info
            collection_info = vector_store.get_collection_info()
            vector_count = collection_info.get('vectors_count', 0)
            
            # Log summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"🏁 [REBUILD-{session_id}] Vector store rebuild completed in {duration:.2f} seconds")
            logger.info(f"📊 [REBUILD-{session_id}] Summary:")
            logger.info(f"  - User ID: {user_id if user_id else 'All users'}")
            logger.info(f"  - PostgreSQL bookmark count: {total_count}")
            logger.info(f"  - Vector store count: {vector_count}")
            logger.info(f"  - Successful additions: {success_count}")
            logger.info(f"  - Errors: {error_count}")
            
            # Check for count mismatch
            if total_count != vector_count:
                mismatch_msg = f"Count mismatch! PostgreSQL: {total_count}, Vector: {vector_count}"
                logger.warning(f"⚠️ [REBUILD-{session_id}] {mismatch_msg}")
            else:
                logger.info(f"✅ [REBUILD-{session_id}] Databases are in sync")
            
            return {
                "success": True,
                "postgres_count": total_count,
                "vector_count": vector_count,
                "successful_additions": success_count,
                "errors": error_count,
                "duration_seconds": duration,
                "is_in_sync": total_count == vector_count,
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
        
    # Set up user-specific paths
    if user_id:
        logger.info(f"👤 [UPDATE-{session_id}] Processing for user_id: {user_id}")
        
        # Define user-specific directories
        from pathlib import Path
        user_dir = f"user_{user_id}"
        
        # Use relative paths from the current directory
        # Railway deployment has file paths without /app prefix when using the Path object
        base_dir = Path(DATABASE_DIR)
        database_dir = base_dir / user_dir
        
        # Ensure directories exist
        database_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up user-specific files with Path objects
        progress_file = database_dir / 'update_progress.json'
        bookmarks_file = database_dir / 'twitter_bookmarks.json'
    else:
        # Use default paths
        progress_file = os.path.join(LOG_DIR, f'update_progress_{session_id}.json')
        bookmarks_file = os.path.join(DATABASE_DIR, 'twitter_bookmarks.json')
    
    logger.info(f"Starting final environment bookmark update process for session {session_id} from index {start_index}")
    logger.info(f"Using bookmarks file: {bookmarks_file}")
    logger.info(f"Using progress file: {progress_file}")
    
    # Check if bookmarks file exists
    if not os.path.exists(bookmarks_file):
        logger.error(f"❌ [UPDATE-{session_id}] Bookmarks file not found: {bookmarks_file}")
        return {
            'success': False,
            'error': f'Bookmarks file not found: {bookmarks_file}',
            'session_id': session_id,
            'user_id': user_id
        }
    
    # If rebuild_vector is True, only rebuild vector store and return
    if rebuild_vector:
        logger.info("Rebuild vector flag is set, rebuilding vector store")
        rebuild_result = rebuild_vector_store(session_id, user_id)
        return {
            'success': rebuild_result[0],
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
        rebuild_result = rebuild_vector_store(session_id, user_id)
        
        # Reset progress file to start fresh
        if os.path.exists(progress_file):
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
            'message': 'Update loop detected and resolved with vector store rebuild',
            'loop_data': loop_data,
            'rebuild_result': rebuild_result,
            'session_id': session_id,
            'is_complete': True,
            'user_id': user_id
        }

    # Load progress if exists
    current_progress = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                current_progress = json.load(f)
            
            # Check if progress belongs to current user
            if user_id and current_progress.get('user_id') != user_id:
                logger.info(f"Progress file exists but belongs to different user, creating new for user {user_id}")
                current_progress = {'user_id': user_id}
            elif user_id:
                logger.info(f"Loaded progress for user {user_id}: {current_progress}")
            else:
                logger.info(f"Loaded progress: {current_progress}")
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
    
    # Add user_id to progress if not present
    if user_id and 'user_id' not in current_progress:
        current_progress['user_id'] = user_id
    
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
    logger.info(f"Reading bookmarks from {progress_file}")
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
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
                        
                        # Log progress with memory usage info
                        progress_percent = ((i + 1) / total_bookmarks) * 100
                        memory_usage = os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()
                        logger.info(f"✅ [UPDATE-{session_id}] Progress: {progress_percent:.1f}% ({i + 1}/{total_bookmarks}) - Memory: {memory_usage}KB")
                        
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    stats['errors'] += 1
                    continue
            
            # Return progress information
            is_complete = stats['total_processed'] >= total_bookmarks
            logger.info(f"🏁 [UPDATE-{session_id}] Update {'completed' if is_complete else 'in progress'}: {stats['total_processed']}/{total_bookmarks} bookmarks processed")
            
            if is_complete:
                # If update is complete, update the vector store too
                logger.info("Performing final vector store rebuild")
                rebuild_result = rebuild_vector_store(session_id, user_id)
                if not rebuild_result[0]:
                    logger.warning(f"Vector rebuild warning: {rebuild_result[1]}")
                else:
                    logger.info("Vector store rebuild successful")
            
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
                'next_index': stats['total_processed'] if not is_complete else None,
                'vector_store_updated': is_complete,
                'vector_rebuild_result': rebuild_result if is_complete else None,
                'user_id': user_id
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
            },
            'user_id': user_id
        }
        
    except Exception as e:
        logger.error(f"Update error: {str(e)}")
        return {'success': False, 'error': str(e), 'session_id': session_id if session_id else None, 'user_id': user_id}

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