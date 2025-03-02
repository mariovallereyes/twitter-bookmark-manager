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
import shutil

# Import the Bookmark model and db_session
from .db_pa import db_session, get_db_session
from twitter_bookmark_manager.database.models import Bookmark
from .vector_store_pa import VectorStore  # Add import for the vector store

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

def rebuild_vector_store_pa(session_id=None):
    """Rebuild the Qdrant vector store from the PostgreSQL database
    
    This is a critical function that ensures the vector store (Qdrant) is in sync
    with the SQL database. It completely rebuilds the vector store by:
    1. Initializing a fresh vector store
    2. Retrieving all bookmarks from the PostgreSQL database
    3. Adding each bookmark to the vector store
    
    Args:
        session_id (str, optional): Unique identifier for logging purposes
    
    Returns:
        dict: Result summary with counts and status
    """
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        
    logger.info(f"🔄 [REBUILD-{session_id}] Starting vector store (Qdrant) rebuild process")
    start_time = datetime.now()
    
    try:
        # Initialize a new vector store
        vector_store = VectorStore()
        logger.info(f"✅ [REBUILD-{session_id}] Vector store initialized with collection: {vector_store.collection_name}")
        
        # Get all bookmarks from PostgreSQL
        with get_db_session() as session:
            # Count total bookmarks first for logging
            total_count = session.query(Bookmark).count()
            logger.info(f"📊 [REBUILD-{session_id}] Found {total_count} bookmarks in PostgreSQL database")
            
            if total_count == 0:
                logger.warning(f"⚠️ [REBUILD-{session_id}] No bookmarks found in database, nothing to rebuild")
                return {
                    "success": True,
                    "message": "No bookmarks found in database",
                    "postgres_count": 0,
                    "vector_count": 0,
                    "duration_seconds": (datetime.now() - start_time).total_seconds()
                }
            
            # Process bookmarks in batches to avoid memory issues
            BATCH_SIZE = 50
            processed_count = 0
            success_count = 0
            error_count = 0
            
            # Use batched query to reduce memory pressure
            for i in range(0, total_count, BATCH_SIZE):
                batch = session.query(Bookmark).limit(BATCH_SIZE).offset(i).all()
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
                            'author_name': bookmark.author_name or ''
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
                "is_in_sync": total_count == vector_count
            }
            
    except Exception as e:
        error_msg = f"Error during vector store rebuild: {str(e)}"
        logger.error(f"❌ [REBUILD-{session_id}] {error_msg}")
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }

def detect_update_loop(progress_file, max_loop_count=3):
    """
    Detect and prevent infinite update loops by analyzing the progress file
    
    Args:
        progress_file (str): Path to the progress file
        max_loop_count (int): Maximum number of times an index can be repeated
        
    Returns:
        tuple: (is_in_loop, loop_data) where loop_data contains diagnostic information
    """
    try:
        if not os.path.exists(progress_file):
            return False, {"message": "No progress file found"}
            
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            
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
        with open(progress_file, 'w') as f:
            json.dump(progress, f)
            
        if is_in_loop:
            logger.warning(f"⚠️ Loop detected! Index {current_index} has been processed {loop_detection['count'][str(current_index)]} times")
            
        return is_in_loop, {
            "current_index": current_index,
            "occurrences": loop_detection['count'].get(str(current_index), 0),
            "recent_indices": loop_detection['indices'],
            "timestamps": loop_detection['timestamps'],
            "is_in_loop": is_in_loop
        }
            
    except Exception as e:
        logger.error(f"Error in loop detection: {str(e)}")
        return False, {"error": str(e)}

# Enhance the pa_update_bookmarks function with better debugging and vector store updates
def pa_update_bookmarks(session_id=None, start_index=0, rebuild_vector=False):
    """PythonAnywhere-specific function to update bookmarks from JSON file
    
    Args:
        session_id (str, optional): Unique identifier for this update session. If not provided, one will be generated.
        start_index (int, optional): Index to start/resume processing from. Defaults to 0.
        rebuild_vector (bool, optional): If True, rebuild the vector store at the end of the update. Defaults to False.
        
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
        
        # If rebuild_vector is True, only rebuild vector store and return
        if rebuild_vector:
            logger.info(f"🔄 Rebuild vector store flag is set, initiating rebuild...")
            rebuild_result = rebuild_vector_store_pa(session_id)
            return {
                'success': rebuild_result.get('success', False),
                'message': 'Vector store rebuild completed',
                'rebuild_result': rebuild_result,
                'session_id': session_id,
                'is_complete': True
            }
            
        # Check for update loops
        is_in_loop, loop_data = detect_update_loop(progress_file)
        if is_in_loop:
            logger.warning(f"⚠️ [UPDATE-{session_id}] Update loop detected at index {loop_data['current_index']}! Skipping this bookmark and continuing with next one")
            
            # Increment the start_index to skip the problematic bookmark
            if start_index == loop_data['current_index']:
                start_index += 1
                logger.info(f"⏭️ [UPDATE-{session_id}] Adjusted start_index to {start_index} to skip problematic bookmark")
            
            # Update the loop detection data in the progress file to reset the counter for this index
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    # Reset counter for the problematic index
                    if 'loop_detection' in progress and 'count' in progress['loop_detection']:
                        problem_index = str(loop_data['current_index'])
                        if problem_index in progress['loop_detection']['count']:
                            progress['loop_detection']['count'][problem_index] = 0
                            logger.info(f"🔄 [UPDATE-{session_id}] Reset loop counter for index {problem_index}")
                    
                    # Update the progress file
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f)
                except Exception as e:
                    logger.error(f"❌ [UPDATE-{session_id}] Error updating progress file during loop handling: {e}")
            
            # Continue processing rather than returning
            # Note: Removing the return statement allows processing to continue

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
                    logger.info(f"🔄 [UPDATE-{session_id}] Database update complete, now updating vector store...")
                    rebuild_result = rebuild_vector_store_pa(session_id)
                    logger.info(f"✅ [UPDATE-{session_id}] Vector store update completed")
                
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
                    'vector_rebuild_result': rebuild_result if is_complete else None
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
    import argparse
    parser = argparse.ArgumentParser(description='PythonAnywhere Bookmark Update Tool')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store from PostgreSQL database')
    parser.add_argument('--start', type=int, default=0, help='Start index for processing bookmarks')
    args = parser.parse_args()
    
    if args.rebuild:
        logger.info("Rebuilding vector store only")
        result = rebuild_vector_store_pa()
        logger.info(f"Rebuild result: {result}")
    else:
        logger.info(f"Updating bookmarks starting from index {args.start}")
        result = pa_update_bookmarks(start_index=args.start)
        logger.info(f"Update result: {result}") 