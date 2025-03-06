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
            
        # Map the data to our model fields
        mapped_data = {
            'id': tweet_id,
            'raw_data': bookmark_data,
            'tweeted_at': tweeted_at
        }
        
        # Add user_id if provided
        if user_id:
            mapped_data['user_id'] = user_id
            
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

def rebuild_vector_store(session_id=None, user_id=None, batch_size=50, force_full_rebuild=False):
    """
    Rebuild vector store for bookmark embeddings with improved robustness.
    
    Args:
        session_id (str, optional): Session ID for tracking progress
        user_id (int, optional): User ID for multi-user support
        batch_size (int, optional): Batch size for vector processing. Defaults to 50.
        force_full_rebuild (bool, optional): Whether to force a full rebuild. Defaults to False.
        
    Returns:
        dict: Result dictionary with information about the rebuild process
    """
    try:
        logger.info(f"Starting vector store rebuild for user {user_id} with session {session_id}")
        
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        monitor_memory("at start of rebuild_vector_store")
        
        # Initialize embedding model with reduced resources
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            
            logger.info("Initializing embedding model with reduced resources")
            model_kwargs = {'device': 'cpu'}
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs=model_kwargs
            )
            logger.info("Embedding model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f"Error initializing embedding model: {str(e)}",
                'session_id': session_id
            }
        
        # Set up document store path
        try:
            # Get bookmarks from database
            with get_db_session() as session:
                query = session.query(Bookmark)
                
                # Filter by user_id if provided
                if user_id:
                    query = query.filter(Bookmark.user_id == user_id)
                    
                # Get count first to determine batches
                total_bookmarks = query.count()
                logger.info(f"Found {total_bookmarks} bookmarks to process for vector rebuild")
                
                if total_bookmarks == 0:
                    logger.warning("No bookmarks found to rebuild vectors")
                    return {
                        'success': True,
                        'message': "No bookmarks found to rebuild vectors",
                        'session_id': session_id
                    }
                
                # Find valid base directory
                base_dir = get_user_directory(user_id)
                if not base_dir:
                    logger.error(f"Could not find valid base directory for user {user_id}")
                    return {
                        'success': False,
                        'error': f"Could not find valid base directory for user {user_id}",
                        'session_id': session_id
                    }
                
                # Set up paths
                document_store_dir = os.path.join(base_dir, "document_store")
                os.makedirs(document_store_dir, exist_ok=True)
                
                document_store_path = os.path.join(document_store_dir, "faiss_index")
                
                # Back up existing index if it exists
                if os.path.exists(document_store_path) and not os.path.isdir(document_store_path):
                    backup_path = f"{document_store_path}_backup_{int(time.time())}"
                    try:
                        shutil.copy2(document_store_path, backup_path)
                        logger.info(f"Backed up existing index to {backup_path}")
                    except Exception as e:
                        logger.warning(f"Could not back up existing index: {e}")
                
                # Process bookmarks in batches
                total_processed = 0
                documents = []
                
                # Use query batching to avoid loading all bookmarks at once
                for bookmark_offset in range(0, total_bookmarks, batch_size):
                    batch = query.limit(batch_size).offset(bookmark_offset).all()
                    batch_documents = []
                    
                    for bookmark in batch:
                        try:
                            # Extract text content from bookmark
                            tweet_text = ""
                            if bookmark.raw_data:
                                raw_data = bookmark.raw_data
                                tweet_text = raw_data.get('full_text') or raw_data.get('text') or ""
                                
                                # Add author info if available
                                author = raw_data.get('author_name') or raw_data.get('name')
                                if author:
                                    tweet_text = f"Author: {author}\n{tweet_text}"
                                    
                            # Skip empty content
                            if not tweet_text.strip():
                                logger.warning(f"Empty content for bookmark {bookmark.id}, skipping")
                                continue
                            
                            # Create document with metadata
                            from langchain.schema import Document
                            document = Document(
                                page_content=tweet_text,
                                metadata={
                                    "id": bookmark.id,
                                    "user_id": bookmark.user_id,
                                    "source": "twitter_bookmark"
                                }
                            )
                            batch_documents.append(document)
                            
                        except Exception as e:
                            logger.error(f"Error processing bookmark {bookmark.id} for vector embedding: {e}")
                            continue
                    
                    # Add batch to documents
                    documents.extend(batch_documents)
                    total_processed += len(batch_documents)
                    
                    # Log progress
                    progress_percent = min(100, (total_processed / total_bookmarks) * 100)
                    logger.info(f"Vector rebuild progress: {progress_percent:.1f}% ({total_processed}/{total_bookmarks})")
                    
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    
                    # Monitor memory
                    monitor_memory(f"after processing batch of {len(batch_documents)} documents")
                
                # Create vector store with batching and better error handling
                try:
                    from langchain.vectorstores import FAISS
                    
                    logger.info(f"Creating FAISS vector store with {len(documents)} documents")
                    vector_store = FAISS.from_documents(documents, embedding_model)
                    
                    # Save vector store
                    logger.info(f"Saving vector store to {document_store_path}")
                    vector_store.save_local(document_store_path)
                    
                    # Calculate duration
                    duration_seconds = time.time() - start_time
                    logger.info(f"Vector store rebuild completed in {duration_seconds:.2f} seconds")
                    
                    return {
                        'success': True,
                        'message': f"Vector store rebuilt successfully with {total_processed} documents",
                        'total_processed': total_processed,
                        'duration_seconds': duration_seconds,
                        'session_id': session_id
                    }
                    
                except Exception as e:
                    logger.error(f"Error creating or saving vector store: {e}")
                    logger.error(traceback.format_exc())
                    return {
                        'success': False,
                        'error': f"Error creating or saving vector store: {str(e)}",
                        'session_id': session_id
                    }
        
        except Exception as e:
            logger.error(f"Error during vector store rebuild: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f"Error during vector store rebuild: {str(e)}",
                'session_id': session_id
            }
            
    except Exception as e:
        logger.error(f"Unexpected error in rebuild_vector_store: {e}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': f"Unexpected error in rebuild_vector_store: {str(e)}",
            'session_id': session_id
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
    Update bookmarks database from JSON file with improved ORM approach.
    
    Args:
        session_id (str, optional): Unique ID for this update session. Generated if not provided.
        start_index (int, optional): Index to start/resume processing from. Defaults to 0.
        rebuild_vector (bool, optional): Whether to rebuild the vector store. Defaults to False.
        user_id (int, optional): User ID for multi-user support. Defaults to None.
        
    Returns:
        dict: Result dictionary with progress information and success status
    """
    try:
        # Track execution time
        start_time = time.time()
        
        # Generate a session ID if none provided
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
            
        # Monitor memory at start
        monitor_memory(f"start of update session {session_id}")
        
        # Set up user directory
        user_dir = f"user_{user_id}" if user_id else ""
        
        # Find the bookmark file using our robust path finder
        bookmarks_file = find_file_in_possible_paths(
            'twitter_bookmarks.json', 
            user_id=user_id,
            additional_paths=[
                os.path.join('/app', str(DATABASE_DIR).lstrip('/'), user_dir, 'twitter_bookmarks.json'),
                os.path.join('/app/database', user_dir, 'twitter_bookmarks.json'),
                os.path.join(DATABASE_DIR, user_dir, 'twitter_bookmarks.json'),
                os.path.join('/database', user_dir, 'twitter_bookmarks.json')
            ]
        )
        
        # Find or create the base directory for this user
        database_dir = get_user_directory(user_id)
        if not database_dir:
            logger.error(f"Could not find or create valid database directory for user {user_id}")
            return {
                'success': False, 
                'error': 'Could not find or create valid database directory',
                'session_id': session_id,
                'user_id': user_id
            }
        
        # Set up paths for progress tracking
        progress_file = os.path.join(database_dir, 'update_progress.json')
        
        logger.info(f"Starting bookmark update process for session {session_id} from index {start_index}")
        if bookmarks_file:
            logger.info(f"Using bookmarks file: {bookmarks_file}")
        else:
            logger.error(f"❌ [UPDATE-{session_id}] Bookmarks file not found")
            logger.error(f"❌ [UPDATE-{session_id}] DATABASE_DIR setting: {DATABASE_DIR}")
            
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
                'error': 'Bookmarks file not found',
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
        
        if os.path.exists(progress_file) and start_index > 0:
            try:
                with open(progress_file, 'r') as f:
                    current_progress = json.load(f)
                    # Get processed IDs from progress file
                    if 'processed_ids' in current_progress:
                        processed_ids = set(current_progress['processed_ids'])
                        logger.info(f"📊 [UPDATE-{session_id}] Resuming from index {start_index} with {len(processed_ids)} already processed IDs")
            except Exception as e:
                logger.error(f"❌ [UPDATE-{session_id}] Error reading progress: {e}")
        
        logger.info(f"📋 [UPDATE-{session_id}] STEP 3: Initiating database update")
        
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
            
            # Get existing bookmark URLs once using ORM - simplified version with URL as the key
            existing_bookmarks = {}
            
            try:
                # Use ORM approach to get existing bookmarks
                with get_db_session() as session:
                    query = session.query(Bookmark)
                    
                    # Filter by user_id if provided
                    if user_id:
                        query = query.filter(Bookmark.user_id == user_id)
                        
                    # Execute the query
                    for bookmark in query.all():
                        if bookmark.raw_data and 'tweet_url' in bookmark.raw_data:
                            existing_bookmarks[bookmark.raw_data['tweet_url']] = bookmark
                    
                    logger.info(f"Found {len(existing_bookmarks)} existing bookmarks in database")
                    monitor_memory("after loading existing bookmarks")
            except Exception as e:
                logger.warning(f"Error querying bookmarks table: {e}")
                logger.info("Proceeding with empty existing bookmarks")
            
            # Process bookmarks in smaller batches
            BATCH_SIZE = 10  # Small batch size for better error handling
            current_batch = []
            batch_count = 0
            
            # Begin processing bookmarks
            for i, bookmark_data in enumerate(new_bookmarks[start_index:], start_index):
                try:
                    # Get or generate tweet_url for tracking
                    tweet_url = bookmark_data.get('tweet_url')
                    
                    # If no tweet_url but we have a tweet ID, create one
                    if not tweet_url and (tweet_id := bookmark_data.get('id')):
                        tweet_url = f"https://twitter.com/i/status/{tweet_id}"
                        bookmark_data['tweet_url'] = tweet_url
                    
                    # We'll use this key for tracking duplicates 
                    tracking_key = tweet_url if tweet_url else f"item_{i}"
                    
                    # Skip if we've already processed this URL in this session
                    if tracking_key in processed_ids:
                        logger.info(f"Skipping duplicate bookmark at index {i}: {tracking_key}")
                        continue
                    
                    # Map the data
                    mapped_data = map_bookmark_data(bookmark_data, user_id)
                    if not mapped_data:
                        logger.error(f"Failed to map bookmark at index {i}")
                        stats['errors'] += 1
                        continue
                    
                    # Add to current batch
                    current_batch.append((tracking_key, mapped_data, bookmark_data))
                    
                    # Process batch if full or last item
                    if len(current_batch) >= BATCH_SIZE or i == len(new_bookmarks) - 1 + start_index:
                        batch_count += 1
                        
                        # Monitor memory before processing batch
                        monitor_memory(f"before processing batch {batch_count}")
                        
                        batch_success = 0
                        batch_errors = 0
                        
                        # Process entire batch in a single transaction
                        with get_db_session() as session:
                            try:
                                # Process each item in batch
                                for url, data, raw in current_batch:
                                    try:
                                        # Check if bookmark exists
                                        if url not in existing_bookmarks:
                                            # Create new bookmark using ORM
                                            new_bookmark = Bookmark(**data)
                                            session.add(new_bookmark)
                                            stats['new_count'] += 1
                                        else:
                                            # Update existing bookmark using ORM
                                            existing = existing_bookmarks[url]
                                            for key, value in data.items():
                                                setattr(existing, key, value)
                                            stats['updated_count'] += 1
                                            
                                        # Add to processed IDs
                                        processed_ids.add(url)
                                        batch_success += 1
                                    except Exception as e:
                                        # Log individual bookmark errors
                                        logger.error(f"Error processing bookmark {url}: {e}")
                                        stats['errors'] += 1
                                        batch_errors += 1
                                        # Continue with next bookmark - don't break the whole batch
                                
                                # Commit the whole batch
                                session.commit()
                                
                            except Exception as e:
                                # Only rollback on batch-level exceptions
                                session.rollback()
                                logger.error(f"Error processing batch {batch_count}: {e}")
                                logger.error(traceback.format_exc())
                                stats['errors'] += len(current_batch)
                                batch_errors += len(current_batch)
                        
                        # Update progress file
                        stats['total_processed'] = i + 1
                        progress_data = {
                            'session_id': session_id,
                            'last_processed_index': i + 1,
                            'processed_ids': list(processed_ids),
                            'stats': stats,
                            'last_update': datetime.now().isoformat()
                        }
                        
                        try:
                            with open(progress_file, 'w') as f:
                                json.dump(progress_data, f)
                        except Exception as e:
                            logger.error(f"Error updating progress file: {e}")
                        
                        # Log batch results
                        logger.info(f"Batch {batch_count}: {batch_success} succeeded, {batch_errors} failed")
                        
                        # Clear batch
                        current_batch = []
                        
                        # Force garbage collection to free memory
                        import gc
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error in batch processing loop: {e}")
                    logger.error(traceback.format_exc())
                    stats['errors'] += 1
            
            # Calculate duration
            duration_seconds = time.time() - start_time
            
            logger.info(f"Update completed in {duration_seconds:.2f} seconds")
            logger.info(f"Stats: {stats['new_count']} new, {stats['updated_count']} updated, {stats['errors']} errors")
            
            # Rebuild vector store if requested
            vector_rebuilt = False
            rebuild_result = None
            if rebuild_vector:
                logger.info("Rebuilding vector store as requested")
                rebuild_result = rebuild_vector_store(session_id=session_id, user_id=user_id)
                vector_rebuilt = rebuild_result.get('success', False)
            
            # Return success response
            return {
                'success': True,
                'message': 'Database update completed',
                'processed_this_session': stats['total_processed'],
                'new_bookmarks': stats['new_count'],
                'updated_bookmarks': stats['updated_count'],
                'errors': stats['errors'],
                'duration_seconds': duration_seconds,
                'session_id': session_id,
                'vector_rebuilt': vector_rebuilt,
                'rebuild_result': rebuild_result,
                'is_complete': True,
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error updating bookmarks: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'session_id': session_id,
                'user_id': user_id
            }
    
    except Exception as e:
        logger.error(f"Unhandled exception in final_update_bookmarks: {e}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'session_id': session_id,
            'user_id': user_id
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
    """
    Get current memory usage of the process in MB with improved platform support.
    Works on Windows, Linux, and Mac.
    
    Returns:
        str: Memory usage in MB with platform info
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        platform_name = psutil.POSIX and "POSIX" or "Windows"
        return f"{memory_mb:.1f}MB ({platform_name})"
    except ImportError:
        # Fallback if psutil is not available
        try:
            if os.name == 'posix':
                # Linux/Mac
                memory_kb = os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()
                memory_mb = float(memory_kb) / 1024
                return f"{memory_mb:.1f}MB (POSIX-fallback)"
            elif os.name == 'nt':
                # Windows
                import ctypes
                kernel32 = ctypes.windll.kernel32
                process_handle = kernel32.OpenProcess(0x1000, False, os.getpid())
                memory_info = ctypes.c_size_t()
                kernel32.GetProcessMemoryInfo(process_handle, ctypes.byref(memory_info), ctypes.sizeof(memory_info))
                kernel32.CloseHandle(process_handle)
                memory_mb = memory_info.value / 1024 / 1024
                return f"{memory_mb:.1f}MB (Windows-fallback)"
            else:
                return "Unknown platform"
        except:
            return "Unable to determine memory usage"
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return f"Error ({str(e)})"

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