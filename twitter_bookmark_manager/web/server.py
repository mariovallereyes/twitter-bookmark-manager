import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from core.search import BookmarkSearch
from datetime import datetime
from werkzeug.utils import secure_filename
import json
from pathlib import Path
import shutil
import logging
from core.chat.engine import BookmarkChat
from core.universal_bookmark_extractor import BookmarkExtractor
from core.dashboard.dashboard_routes import dashboard  # Import the dashboard blueprint

logger = logging.getLogger(__name__)

app = Flask(__name__)
search = BookmarkSearch()
chat_engine = BookmarkChat(search_engine=search)

# Register the dashboard blueprint
app.register_blueprint(dashboard)

# RESULTS_PER_PAGE = 50

app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER='temp_uploads'
)

@app.route('/')
def index():
    """Home page with search interface"""
    categories = search.get_categories()
    # Get 5 most recent bookmarks for homepage
    latest_tweets = search.get_all_bookmarks(limit=5)
    
    # Format the latest tweets consistently
    formatted_latest = [{
        'id': str(tweet['id']),
        'text': tweet['text'],
        'author_username': tweet['author'].replace('@', ''),
        'categories': tweet['categories'],
        'created_at': tweet['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    } for tweet in latest_tweets]
    
    return render_template('index.html', 
                         categories=categories,
                         results=[],
                         latest_tweets=formatted_latest,
                         total_results=0,
                         showing_results=0,
                         total_tweets=search.get_total_tweets())

@app.route('/search')
def search_bookmarks():
    """Search endpoint"""
    query = request.args.get('q', '')
    user = request.args.get('user', '')  # This line is already there
    selected_categories = request.args.getlist('categories[]')
    
    # Get all results - Add this condition
    if user:
        all_results = search.search_by_user(user)  # New method to add in search.py
    else:
        all_results = search.search(
            query=query,
            categories=selected_categories if selected_categories else None,
            limit=1000
        )
    
    # Format results
    formatted_results = []
    for result in all_results:
        formatted_results.append({
            'id': str(result['id']),
            'text': result['text'],
            'author_username': result['author'].replace('@', ''),
            'categories': result['categories'],
            'created_at': result['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return render_template('index.html',
                         categories=search.get_categories(),
                         results=formatted_results,
                         query=query,
                         total_results=len(formatted_results),
                         showing_results=len(formatted_results),
                         total_tweets=search.get_total_tweets())

@app.route('/recent')
def recent():
    """Show recent bookmarks"""
    all_results = search.get_all_bookmarks(limit=1000)
    
    # Format results to ensure correct ID and username
    formatted_results = []
    for result in all_results:
        formatted_results.append({
            'id': str(result['id']),  # Ensure ID is a string
            'text': result['text'],
            'author_username': result['author'].replace('@', ''),  # Remove @ if present
            'categories': result['categories'],
            'created_at': result['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return render_template('index.html',
                         categories=search.get_categories(),
                         results=formatted_results,
                         is_recent=True,
                         total_results=len(formatted_results),
                         showing_results=len(formatted_results))

@app.route('/upload-bookmarks', methods=['POST'])
def upload_bookmarks():
    """Handle bookmark JSON file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Only JSON files are allowed'}), 400
        
        # Test file validity without moving anything
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        try:
            json.loads(file_content)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON file'}), 400
        
        # Create temp directory if it doesn't exist
        Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        temp_path = Path(app.config['UPLOAD_FOLDER']) / secure_filename(file.filename)
        file.save(str(temp_path))
        
        # Target file location
        current_file = Path('database/twitter_bookmarks.json')
        
        # Create database/json_history directory if it doesn't exist
        Path('database/json_history').mkdir(exist_ok=True, parents=True)
        
        # Backup current file first if it exists
        if current_file.exists():
            backup_date = datetime.now().strftime("%Y%m%d")
            history_file = Path('database/json_history') / f'twitter_bookmarks_{backup_date}.json'
            
            # Check if we already did a backup today
            if history_file.exists():
                return jsonify({'error': 'Already processed a file today'}), 400
            
            # Copy the current file to history
            try:
                shutil.copy2(str(current_file), str(history_file))
            except Exception as e:
                print(f"Error copying file to history: {str(e)}")
                return jsonify({'error': 'Error backing up current database'}), 500
        
        # Move the uploaded file to the target location
        try:
            shutil.move(str(temp_path), str(current_file))
            return jsonify({'message': 'File processed successfully'})
        except Exception as e:
            print(f"Error moving new file: {str(e)}")
            return jsonify({'error': 'Error updating database file'}), 500
        
    except Exception as e:
        # Log the error (you can add proper logging)
        print(f"Error in upload_bookmarks: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()

@app.route('/chat')
def chat_interface():
    """Render chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        message = request.json.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Use the existing chat_engine instance
        response = await chat_engine.process_message(message)
        
        # Format the response properly for the frontend
        return jsonify({
            'success': True,
            'response': response['text'],  # Extract just the text for the main response
            'model': response['model'],    # Pass the model type
            'bookmarks': response.get('bookmarks_used', [])  # Pass any bookmarks that were found
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Chat error: {str(e)}"
        }), 500

@app.route('/update-database', methods=['POST'])
def update_database():
    """Handle database update after file upload"""
    try:
        # Check if the JSON file exists
        json_file_path = Path('database/twitter_bookmarks.json')
        if not json_file_path.exists():
            return jsonify({
                'error': 'No Twitter bookmarks JSON file found. Please upload a file first.'
            }), 400
        
        # Import required modules
        from database.update_bookmarks import update_bookmarks
        from core.process_categories import CategoryProcessor
        
        # Step 1: Update SQL with any new bookmarks
        try:
            update_result = update_bookmarks(rebuild_vectors=False)
            print(f"Update result: {update_result}")
        except Exception as e:
            print(f"Error updating bookmarks: {str(e)}")
            return jsonify({
                'error': f'Error updating SQL database: {str(e)}'
            }), 500
        
        # Step 2: Process categories for new bookmarks
        try:
            processor = CategoryProcessor()
            category_results = processor.process_all_bookmarks()
        except Exception as e:
            print(f"Error processing categories: {str(e)}")
            return jsonify({
                'error': f'Error processing categories: {str(e)}'
            }), 500
        
        # Step 3: Rebuild vector store to ensure sync
        try:
            update_bookmarks(rebuild_vectors=True)
        except Exception as e:
            print(f"Error rebuilding vector store: {str(e)}")
            return jsonify({
                'error': f'Error rebuilding vector store: {str(e)}'
            }), 500
        
        return jsonify({
            'message': 'Database updated successfully',
            'steps': [
                'File validated',
                'SQL database updated',
                'Categories assigned',
                'Vector store rebuilt',
                'Update completed successfully!'
            ]
        })
        
    except Exception as e:
        print(f"Error updating database: {str(e)}")
        return jsonify({
            'error': f'Error updating database: {str(e)}',
            'steps': ['Error occurred during update']
        }), 500

@app.route('/get-extraction-script')
def get_extraction_script():
    """Endpoint to serve the bookmark extraction script."""
    extractor = BookmarkExtractor()
    return jsonify(extractor.get_injection_wrapper())

if __name__ == '__main__':
    app.run(debug=True)