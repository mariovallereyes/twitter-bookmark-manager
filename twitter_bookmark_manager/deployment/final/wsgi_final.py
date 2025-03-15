"""
Bare minimum WSGI file that provides a stable application no matter what.
"""

import os
import sys
import logging
from flask import Flask, jsonify, render_template_string

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('wsgi.log')  # File handler
    ]
)
logger = logging.getLogger('wsgi_final')

# Disable all complex features
os.environ['DISABLE_VECTOR_STORE'] = 'true'
os.environ['FLASK_DEBUG'] = 'false'

# Create an extremely minimal Flask application
app = Flask(__name__)
application = app  # WSGI entry point

@app.route('/')
def index():
    """Minimal home page."""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Twitter Bookmark Manager - Emergency Mode</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .message { background: #f4f4f4; border-left: 4px solid #0066cc; padding: 10px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Twitter Bookmark Manager</h1>
            <div class="message">
                <h2>Emergency Mode Active</h2>
                <p>The application is running in emergency mode due to startup issues.</p>
                <p>Full functionality is temporarily disabled until we resolve the issues.</p>
                <p>The original application code has been preserved in wsgi_final_backup.py</p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/-/health')
def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({"status": "ok", "mode": "emergency"}), 200

@app.route('/api/health')
def api_health():
    """API health check endpoint."""
    logger.info("API health check requested")
    return jsonify({"status": "ok", "mode": "emergency"}), 200

@app.route('/<path:path>')
def catch_all(path):
    """Catch all routes."""
    logger.info(f"Catch-all route accessed: {path}")
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Twitter Bookmark Manager - Emergency Mode</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .message { background: #f4f4f4; border-left: 4px solid #0066cc; padding: 10px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Twitter Bookmark Manager</h1>
            <div class="message">
                <h2>Emergency Mode Active</h2>
                <p>The application is running in emergency mode due to startup issues.</p>
                <p>Full functionality is temporarily disabled until we resolve the issues.</p>
                <p>The original application code has been preserved in wsgi_final_backup.py</p>
            </div>
        </div>
    </body>
    </html>
    """)

logger.info("Bare WSGI application initialized successfully") 