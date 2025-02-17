from flask import Flask, jsonify, request
from pathlib import Path
import json

app = Flask(__name__)

@app.route('/test-upload', methods=['POST'])
def test_upload():
    """Test route for file upload without affecting real data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Basic checks
        test_results = {
            'filename': file.filename,
            'is_json': file.filename.endswith('.json'),
            'size': 0,
            'json_valid': False,
            'content_preview': None
        }
        
        # Test JSON validity
        if test_results['is_json']:
            try:
                content = file.read()
                test_results['size'] = len(content)
                json_data = json.loads(content)
                test_results['json_valid'] = True
                # Preview first item if it's a list
                if isinstance(json_data, list) and json_data:
                    test_results['content_preview'] = json_data[0]
            except json.JSONDecodeError:
                test_results['json_valid'] = False
        
        return jsonify({
            'success': True,
            'message': 'Test upload received and validated',
            'test_results': test_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Using different port than main server