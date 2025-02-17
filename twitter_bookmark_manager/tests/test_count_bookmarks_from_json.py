import json
from pathlib import Path

# Get root directory and file path
root_dir = Path(__file__).parent.parent
json_path = root_dir / 'database' / 'twitter_bookmarks.json'

with open(json_path, 'r', encoding='utf-8') as f:
    bookmarks = json.load(f)
    print(f"Total bookmarks: {len(bookmarks)}")