from core.data_ingestion import BookmarkIngester
from pathlib import Path
import json
import logging
from database.db import get_db_session
from database.models import Bookmark

# Completely disable all logging
logging.disable(logging.CRITICAL)

def main():
    try:
        # Load data first to inspect
        with open('database/twitter_bookmarks.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Show first few tweet URLs and extracted IDs
        print("First 3 bookmarks:")
        for i in range(min(3, len(data))):
            url = data[i].get('tweet_url', 'No URL')
            extracted_id = url.split('/')[-1] if url else 'No ID'
            print(f"{i+1}. URL: {url}")
            print(f"   ID: {extracted_id}")
            print(f"   Text: {data[i].get('full_text', 'No text')[:100]}...")
            print()
        
        # Process bookmarks
        ingester = BookmarkIngester(
            json_path='database/twitter_bookmarks.json',
            media_dir=Path('media')
        )
        results = ingester.process_all_bookmarks()
        
        # Check database
        with get_db_session() as session:
            stored_count = session.query(Bookmark).count()
            print(f"\nResults:")
            print(f"Attempted to process: {len(results)}")
            print(f"Successfully stored: {stored_count}")
            
            # Show first few errors if any
            errors = [r for r in results if 'error' in r]
            if errors:
                print("\nFirst 3 errors:")
                for i, error in enumerate(errors[:3]):
                    print(f"{i+1}. {error['error']}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()