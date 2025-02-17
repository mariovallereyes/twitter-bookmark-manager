import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from core.ai_categorization import BookmarkCategorizer
from database.db import get_db_session
from database.models import Bookmark, Category, bookmark_categories
from config.constants import BOOKMARK_CATEGORIES
import logging
from typing import List, Dict, Tuple
from sqlalchemy import func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoryProcessor:
    def __init__(self):
        """Initialize with categorizer and categories"""
        self.categorizer = BookmarkCategorizer()
        
        # Separate main and media categories
        self.main_categories = [
            cat for cat in BOOKMARK_CATEGORIES 
            if cat['name'] not in ['Videos', 'Images']
        ]
        self.media_categories = [
            cat for cat in BOOKMARK_CATEGORIES 
            if cat['name'] in ['Videos', 'Images']
        ]
        
        # Use examples from BOOKMARK_CATEGORIES for better classification
        self.category_examples = {
            cat['name']: cat['examples'] for cat in BOOKMARK_CATEGORIES
        }
        
    def get_category(self, text: str, session) -> Tuple[Category, float]:
        """Get or create category using zero-shot classification"""
        try:
            if not text or not text.strip():
                # Get default category for empty texts
                default_cat = session.query(Category).filter_by(name='Tech News & Trends').first()
                if not default_cat:
                    default_cat = Category(name='Tech News & Trends')
                    session.add(default_cat)
                return default_cat, 1.0
                
            # Get prediction
            result = self.categorizer.classifier(
                text,
                candidate_labels=[cat['name'] for cat in self.main_categories],
                multi_label=False
            )
            
            # Get or create category
            cat_name = result['labels'][0]
            category = session.query(Category).filter_by(name=cat_name).first()
            if not category:
                category = Category(name=cat_name)
                session.add(category)
                
            return category, result['scores'][0]
            
        except Exception as e:
            logger.error(f"Error in category classification: {e}")
            # Fallback to Tech News
            default_cat = session.query(Category).filter_by(name='Tech News & Trends').first()
            if not default_cat:
                default_cat = Category(name='Tech News & Trends')
                session.add(default_cat)
            return default_cat, 1.0

    def detect_media_category(self, bookmark: Bookmark, session) -> Category or None:
        """Detect if bookmark has media and return appropriate category"""
        if not bookmark.media_files:
            return None
            
        media_type = None
        for media in bookmark.media_files:
            type_str = media.get('type', '').lower()
            if 'video' in type_str:
                media_type = 'Videos'
                break
            elif 'photo' in type_str or 'image' in type_str:
                media_type = 'Images'
                break
                
        if media_type:
            category = session.query(Category).filter_by(name=media_type).first()
            if not category:
                category = Category(name=media_type)
                session.add(category)
            return category
        return None

    def process_bookmark(self, bookmark: Bookmark) -> Dict[str, any]:
        """Process a single bookmark and assign categories"""
        try:
            if not bookmark.text:
                logger.warning(f"Empty text for bookmark {bookmark.id}")
                return {
                    'bookmark_id': bookmark.id,
                    'success': False,
                    'error': 'Empty text'
                }

            # Get primary category
            primary_cat, confidence = self.get_category(bookmark.text, get_db_session())
            
            # Check for media category
            media_cat = self.detect_media_category(bookmark, get_db_session())
            
            # Save categories
            with get_db_session() as session:
                # Clear existing categories
                session.execute(
                    bookmark_categories.delete().where(
                        bookmark_categories.c.bookmark_id == bookmark.id
                    )
                )
                
                # Add primary category
                bookmark.categories.append(primary_cat)
                
                # Add media category if exists
                if media_cat:
                    bookmark.categories.append(media_cat)
                
                session.commit()
                
                logger.info(f"Categorized bookmark {bookmark.id}: {primary_cat.name}" + 
                          (f" + {media_cat.name}" if media_cat else ""))
                
                return {
                    'bookmark_id': bookmark.id,
                    'success': True,
                    'primary_category': primary_cat.name,
                    'media_category': media_cat.name if media_cat else None
                }
                
        except Exception as e:
            logger.error(f"Error processing bookmark {bookmark.id}: {e}")
            return {
                'bookmark_id': bookmark.id,
                'success': False,
                'error': str(e)
            }

    def process_all_bookmarks(self, batch_size: int = 50) -> List[Dict[str, any]]:
        """Process only uncategorized bookmarks in batches"""
        results = []
        processed = 0
        
        with get_db_session() as session:
            # Query only bookmarks that have no categories
            uncategorized = session.query(Bookmark)\
                .outerjoin(bookmark_categories)\
                .filter(bookmark_categories.c.category_id.is_(None))\
                .all()
                
            total = len(uncategorized)
            logger.info(f"Found {total} uncategorized bookmarks to process")
            
            if total == 0:
                logger.info("No uncategorized bookmarks found!")
                return results
            
            # First, ensure all categories exist
            for cat in BOOKMARK_CATEGORIES:
                if not session.query(Category).filter_by(name=cat['name']).first():
                    session.add(Category(name=cat['name']))
            session.commit()
            
            # Process only uncategorized bookmarks
            for bookmark in uncategorized:
                try:
                    # Get primary category
                    primary_cat, confidence = self.get_category(bookmark.text, session)
                    
                    # Get media category if applicable
                    media_cat = self.detect_media_category(bookmark, session)
                    
                    # Add categories
                    bookmark.categories.append(primary_cat)
                    if media_cat:
                        bookmark.categories.append(media_cat)
                    
                    results.append({
                        'bookmark_id': bookmark.id,
                        'success': True,
                        'primary_category': primary_cat.name,
                        'media_category': media_cat.name if media_cat else None
                    })
                    
                    processed += 1
                    if processed % 10 == 0:
                        session.commit()
                        logger.info(f"Processed {processed}/{total} bookmarks")
                        
                except Exception as e:
                    logger.error(f"Error processing bookmark {bookmark.id}: {e}")
                    results.append({
                        'bookmark_id': bookmark.id,
                        'success': False,
                        'error': str(e)
                    })
            
            # Final commit
            session.commit()
            
            # Print summary
            success_count = len([r for r in results if r['success']])
            logger.info(f"Completed! Successfully categorized: {success_count}/{len(results)}")
            
            return results

def main():
    """Main function to run the categorization process"""
    try:
        processor = CategoryProcessor()
        results = processor.process_all_bookmarks()
        
        # Print final summary
        success = len([r for r in results if r['success']])
        total = len(results)
        print(f"\n✅ Processing complete!")
        print(f"Successfully categorized: {success}/{total} bookmarks")
        print(f"Failed: {total - success}")
        
        # Print category distribution
        with get_db_session() as session:
            cats = session.query(Category.name, func.count(bookmark_categories.c.bookmark_id))\
                .join(bookmark_categories)\
                .group_by(Category.name)\
                .all()
            print("\nCategory Distribution:")
            for cat, count in cats:
                print(f"{cat}: {count} bookmarks")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()