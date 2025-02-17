from core.search import BookmarkSearch
import logging
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_tweet(result: Dict[str, Any], is_search_result: bool = True) -> str:
    """Format a tweet result for display"""
    output = []
    
    # Score only for search results
    if is_search_result and 'score' in result:
        output.append(f"Score: {result['score']:.2f}")
    
    # Author
    output.append(f"Author: {result['author']}")
    
    # Categories
    if result.get('categories'):
        output.append(f"Categories: {', '.join(result['categories'])}")
    
    # Created date
    output.append(f"Created: {result['created_at'].strftime('%Y-%m-%d %H:%M')}")
    
    # Tweet text
    if result.get('text'):
        output.append(f"\nTweet: {result['text']}")
    
    return "\n".join(output)

def get_selected_categories(search: BookmarkSearch) -> List[str]:
    """Get multiple category selections from user"""
    categories = search.get_categories()
    if not categories:
        print("\n❌ No categories available")
        return []
        
    print("\n📑 Available categories:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")
    
    selected = []
    while True:
        choice = input("\nEnter category numbers (comma-separated) or 'done': ").strip().lower()
        if choice == 'done':
            break
            
        try:
            # Parse comma-separated numbers
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = [categories[i] for i in indices if 0 <= i < len(categories)]
            
            # Show selected categories
            print("\nSelected categories:")
            for cat in selected:
                print(f"- {cat}")
            
            break
            
        except (ValueError, IndexError):
            print("❌ Invalid selection. Please try again.")
    
    return selected

def main():
    try:
        # Initialize search
        search = BookmarkSearch()
        print("\n✓ Search engine initialized")
        
        while True:
            print("\n🔍 Choose an option:")
            print("1. Search bookmarks")
            print("2. Search by category")
            print("3. Search by multiple categories")
            print("4. List recent bookmarks")
            print("5. Show categories")
            print("6. Exit")
            
            choice = input("\nEnter choice (1-6): ")
            
            if choice == "1":
                query = input("\nEnter search query: ")
                limit = int(input("Number of results (default 5): ") or "5")
                
                print(f"\nSearching for '{query}'...")
                results = search.search(query, limit=limit)
                
                if results:
                    print(f"\n📚 Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {format_tweet(result)}")
                else:
                    print("\n❌ No results found")
                    
            elif choice == "2":
                # Show available categories first
                categories = search.get_categories()
                if not categories:
                    print("\n❌ No categories available")
                    continue
                    
                print("\n📑 Available categories:")
                for i, cat in enumerate(categories, 1):
                    print(f"{i}. {cat}")
                
                # Get category choice
                cat_choice = input("\nEnter category number: ")
                try:
                    category = categories[int(cat_choice) - 1]
                except (ValueError, IndexError):
                    print("\n❌ Invalid category choice")
                    continue
                
                # Get search query
                query = input("\nEnter search query (or press Enter to see all in category): ")
                limit = int(input("Number of results (default 5): ") or "5")
                
                print(f"\nSearching in category '{category}'...")
                results = search.search(query, limit=limit, category=category)
                
                if results:
                    print(f"\n📚 Found {len(results)} results in {category}:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {format_tweet(result)}")
                else:
                    print(f"\n❌ No results found in category {category}")
                    
            elif choice == "3":
                # Get multiple category selections
                selected_categories = get_selected_categories(search)
                if not selected_categories:
                    continue
                
                # Get search query
                query = input("\nEnter search query (or press Enter to see all in categories): ")
                limit = int(input("Number of results (default 5): ") or "5")
                
                print(f"\nSearching in categories: {', '.join(selected_categories)}...")
                results = search.search(query, limit=limit, categories=selected_categories)
                
                if results:
                    print(f"\n📚 Found {len(results)} results matching ANY of the selected categories:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {format_tweet(result)}")
                else:
                    print("\n❌ No results found matching ANY of the selected categories")
                    
            elif choice == "4":
                limit = int(input("\nNumber of bookmarks to show (default 10): ") or "10")
                bookmarks = search.get_all_bookmarks(limit=limit)
                
                if bookmarks:
                    print(f"\n📚 Showing {len(bookmarks)} recent bookmarks:")
                    for i, b in enumerate(bookmarks, 1):
                        print(f"\n{i}. {format_tweet(b, is_search_result=False)}")
                else:
                    print("\n❌ No bookmarks found")
                    
            elif choice == "5":
                categories = search.get_categories()
                if categories:
                    print("\n📑 Available categories:")
                    for i, cat in enumerate(categories, 1):
                        print(f"{i}. {cat}")
                else:
                    print("\n❌ No categories found")
                    
            elif choice == "6":
                print("\nGoodbye! 👋")
                break
                
            else:
                print("\n❌ Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.exception("Error in main")

if __name__ == "__main__":
    main()