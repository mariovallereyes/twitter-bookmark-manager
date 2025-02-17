import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chat import BookmarkChat
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def interactive_chat():
    try:
        logger.info("Initializing BookmarkChat...")
        chat = BookmarkChat()
        logger.info("Chat initialized! Type your questions (or 'quit' to exit)")
        
        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
                
            if user_input:
                # Get response
                response = chat.chat(user_input)
                
                # Print response
                print("\n🤖 Assistant:")
                print(response['response'])
                print(f"\nBookmarks used: {response['bookmarks_used']}")
                
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    print("\n🤖 Welcome to Bilbeny's Bookmarks Chat!\n")
    interactive_chat()