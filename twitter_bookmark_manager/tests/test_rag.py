from core.rag import BookmarkRAG
import logging
import json

logging.disable(logging.CRITICAL)

def main():
    try:
        rag = BookmarkRAG()
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            print("\nProcessing your query...")
            result = rag.chat(query)

            # 🔹 Print the raw result for debugging
            print("\n🔹 FULL RESULT OBJECT:", json.dumps(result, indent=2))
            print("🔹 TYPE OF result:", type(result))

            # 🔹 Ensure 'response' exists
            if 'response' not in result:
                print("\n❌ ERROR: 'response' key is missing in result!")
                return

            # ✅ Safe to print response now
            print(f"\n✅ Response: {result['response'].get('text', 'ERROR: No text found')}")

            # ✅ Print sources safely
            print("\n📌 Sources:")
            for url in result.get('sources', []):
                print(f"- {url}")

    except Exception as e:
        print(f"❌ ERROR in test_rag.py: {str(e)}")

if __name__ == "__main__":
    main()
