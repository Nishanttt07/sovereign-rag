from core.retrieval.hybrid_search import RAGPipeline
import sys

def main():
    rag = RAGPipeline()
    q = "what is in Rutuja.pptx"
    print(f"Testing query: {q}")
    for event, payload in rag.query_with_feedback(q):
        if event == "status":
            print(f"Status: {payload}")
        elif event == "text":
            print(f"Text: {payload}")
        elif event == "chunks":
            print(f"Found {len(payload)} chunks.")

if __name__ == "__main__":
    main()
