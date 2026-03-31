import ollama
from core.config import EMBEDDING_MODEL

class Embedder:
    def __init__(self):
        # Force the model to be the one we just pulled
        self.model = "nomic-embed-text"

    def embed(self, text):
        """
        Embeds a single string.
        """
        try:
            # Clean text to prevent dimension errors
            cleaned_text = text.replace("\n", " ").strip()
            if not cleaned_text:
                return None
                
            response = ollama.embeddings(
                model=self.model, 
                prompt=cleaned_text
            )
            return response['embedding']
        except Exception as e:
            print(f"❌ EMBEDDING ERROR: {e}")
            return None

    def embed_batch(self, texts):
        """
        Embeds a list of strings (Used by VectorDB ingestion).
        """
        batch_results = []
        for text in texts:
            vector = self.embed(text)
            if vector:
                batch_results.append(vector)
            # If embedding fails (None), we skip adding it to the list
            # to prevent crashing the DB with null vectors.
        return batch_results