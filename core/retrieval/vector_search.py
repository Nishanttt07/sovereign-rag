import lancedb
import os
import time
from core.config import DB_DIR
from core.models.embedder import Embedder

class VectorDB:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.embedder = Embedder()
        self._connect()

    def _connect(self):
        """Establishes connection to the DB."""
        self.db = lancedb.connect(str(DB_DIR))
        
        # Initialize Table if missing
        if "vectors" not in self.db.table_names():
             self._create_empty_table()
        
        try:
            self.table = self.db.open_table("vectors")
        except Exception:
            self._create_empty_table()
            self.table = self.db.open_table("vectors")

    def _create_empty_table(self):
        data = [{
            "vector": [0.0] * 768, 
            "text": "init", 
            "metadata": {
                "source": "init", "page": 0, 
                "image_path": "None", "image_caption": "None", "type": "text"
            }
        }]
        self.db.create_table("vectors", data=data, mode="overwrite")

    def add_chunks(self, chunks):
        try:
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            vectors = self.embedder.embed_batch(texts)
            
            data = []
            for i, vector in enumerate(vectors):
                meta = metadatas[i]
                if "image_path" not in meta: meta["image_path"] = "None"
                if "image_caption" not in meta: meta["image_caption"] = "None"
                if "type" not in meta: meta["type"] = "text"
                
                data.append({"vector": vector, "text": texts[i], "metadata": meta})
            
            if data:
                self.table.add(data)
        except Exception as e:
            print(f"⚠️ Add Error: {e}")
            # Try to heal connection
            self._connect()

    def search(self, query, top_k=25):
        """Search with Auto-Retry for Stale Connections"""
        for attempt in range(2): # Try twice
            try:
                query_vector = self.embedder.embed(query)
                if query_vector is None: return []
                
                results = self.table.search(query_vector) \
                    .metric("cosine") \
                    .limit(top_k + 5) \
                    .to_list()
                
                return [r for r in results if r['text'] != 'init'][:top_k]
            
            except Exception as e:
                print(f"⚠️ Search Error (Attempt {attempt+1}): {e}")
                if "Not found" in str(e) or "IO" in str(e):
                    print("♻️ healing database connection...")
                    self._connect()
                    time.sleep(0.5)
                else:
                    return []
        return []

    def search_keyword(self, query, top_k=10):
        """Keyword Search with Auto-Retry"""
        for attempt in range(2):
            try:
                keywords = [w.lower() for w in query.split() if len(w) > 2]
                if not keywords: return []

                all_data = self.table.to_pandas()
                
                matches = []
                for _, row in all_data.iterrows():
                    if row['text'] == 'init': continue

                    text = row['text'].lower()
                    caption = row['metadata'].get('image_caption', 'None').lower()
                    
                    score = sum(1 for k in keywords if k in text or k in caption)
                    if score > 0:
                        matches.append({
                            "text": row['text'],
                            "metadata": row['metadata'],
                            "score": score,
                            "_distance": 0.0
                        })
                
                matches.sort(key=lambda x: x['score'], reverse=True)
                return matches[:top_k]
                
            except Exception as e:
                print(f"⚠️ Keyword Error (Attempt {attempt+1}): {e}")
                self._connect()
        return []