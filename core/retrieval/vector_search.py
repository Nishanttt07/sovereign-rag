import lancedb
import os
import time
from core.config import DB_DIR
from core.models.embedder import Embedder

class VectorDB:
    def __init__(self):
        print(f"--- DEBUG: Initializing VectorDB at {DB_DIR} ---")
        os.makedirs(DB_DIR, exist_ok=True)
        self.embedder = Embedder()
        self._connect()

    def _connect(self):
        """Establishes connection WITHOUT auto-wiping data."""
        self.db = lancedb.connect(str(DB_DIR))
        
        # Check if table exists
        if "vectors" in self.db.table_names():
            table = self.db.open_table("vectors")
            print(f"--- DEBUG: Table found. Schema fields: {table.schema.names} ---")
            
            # DISABLE AUTO-WIPE: We only print a warning now.
            if "image_rect" not in table.schema.names:
                print("⚠️ WARNING: Potential schema mismatch detected, but KEEPING DATA.")
        else:
            print("--- DEBUG: No table found. Creating new one. ---")
            self._create_empty_table()
            
        self.table = self.db.open_table("vectors")
        print(f"--- DEBUG: Connected to table 'vectors'. Rows: {self.table.count_rows()} ---")

    def _create_empty_table(self):
        """Creates the table with the complete schema."""
        data = [{
            "vector": [0.0] * 768, 
            "text": "init", 
            "metadata": {
                "source": "init", 
                "page": 0, 
                "type": "init", 
                "image_caption": "None", 
                "image_rect": [0.0, 0.0, 0.0, 0.0], 
                "image_xref": 0, 
                "image_path": "None"
            }
        }]
        self.db.create_table("vectors", data=data, mode="overwrite")
        print("✅ Database initialized with Complete Schema.")

    def add_chunks(self, chunks):
        try:
            if not chunks: return
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            vectors = self.embedder.embed_batch(texts)
            
            data = []
            for i, vector in enumerate(vectors):
                meta = metadatas[i]
                # Enforce schema consistency
                meta.setdefault("image_rect", [0.0, 0.0, 0.0, 0.0])
                meta.setdefault("image_xref", 0)
                meta.setdefault("image_caption", "None")
                meta.setdefault("image_path", "None")
                meta.setdefault("type", "text")
                data.append({"vector": vector, "text": texts[i], "metadata": meta})
            
            if data:
                self.table.add(data)
                print(f"✅ DB committed {len(data)} chunks.")
        except Exception as e:
            print(f"⚠️ Add Error: {e}")
            # Do NOT call _connect() recursively to avoid loops

    def search(self, query, top_k=25):
        try:
            print(f"--- DEBUG: Semantic Search for '{query}' ---")
            query_vector = self.embedder.embed(query)
            if query_vector is None: return []
            
            # Fetch more results to ensure fallback candidates exist
            results = self.table.search(query_vector).metric("cosine").limit(top_k + 20).to_list()
            clean_results = [r for r in results if r['text'] != 'init'][:top_k]
            print(f"--- DEBUG: Semantic Search found {len(clean_results)} results ---")
            return clean_results
        except Exception as e:
            print(f"⚠️ Search Error: {e}")
            return []

    def search_keyword(self, query, top_k=15):
        try:
            keywords = [w.lower() for w in query.split() if len(w) > 2]
            if not keywords: return []
            
            print(f"--- DEBUG: Keyword Search for {keywords} ---")
            
            # Fetch all rows safely
            try:
                all_rows = self.table.search().limit(10000).to_list()
            except Exception:
                return []

            matches = []
            for row in all_rows:
                if row['text'] == 'init': continue
                
                t = str(row['text']).lower()
                c = str(row['metadata'].get('image_caption', '')).lower()
                
                score = 0
                for k in keywords:
                    if k in t: score += 2
                    if k in c: score += 10
                
                if score > 0:
                    row['score'] = score
                    matches.append(row)

            matches.sort(key=lambda x: x['score'], reverse=True)
            print(f"--- DEBUG: Keyword Search found {len(matches)} matches ---")
            return matches[:top_k]
            
        except Exception as e:
            print(f"⚠️ Keyword Search Error: {e}")
            return []