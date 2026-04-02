import lancedb
import os
import time
from core.config import DB_DIR
from core.models.embedder import Embedder

import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter

class VectorDB:
    def __init__(self):
        print(f"--- DEBUG: Initializing VectorDB at {DB_DIR} ---")
        os.makedirs(DB_DIR, exist_ok=True)
        self.embedder = Embedder()
        self._connect()

    def _connect(self):
        """Establishes connection WITHOUT auto-wiping data."""
        self.db = lancedb.connect(str(DB_DIR))
        
        if "vectors" in self.db.table_names():
            table = self.db.open_table("vectors")
            print(f"--- DEBUG: Table found. Schema fields: {table.schema.names} ---")
            
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

    # ==========================================
    # 📄 PDF INGESTION (Markdown Preserved)
    # ==========================================
    def ingest_pdf_markdown(self, pdf_path):
        print(f"📄 Processing PDF {pdf_path} with Markdown Extractor...")
        try:
            md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=150)
            
            final_chunks = []
            for page_data in md_pages:
                page_text = page_data.get('text', '')
                page_num = page_data.get('metadata', {}).get('page', 0) + 1 
                text_chunks = splitter.create_documents([page_text])
                
                for chunk in text_chunks:
                    final_chunks.append({
                        "text": chunk.page_content,
                        "metadata": {"source": os.path.basename(pdf_path), "page": page_num, "type": "text"}
                    })
            
            self.add_chunks(final_chunks)
            print("✅ PDF Ingested Successfully!")
            return True
        except Exception as e:
            print(f"❌ Error ingesting PDF: {e}")
            return False

    def add_chunks(self, chunks):
        try:
            if not chunks: return
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            vectors = self.embedder.embed_batch(texts)
            
            data = []
            for i, vector in enumerate(vectors):
                meta = metadatas[i]
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

    def search(self, query, top_k=25):
        try:
            self.table = self.db.open_table("vectors")
            print(f"--- DEBUG: Semantic Search for '{query}' ---")
            query_vector = self.embedder.embed(query)
            if query_vector is None: return []
            
            results = self.table.search(query_vector).metric("cosine").limit(top_k + 20).to_list()
            clean_results = [r for r in results if r['text'] != 'init'][:top_k]
            print(f"--- DEBUG: Semantic Search found {len(clean_results)} results ---")
            return clean_results
        except Exception as e:
            print(f"⚠️ Search Error: {e}")
            return []

    def search_keyword(self, query, top_k=15):
        try:
            import re
            self.table = self.db.open_table("vectors")
            
            # 🔥 THE FIX: Ignore conversational filler words
            ignore_words = {"what", "is", "are", "the", "how", "why", "who", "when", "where", "can", "you", "me", "and", "for", "with", "about", "explain", "summarize", "describe", "details", "of", "in", "to", "a", "an"}
            clean_query = re.sub(r'[^\w\s\.]', '', query.lower())
            
            keywords = [w for w in clean_query.split() if len(w) > 2 and w not in ignore_words]
            if not keywords: return []
            
            print(f"--- DEBUG: Keyword Search for {keywords} ---")
            
            try:
                all_rows = self.table.search().limit(10000).to_list()
            except Exception:
                return []

            matches = []
            for row in all_rows:
                if row['text'] == 'init': continue
                
                t = str(row['text']).lower()
                c = str(row['metadata'].get('image_caption', '')).lower()
                s = str(row['metadata'].get('source', '')).lower()
                
                score = 0
                for k in keywords:
                    if k in t: score += 2
                    if k in c: score += 10
                    if k in s: score += 100 
                
                if score > 0:
                    row['score'] = score
                    matches.append(row)

            matches.sort(key=lambda x: x['score'], reverse=True)
            print(f"--- DEBUG: Keyword Search found {len(matches)} matches ---")
            return matches[:top_k]
            
        except Exception as e:
            print(f"⚠️ Keyword Search Error: {e}")
            return []