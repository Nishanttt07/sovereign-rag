from core.retrieval.vector_search import VectorDB
from core.models.llm import LLMEngine
from core.config import SYSTEM_PROMPT, STOP_WORDS, SEARCH_THRESHOLD, SQL_TRIGGERS
from core.retrieval.sql_agent import SQLAgent
import re

class RAGPipeline:
    def __init__(self):
        self.db = VectorDB()
        self.llm = LLMEngine()

    def _smart_rerank(self, results, query):
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        query_terms = list(set([t for t in clean_query.split() if len(t) > 2]))
        
        reranked = []
        for res in results:
            meta = res.get('metadata', {})
            text = res.get('text', '').lower().replace('|', ' ') 
            caption = meta.get('image_caption', 'None').lower()
            
            clean_caption = re.sub(r'[^\w\s]', '', caption)
            clean_text = re.sub(r'[^\w\s]', '', text)
            combined_clean = clean_caption + " " + clean_text
            
            score = 0
            distance = res.get('_distance', 1.0)
            score += max(0, (1.0 - distance) * 500) 
            score += res.get('score', 0) * 20
            
            if meta.get('type') == "image_index":
                if clean_query in clean_caption: score += 3000
                elif clean_query in combined_clean: score += 2000 
                match_count = sum(1 for t in query_terms if t in combined_clean)
                score += (match_count * 100) 
            
            elif meta.get('type') == "text":
                if clean_query in clean_text: score += 1500 
                match_count = sum(1 for t in query_terms if t in clean_text)
                score += (match_count * 50)

            reranked.append((score, res))
        
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in reranked]

    def query_with_feedback(self, query):
        query_lower = query.lower()
        sql_context = ""
        
        # ==========================================
        # 🚦 THE ROUTER: TABULAR DATA CHECK
        # ==========================================
        # Check if query hits SQL keywords or references a table
        use_sql = any(trigger in query_lower for trigger in SQL_TRIGGERS) or "table" in query_lower
        
        if use_sql:
            yield ("status", "📊 Querying SQL Database Agent...")
            sql_agent = SQLAgent()
            sql_result = sql_agent.query(query)
            
            # If the SQL Agent found something, we wrap it in a clear tag
            if sql_result and "Error" not in sql_result:
                sql_context = f"\n[TABULAR DATABASE RESULTS]:\n{sql_result}\n"
            else:
                sql_context = "\n[SQL NOTIFICATION]: SQL Agent searched but found no matching records.\n"

        # ==========================================
        # 🔍 VECTOR SEARCH: UNSTRUCTURED DATA (PDFs)
        # ==========================================
        yield ("status", "🔍 Searching Document Vectors...")
        
        all_results = []
        vec_res = self.db.search(query, top_k=25) 
        key_res = self.db.search_keyword(query, top_k=20)
        all_results.extend(vec_res + key_res)

        seen = set()
        unique_results = []
        for r in all_results:
            if r['text'] not in seen:
                unique_results.append(r); seen.add(r['text'])
        
        results = self._smart_rerank(unique_results, query)[:15]

        # ==========================================
        # 🖼️ SPATIAL ASSET EXTRACTION
        # ==========================================
        clean_query = re.sub(r'[^\w\s]', '', query_lower)
        query_terms = list(set([w for w in clean_query.split() if len(w) > 2 and w not in STOP_WORDS]))
        image_candidates = []
        
        for res in results:
            meta = res['metadata']
            if meta.get('type') == 'image_index':
                cap = meta.get('image_caption', 'None').lower()
                clean_cap = re.sub(r'[^\w\s]', '', cap)
                match_count = sum(1 for k in query_terms if k in clean_cap)
                
                if (clean_query in clean_cap) or (match_count >= 2):
                    image_candidates.append({
                        "source": meta.get('source'), "page": meta.get('page'),
                        "rect": meta.get('image_rect'), "xref": meta.get('image_xref', 0),
                        "caption": meta.get('image_caption'), "image_path": meta.get('image_path', 'None'),
                        "score": match_count 
                    })

        image_candidates.sort(key=lambda x: x['score'], reverse=True)
        for asset in image_candidates[:2]:
            yield ("spatial_image", asset)

        # ==========================================
        # 🧠 HYBRID SYNTHESIS (LLM) - STRENGTHENED
        # ==========================================
        yield ("status", "⚡ Synthesizing Multi-Modal Answer...")
        
        pdf_context = ""
        if results:
            results.sort(key=lambda x: x['metadata'].get('page', 0))
            pdf_context = "\n[DOCUMENT EXCERPTS FROM PDF]:\n" + "\n".join([f"[Page {r['metadata']['page']}]: {r['text']}" for r in results])
        
        # Merge SQL Data and PDF Data
        master_context = f"{sql_context}\n{pdf_context}"
        
        # This prompt is designed to stop the "I am an AI and can't see databases" lie.
        final_prompt = f"""{SYSTEM_PROMPT}
        
        ### DATA SOURCE INSTRUCTIONS:
        - Below is CONTEXT extracted from a SQL Database and a PDF Vector Store.
        - If '[TABULAR DATABASE RESULTS]' is present, it contains real data from the user's spreadsheets. Use it!
        - Do NOT claim you cannot access databases. The data is already provided in this prompt.
        - Cite the source (SQL Database or PDF Page Number) for every fact you provide.
        - If both sources are empty, say "NO CONTEXT FOUND !!".
        
        ### CONTEXT DATA:
        {master_context}
        """
        
        messages = [{'role': 'system', 'content': final_prompt}, {'role': 'user', 'content': query}]
        
        for chunk in self.llm.chat_stream(messages): 
            yield ("text", chunk)
            
        yield ("chunks", results)