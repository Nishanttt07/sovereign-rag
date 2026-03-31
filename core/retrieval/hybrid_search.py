from core.retrieval.vector_search import VectorDB
from core.models.llm import LLMEngine
from core.config import SYSTEM_PROMPT, STOP_WORDS, SEARCH_THRESHOLD, SQL_TRIGGERS
from core.retrieval.sql_agent import SQLAgent
import re

class RAGPipeline:
    def __init__(self):
        self.db = VectorDB()
        # 🔥 THE UPGRADE: We explicitly call Llama 3.2 (3B) for the final chat synthesis.
        # It fits perfectly in 4GB VRAM alongside Qwen and provides excellent, fast summaries.
        self.llm = LLMEngine(model_name="llama3.2")

    def _smart_rerank(self, results, query):
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        query_terms = list(set([t for t in clean_query.split() if len(t) > 2]))
        
        reranked = []
        for res in results:
            meta = res.get('metadata', {})
            score = 0
            distance = res.get('_distance', 1.0)
            score += max(0, (1.0 - distance) * 500)
            
            if meta.get('type') == "image_index":
                score += 2000
                
            reranked.append((score, res))
            
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in reranked]

    def _extract_images(self, results, query_lower):
        """Safely extracts relevant images and guarantees a list is returned."""
        clean_query = re.sub(r'[^\w\s]', '', query_lower)
        query_terms = list(set([w for w in clean_query.split() if len(w) > 2 and w not in STOP_WORDS]))
        image_candidates = []
        
        for res in results:
            meta = res.get('metadata', {})
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
        return image_candidates[:2]  # Always returns a list, even if empty

    def query_with_feedback(self, query):
        query_lower = query.lower()
        sql_context = ""
        
        # 🚦 THE ROUTER
        is_sql_query = any(trigger in query_lower for trigger in SQL_TRIGGERS) or "table" in query_lower or "rows" in query_lower

        # ==========================================
        # 1. SQL AGENT EXECUTION & DIRECT UI BYPASS
        # ==========================================
        if is_sql_query:
            yield ("status", "📊 Fetching database records (Qwen 2.5)...")
            agent = SQLAgent()
            sql_context = agent.query(query)
            
            if sql_context and "Error" not in sql_context and "empty" not in sql_context.lower():
                # 🔥 THE BYPASS: Yield the markdown table directly to the frontend!
                # This bypasses the LLM's tendency to refuse or hallucinate tables.
                yield ("text", f"### 🗄️ Retrieved Records:\n{sql_context}\n\n---\n**🤖 AI Analysis:**\n")

        # ==========================================
        # 2. VECTOR SEARCH (PDFs)
        # ==========================================
        # If SQL was successful, we drastically reduce the PDF chunks to save VRAM 
        # and prevent Llama 3.2 from getting distracted.
        top_k_val = 2 if sql_context else 15 
        
        yield ("status", "🔍 Searching documents...")
        all_results = self.db.search(query, top_k=top_k_val) + self.db.search_keyword(query, top_k=5)
        
        seen = set()
        unique_results = []
        for r in all_results:
            if r['text'] not in seen:
                unique_results.append(r)
                seen.add(r['text'])
                
        results = self._smart_rerank(unique_results, query)[:top_k_val]

        # Safely extract and yield spatial images
        for asset in self._extract_images(results, query_lower): 
            yield ("spatial_image", asset)

        # ==========================================
        # 3. LLM SYNTHESIS (Llama 3.2)
        # ==========================================
        yield ("status", "⚡ Synthesizing response (Llama 3.2)...")
        
        pdf_text = "\n".join([f"Page {r['metadata']['page']}: {r['text']}" for r in results])
        
        # A highly de-weaponized prompt. We don't say "SQL" or "Database" to avoid triggering
        # any internal safety refusals about external connections.
        final_system_prompt = f"""{SYSTEM_PROMPT}
        You are a helpful technical assistant analyzing text records.
        Do NOT mention that you are an AI. 
        Do NOT apologize. 
        Just directly answer the user's question based ONLY on the text below. 
        If the data is provided in a table format below, summarize the key finding in 1 or 2 short sentences.
        """

        context_to_send = f"RECORDS:\n{sql_context}\n\nMANUAL EXCERPTS:\n{pdf_text}"

        messages = [
            {'role': 'system', 'content': final_system_prompt},
            {'role': 'user', 'content': f"Context:\n{context_to_send}\n\nQuestion: {query}"}
        ]
        
        for chunk in self.llm.chat_stream(messages): 
            yield ("text", chunk)
            
        yield ("chunks", results)