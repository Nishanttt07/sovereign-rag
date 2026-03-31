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
        # Check if the user's prompt contains any YAML SQL triggers
        use_sql = any(trigger in query_lower for trigger in SQL_TRIGGERS)
        
        if use_sql:
            yield ("status", "📊 Analyzing Tabular Data via SQL Agent...")
            sql_agent = SQLAgent()
            sql_result = sql_agent.query(query)
            if sql_result and "[SQL Agent searched the database but found 0 results" not in sql_result:
                sql_context = f"\n[TABULAR DATABASE RESULTS]:\n{sql_result}\n"

        # ==========================================
        # 🔍 VECTOR SEARCH: UNSTRUCTURED DATA (PDFs)
        # ==========================================
        yield ("status", "🔍 Searching Document Database...")
        
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

        # Failsafe: If BOTH databases return absolutely nothing
        if not results and not sql_context:
            yield ("status", "❌ Not found.")
            yield ("text", "I'm sorry, but NO CONTEXT FOUND in Documents or Databases!!")
            return

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
                rich_text = res.get('text', '').lower().replace('|', ' ')
                
                clean_cap = re.sub(r'[^\w\s]', '', cap)
                combined_clean = clean_cap + " " + re.sub(r'[^\w\s]', '', rich_text)
                
                rule_a_exact = (clean_query in combined_clean)
                rule_a_super_exact = (clean_query in clean_cap) 
                
                match_count = sum(1 for k in query_terms if k in combined_clean)
                required_matches = max(2, int(len(query_terms) * float(SEARCH_THRESHOLD)))
                rule_b_keywords = match_count >= required_matches if query_terms else False
                
                if rule_a_exact or rule_b_keywords:
                    img_score = match_count
                    if rule_a_exact: img_score += 1000 
                    if rule_a_super_exact: img_score += 5000 
                        
                    image_candidates.append({
                        "source": meta.get('source'), "page": meta.get('page'),
                        "rect": meta.get('image_rect'), "xref": meta.get('image_xref', 0),
                        "caption": meta.get('image_caption'), "image_path": meta.get('image_path', 'None'),
                        "score": img_score 
                    })

        image_candidates.sort(key=lambda x: x['score'], reverse=True)
        for asset in image_candidates[:2]:
            yield ("spatial_image", asset)

        # ==========================================
        # 🧠 HYBRID SYNTHESIS (LLM)
        # ==========================================
        yield ("status", "⚡ Synthesizing Multi-Modal Answer...")
        
        # Combine PDF Text
        pdf_context = ""
        if results:
            results.sort(key=lambda x: x['metadata'].get('page', 0))
            pdf_context = "\n[DOCUMENT RESULTS]:\n" + "\n".join([f"[Page {r['metadata']['page']}]: {r['text']}" for r in results])
        
        # Merge SQL Data and PDF Data
        master_context = sql_context + pdf_context
        
        final_prompt = f"""{SYSTEM_PROMPT}
        
        Use ONLY the provided context below to answer the user. 
        If the context does not contain the answer, say "NO CONTEXT FOUND !!".
        
        CONTEXT: 
        {master_context}
        """
        
        messages = [{'role': 'system', 'content': final_prompt}, {'role': 'user', 'content': query}]
        
        for chunk in self.llm.chat_stream(messages): 
            yield ("text", chunk)
            
        yield ("chunks", results)