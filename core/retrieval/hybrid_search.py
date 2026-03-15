from core.retrieval.vector_search import VectorDB
from core.models.llm import LLMEngine
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
            
            # THE FIX: Clean punctuation from database text as well!
            clean_caption = re.sub(r'[^\w\s]', '', caption)
            clean_text = re.sub(r'[^\w\s]', '', text)
            combined_clean = clean_caption + " " + clean_text
            
            score = 0
            
            distance = res.get('_distance', 1.0)
            base_vector_score = max(0, (1.0 - distance) * 500) 
            score += base_vector_score
            
            keyword_bonus = res.get('score', 0) * 20
            score += keyword_bonus
            
            if meta.get('type') == "image_index":
                # BOOSTER: Give massive points if the caption perfectly matches
                if clean_query in clean_caption:
                    score += 3000
                elif clean_query in combined_clean:
                    score += 2000 
                
                match_count = sum(1 for t in query_terms if t in combined_clean)
                score += (match_count * 100) 
            
            elif meta.get('type') == "text":
                if clean_query in clean_text:
                    score += 1500 
                
                match_count = sum(1 for t in query_terms if t in clean_text)
                score += (match_count * 50)

            reranked.append((score, res))
        
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in reranked]

    def query_with_feedback(self, query):
        yield ("status", "🔍 Searching Database...")
        
        search_queries = [query]
        
        all_results = []
        for q in search_queries:
            vec_res = self.db.search(q, top_k=25) 
            key_res = self.db.search_keyword(q, top_k=20)
            all_results.extend(vec_res + key_res)

        seen = set()
        unique_results = []
        for r in all_results:
            if r['text'] not in seen:
                unique_results.append(r); seen.add(r['text'])
        
        results = self._smart_rerank(unique_results, query)[:15]

        if not results:
            yield ("status", "❌ Not found.")
            yield ("text", "I'm sorry, but NO CONTEXT FOUND !!")
            return

        # --- IMAGE SCORING AND SORTING ---
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        # Added "whats" to stop words (without punctuation)
        stop_words = {"the", "cell", "what", "how", "why", "explain", "describe", "define", "whats", "are", "is", "a", "an", "diagram", "show", "me", "of", "by", "in", "to", "and", "or", "through"}
        query_terms = list(set([w for w in clean_query.split() if len(w) > 2 and w not in stop_words]))
        
        image_candidates = []
        
        for res in results:
            meta = res['metadata']
            
            if meta.get('type') == 'image_index':
                cap = meta.get('image_caption', 'None').lower()
                rich_text = res.get('text', '').lower().replace('|', ' ')
                
                # Clean punctuation for accurate matching
                clean_cap = re.sub(r'[^\w\s]', '', cap)
                clean_rich = re.sub(r'[^\w\s]', '', rich_text)
                combined_clean = clean_cap + " " + clean_rich
                
                rule_a_exact = (clean_query in combined_clean)
                rule_a_super_exact = (clean_query in clean_cap) # Is it exactly in the caption?
                
                match_count = sum(1 for k in query_terms if k in combined_clean)
                required_matches = max(2, int(len(query_terms) * 0.5))
                rule_b_keywords = match_count >= required_matches if query_terms else False
                
                if rule_a_exact or rule_b_keywords:
                    img_score = match_count
                    if rule_a_exact:
                        img_score += 1000 
                    if rule_a_super_exact:
                        img_score += 5000 # Unbeatable score for exact caption match!
                        
                    target_asset = {
                        "source": meta.get('source'), "page": meta.get('page'),
                        "rect": meta.get('image_rect'), "xref": meta.get('image_xref', 0),
                        "caption": meta.get('image_caption'), "image_path": meta.get('image_path', 'None'),
                        "score": img_score 
                    }
                    image_candidates.append(target_asset)

        image_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        for asset in image_candidates[:2]:
            yield ("spatial_image", asset)

        yield ("status", "⚡ Synthesizing Answer...")
        
        results.sort(key=lambda x: x['metadata'].get('page', 0))
        context = "\n".join([f"[Page {r['metadata']['page']}]: {r['text']}" for r in results])
        
        system_prompt = f"""You are a helpful assistant. Use ONLY the provided context. 
        If no context found then dont give anything any say NO CONTEXT FOUND !!
        Cite page numbers when providing information.
        
        CONTEXT: {context}"""
        
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': query}]
        
        for chunk in self.llm.chat_stream(messages): 
            yield ("text", chunk)
            
        yield ("chunks", results)