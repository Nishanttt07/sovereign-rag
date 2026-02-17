from core.retrieval.vector_search import VectorDB
from core.models.llm import LLMEngine
from core.models.vision import VisionEngine

class RAGPipeline:
    def __init__(self):
        self.db = VectorDB()
        self.llm = LLMEngine()
        self.vision = VisionEngine() 

    def _smart_rerank(self, results, query):
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        query_lower = query.lower()
        
        reranked = []
        for res in results:
            meta = res.get('metadata', {})
            text = res.get('text', '').lower()
            caption = meta.get('image_caption', 'None').lower()
            
            score = 0
            
            # 1. Exact Phrase Match (Highest Priority)
            # If user types "He La Cell", and caption contains "He La Cell", this wins.
            if meta.get('type') == "image_index" and query_lower in caption:
                score += 1000
            
            # 2. Term Match
            elif meta.get('type') == "image_index" and any(t in caption for t in query_terms):
                score += 500
                
            score += sum(100 for t in query_terms if t in caption)
            score += sum(10 for t in query_terms if t in text)
            
            reranked.append((score, res))
        
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in reranked]

    def query_with_feedback(self, query):
        yield ("status", "🔍 Searching...")
        
        # Smart Expansion for "He La"
        search_queries = [query]
        if "hela" in query.lower():
            search_queries.append("he la")
        if "he la" in query.lower():
            search_queries.append("hela")

        all_results = []
        for q in search_queries:
            vec_res = self.db.search(q, top_k=20)
            key_res = self.db.search_keyword(q, top_k=15)
            all_results.extend(vec_res + key_res)

        # Deduplicate
        seen = set()
        unique_results = []
        for r in all_results:
            if r['text'] not in seen:
                unique_results.append(r); seen.add(r['text'])
        
        # Rerank
        results = self._smart_rerank(unique_results, query)[:12]

        if not results:
            yield ("status", "❌ Not found.")
            yield ("text", "I'm sorry, but the documents do not contain information on this topic.")
            return

        # Spatial Asset Selection
        # We now require a stronger match for the image to be displayed
        query_terms = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in {"the", "cell"}]
        
        target_asset = None
        for res in results:
            meta = res['metadata']
            cap = meta.get('image_caption', 'None').lower()
            if meta.get('type') == 'image_index':
                # Precision check: At least one specific word (not just 'cell') must match
                # Or exact phrase match
                if query.lower() in cap or any(k in cap for k in query_terms):
                    target_asset = {
                        "source": meta.get('source'), "page": meta.get('page'),
                        "rect": meta.get('image_rect'), "xref": meta.get('image_xref'),
                        "caption": meta.get('image_caption')
                    }
                    yield ("spatial_image", target_asset)
                    break

        yield ("status", "⚡ Synthesizing...")
        results.sort(key=lambda x: x['metadata'].get('page', 0))
        context = "\n".join([f"[Page {r['metadata']['page']}]: {r['text']}" for r in results])
        
        system_prompt = f"""You are a helpful assistant. Use ONLY the provided context.
        Cite page numbers.
        CONTEXT: {context}"""
        
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': query}]
        
        for chunk in self.llm.chat_stream(messages): yield ("text", chunk)
        yield ("chunks", results)