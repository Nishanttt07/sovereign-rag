from core.retrieval.vector_search import VectorDB
from core.models.llm import LLMEngine
from core.models.vision import VisionEngine
import os

class RAGPipeline:
    def __init__(self):
        self.db = VectorDB()
        self.llm = LLMEngine()
        self.vision = VisionEngine() 

    def _smart_rerank(self, results, query):
        """
        Generic Reranker: Boosts results where query terms appear in 
        metadata captions, regardless of the document type.
        """
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        if not query_terms:
            return results
        
        reranked, normal = [], []
        for res in results:
            meta = res.get('metadata', {})
            text = res.get('text', '').lower()
            caption = meta.get('image_caption', 'None').lower()
            type_ = meta.get('type', 'text')
            
            score = 0
            # Boost matches in extracted image captions
            if type_ == "image_index" and any(term in caption for term in query_terms):
                score += 100
            
            score += sum(20 for term in query_terms if term in caption)
            score += sum(1 for term in query_terms if term in text)
            
            if score > 0:
                reranked.append((score, res))
            else:
                normal.append(res)
                
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in reranked] + normal

    def query_with_feedback(self, query):
        yield ("status", "🔍 Searching Knowledge Base...")
        
        # 1. RETRIEVAL
        vector_results = self.db.search(query, top_k=25)
        keyword_results = self.db.search_keyword(query, top_k=10)
        
        combined = keyword_results + vector_results
        seen, unique_results = set(), []
        for r in combined:
            if r['text'] not in seen:
                unique_results.append(r)
                seen.add(r['text'])
        
        results = self._smart_rerank(unique_results, query)[:10]

        # REJECTION GUARDRAIL: If no results found, do not answer using training data.
        if not results:
            yield ("status", "❌ Information not found.")
            yield ("text", "I couldn't find any information regarding that in the uploaded documents.")
            return

        yield ("status", f"✅ Found {len(results)} relevant segments.")

        # 2. IMAGE SELECTION (Domain-Agnostic)
        target_image, target_caption = None, "None"
        ignore = {"show", "me", "the", "of", "and", "diagram", "image", "figure", "explain", "some"}
        query_words = [w.lower() for w in query.split() if w.lower() not in ignore and len(w) > 2]
        
        for res in results:
            meta = res.get('metadata', {})
            img_path = meta.get('image_path')
            caption = meta.get('image_caption', 'None').lower()
            
            if img_path and img_path != "None" and os.path.exists(img_path):
                if any(w in caption for w in query_words):
                    target_image, target_caption = img_path, meta.get('image_caption')
                    yield ("image", target_image)
                    break
        
        # 3. GENERATION PHASE
        yield ("status", "⚡ Synthesizing Response...")
        
        context_text = ""
        # Dynamically identify the source files to set the context "vibe"
        sources_found = set()
        for i, res in enumerate(results):
            src = res['metadata'].get('source', 'Unknown')
            sources_found.add(src)
            context_text += f"--- Source {i+1} (File: {src}, Page: {res['metadata'].get('page')}) ---\n{res['text']}\n\n"

        visual_desc = ""
        if target_image:
            vision_prompt = f"Analyze this visual element labeled: {target_caption}. Identify its key components."
            visual_desc = self.vision.analyze_image(target_image, vision_prompt)

        # AGNOSTIC SYSTEM PROMPT
        system_prompt = f"""You are a Knowledge Retrieval Assistant. 
        Your ONLY source of truth is the 'PROVIDED CONTEXT' below.

        PROVIDED CONTEXT:
        {context_text}

        VISUAL CONTEXT (if any):
        {visual_desc}

        STRICT OPERATIONAL RULES:
        1. NO OUTSIDE KNOWLEDGE: Your world is limited to the provided context. If the query is not addressed in the text, respond: "The provided documents do not contain information on this topic."
        2. DATA-DRIVEN ROLE: Adopt a tone suitable for the content of the documents (e.g., technical if the text is technical, academic if it is a textbook).
        3. CONCEPTUAL EXPLANATION: Use the text context to explain the user's query. Use the visual context only as a supportive reference.
        4. No hallucinations or assumptions.
        """
        
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': query}]
        for chunk in self.llm.chat_stream(messages):
            yield ("text", chunk)