from core.retrieval.vector_search import VectorDB
from core.models.llm import LLMEngine
from core.config import SYSTEM_PROMPT, STOP_WORDS, SQL_TRIGGERS
from core.retrieval.sql_agent import SQLAgent
import re

class RAGPipeline:
    def __init__(self):
        self.db = VectorDB()
        self.llm = LLMEngine(model_name="llama3.2")

    def _smart_rerank(self, results, query):
        clean_query = re.sub(r'[^\w\s\.]', '', query.lower())
        query_lower = query.lower()

        # 🔥 Detect if a specific filename is mentioned in the user's prompt
        target_source = None
        for res in results:
            source_val = str(res.get('metadata', {}).get('source', 'None'))
            if source_val.lower() != 'none' and source_val.lower() in query_lower:
                target_source = source_val.lower()
                break

        reranked = []
        for res in results:
            meta = res.get('metadata', {})
            source_lower = str(meta.get('source', '')).lower()

            # 🔥 Hard Isolation: If a file was requested, drop chunks from all other files
            if target_source and target_source != source_lower:
                continue

            item_score = 0
            if '_distance' in res:
                distance = res.get('_distance', 1.0)
                item_score += max(0, (1.0 - distance) * 500)

            if 'score' in res:
                item_score += res['score'] * 50

            if meta.get('type') == "image_index":
                item_score += 100

            reranked.append((item_score, res))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in reranked]

    def _extract_images(self, results, query_lower):
        clean_query = re.sub(r'[^\w\s\.]', '', query_lower)
        query_terms = list(set([w for w in clean_query.split() if len(w) > 2 and w not in STOP_WORDS]))
        image_candidates = []
        
        for res in results:
            meta = res.get('metadata', {})
            if meta.get('type') == 'image_index':
                cap = meta.get('image_caption', 'None').lower()
                clean_cap = re.sub(r'[^\w\s\.]', '', cap)
                match_count = sum(1 for k in query_terms if k in clean_cap)
                
                if (clean_query in clean_cap) or (match_count >= 2):
                    image_candidates.append({
                        "source": meta.get('source'), "page": meta.get('page'),
                        "rect": meta.get('image_rect'), "xref": meta.get('image_xref', 0),
                        "caption": meta.get('image_caption'), "image_path": meta.get('image_path', 'None'),
                        "score": match_count 
                    })

        image_candidates.sort(key=lambda x: x['score'], reverse=True)
        return image_candidates[:2]

    def query_with_feedback(self, query):
        query_lower = query.lower().strip()
        sql_context = ""
        
        # 🔥 Conversational Short-Circuit: Handle basic greetings immediately without RAG
        greetings = {"hi", "hello", "hey", "who are you", "what are you", "hi there", "greetings"}
        if query_lower in greetings:
            yield ("text", "Hello! 👋 I am your local Sovereign RAG assistant. Please ask me a question about your uploaded documents.")
            return
        
        # 🔥 The Router: Catches SQL triggers OR spreadsheet extensions
        tabular_extensions = [".csv", ".xlsx"]
        is_sql_query = (
            any(trigger in query_lower for trigger in SQL_TRIGGERS) or 
            "table" in query_lower or 
            "row" in query_lower or
            any(ext in query_lower for ext in tabular_extensions)
        )

        # ==========================================
        # 1. SQL AGENT EXECUTION
        # ==========================================
        if is_sql_query:
            yield ("status", "📊 Fetching database records (Qwen 2.5)...")
            agent = SQLAgent()
            sql_context = agent.query(query)
            
            if sql_context and "Error" not in sql_context and "empty" not in sql_context.lower():
                yield ("text", f"### 🗄️ Retrieved Records:\n{sql_context}\n\n---\n**🤖 AI Analysis:**\n")
            elif "No matching" in sql_context:
                sql_context = "" 

        # ==========================================
        # 2. VECTOR SEARCH (Absolute Isolation)
        # ==========================================
        results = []
        # 🔥 ONLY run Vector Search if the SQL Agent didn't find anything
        if not sql_context:
            yield ("status", "🔍 Searching documents...")
            all_results = self.db.search(query, top_k=20) + self.db.search_keyword(query, top_k=8)
            
            # Intelligent Merge: Never overwrite Keyword scores with blank Semantic matches
            unique_results = {}
            for r in all_results:
                txt = r['text']
                if txt not in unique_results:
                    unique_results[txt] = r
                else:
                    if 'score' in r and 'score' not in unique_results[txt]:
                        unique_results[txt]['score'] = r['score']
                    if '_distance' in r:
                        old_dist = unique_results[txt].get('_distance', 1.0)
                        unique_results[txt]['_distance'] = min(old_dist, r['_distance'])
                    
            results = self._smart_rerank(list(unique_results.values()), query)[:20]

            # 🔥 Strict Adherence: Drop garbage chunks to prevent hallucination
            from core.config import SEARCH_THRESHOLD
            filtered = []
            for r in results:
                if r.get('_distance', 0) <= SEARCH_THRESHOLD:
                    filtered.append(r)
                elif r.get('score', 0) > 10:  # Keyword-only matches must be strong
                    filtered.append(r)
            results = filtered

            for asset in self._extract_images(results, query_lower): 
                yield ("spatial_image", asset)

            # 🔥 Absolute Grounding: If no solid context is found, stop here.
            if not results:
                yield ("text", "I'm sorry, but the provided documents do not contain information related to your query.\n")
                return

        # ==========================================
        # 3. LLM SYNTHESIS (Llama 3.2)
        # ==========================================
        yield ("status", "⚡ Synthesizing response (Llama 3.2)...")
        
        # 🔥 Dynamic Prompting: Strict rules for SQL vs Vector text
        if sql_context:
            context_to_send = f"DATABASE RECORDS:\n{sql_context}"
            task_instruction = """Task: Analyze and synthesize the provided Database Records into a clear, highly readable natural language summary. 
            
CRITICAL RULES:
1. **Interpretation First**: Begin your response by writing 2-3 sentences explicitly detailing your understanding of the user's question (e.g., "You asked for the last 10 records. I interpreted this as retrieving the final 10 chronologically ingested sequence entries...").
2. Do NOT output, repeat, or copy-paste the raw data as a table, grid, or pipe-separated list.
3. Group similar items together (e.g., 'Four issues were reported with DC Electric Motors...').
4. Summarize the key faults, resolutions, and trends in readable paragraphs or bullet points.
5. Do NOT explain how SQL works, do NOT give safety warnings, and do NOT mention internet connectivity. Just synthesize the data provided."""
        else:
            pdf_text = "\n".join([f"Page {r['metadata']['page']}: {r['text']}" for r in results])
            context_to_send = f"MANUAL EXCERPTS:\n{pdf_text}"
            task_instruction = (
                "Task: Answer the user's question using ONLY the provided CONTEXT.\n"
                "If the exact answer cannot be found in the CONTEXT, you must reply with exactly this phrase and nothing else: "
                "'I'm sorry, but the provided documents do not contain information about that.'"
            )

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"QUESTION / TASK: '{query}'\n\n---\n{task_instruction}\n\nCONTEXT:\n{context_to_send}"}
        ]
        
        for chunk in self.llm.chat_stream(messages): 
            yield ("text", chunk)
            
        yield ("chunks", results)