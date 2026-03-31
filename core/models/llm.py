import ollama
from core.config import LLM_MODEL

class LLMEngine:
    def __init__(self, model_name=None):
        # If a specific model is requested (e.g., qwen2.5-coder:3b), use it!
        # Otherwise, fall back to the default LLM_MODEL from config.py.
        self.model = model_name if model_name else LLM_MODEL

    def chat_stream(self, messages):
        """
        Streams response from Ollama using the dynamically assigned model.
        """
        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            yield f"LLM Error: {str(e)}"