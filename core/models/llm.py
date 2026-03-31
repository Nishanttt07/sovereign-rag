import ollama
from core.config import LLM_MODEL

class LLMEngine:
    def __init__(self):
        self.model = LLM_MODEL

    def chat_stream(self, messages):
        """
        Streams response from Ollama (Llama 3.2).
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