import ollama
import os

class VisionEngine:
    def __init__(self, model="moondream"): # <--- CHANGED TO MOONDREAM
        self.model = model

    def analyze_image(self, image_path, prompt):
        """
        Sends an image to Moondream (Low VRAM) for description.
        """
        if not os.path.exists(image_path):
            return "Error: Image file not found."

        try:
            with open(image_path, 'rb') as file:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [file.read()]
                        }
                    ]
                )
            return response['message']['content']
        except Exception as e:
            return f"Vision Error: {e}"