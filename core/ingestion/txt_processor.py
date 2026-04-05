import os
from langchain_text_splitters import MarkdownTextSplitter

class TXTProcessor:
    def __init__(self):
        self.splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=150)

    def process_file(self, file_path, status_callback=None):
        """
        Reads raw .txt files and returns VectorDB-ready chunks.
        """
        file_name = os.path.basename(file_path)
        
        if status_callback:
            status_callback(f"Reading text from {file_name}...")
            
        try:
            # Native Python file read (ignores decoding errors)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
            
            if status_callback:
                status_callback("Chunking text data...")
                
            # Split into safe chunks
            text_chunks = self.splitter.create_documents([full_text])
            
            # Format for VectorDB
            final_chunks = []
            for chunk in text_chunks:
                final_chunks.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "source": file_name,
                        "page": 1, 
                        "type": "text"
                    }
                })
                
            return final_chunks
            
        except Exception as e:
            print(f"❌ Error processing TXT document: {e}")
            return []
