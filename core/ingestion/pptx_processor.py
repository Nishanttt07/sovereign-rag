import os
from markitdown import MarkItDown
from langchain_text_splitters import MarkdownTextSplitter

class PPTXProcessor:
    def __init__(self):
        self.md_converter = MarkItDown()
        # Presentations often have less text per slide, smaller chunks work well
        self.splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=150)

    def process_file(self, file_path, status_callback=None):
        """
        Converts .pptx to Markdown and returns VectorDB-ready chunks.
        """
        file_name = os.path.basename(file_path)
        
        if status_callback:
            status_callback(f"Converting presentation {file_name} to Markdown...")
            
        try:
            # 1. Convert to Markdown using MarkItDown (natively supports .pptx)
            result = self.md_converter.convert(str(file_path))
            full_text = result.text_content
            
            if status_callback:
                status_callback("Chunking presentation slides...")
                
            # 2. Split into safe chunks
            text_chunks = self.splitter.create_documents([full_text])
            
            # 3. Format for VectorDB
            final_chunks = []
            for chunk in text_chunks:
                final_chunks.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "source": file_name,
                        "page": 1, # PPTX logic treats whole doc generically for now
                        "type": "presentation"
                    }
                })
                
            return final_chunks
            
        except Exception as e:
            print(f"❌ Error processing PowerPoint document: {e}")
            return []
