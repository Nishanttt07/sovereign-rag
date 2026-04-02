import os
from markitdown import MarkItDown
from langchain_text_splitters import MarkdownTextSplitter

class OfficeProcessor:
    def __init__(self):
        self.md_converter = MarkItDown()
        # Using 1500 chunk size to prevent wide tables from being split in half
        self.splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)

    def process_file(self, file_path, status_callback=None):
        """
        Converts .docx (and other office files) to Markdown and returns VectorDB-ready chunks.
        """
        file_name = os.path.basename(file_path)
        
        if status_callback:
            status_callback(f"Converting {file_name} to Markdown...")
            
        try:
            # 1. Convert to Markdown
            result = self.md_converter.convert(str(file_path))
            full_text = result.text_content
            
            if status_callback:
                status_callback("Chunking Markdown data...")
                
            # 2. Split into safe chunks
            text_chunks = self.splitter.create_documents([full_text])
            
            # 3. Format for VectorDB
            final_chunks = []
            for chunk in text_chunks:
                final_chunks.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "source": file_name,
                        "page": 1, # Office docs don't have strict physical pages like PDFs
                        "type": "text"
                    }
                })
                
            return final_chunks
            
        except Exception as e:
            print(f"❌ Error processing Office document: {e}")
            return []