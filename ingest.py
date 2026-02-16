import os
import shutil
from core.config import RAW_PDFS_DIR
from core.ingestion.pdf_processor import PDFProcessor
# FIX: Removed '.py' from the end of this line
from core.retrieval.vector_search import VectorDB

def main():
    # Initialize Engines
    processor = PDFProcessor()
    db = VectorDB()

    # List all PDFs in the 'raw_pdfs' folder
    pdf_files = [f for f in os.listdir(RAW_PDFS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDFs found in {RAW_PDFS_DIR}. Please add some!")
        return

    print(f"📂 Found {len(pdf_files)} PDFs to process.")

    for pdf_file in pdf_files:
        pdf_path = RAW_PDFS_DIR / pdf_file
        
        # 1. Process (Extract Text + Images)
        chunks = processor.process_pdf(pdf_path)
        
        # 2. Save to Database
        if chunks:
            db.add_chunks(chunks)
        
        # Optional: Move file to a "processed" folder so we don't read it twice?
        # For now, we leave it there.

    print("\n✅ Ingestion Complete! You can now query your documents.")

if __name__ == "__main__":
    main()