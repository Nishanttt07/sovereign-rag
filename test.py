# test_ingest.py
from core.ingestion.pdf_processor import PDFProcessor

# Change this to a path of a real PDF on your computer
TEST_PDF = "data/raw_pdfs/test_doc.pdf" 

if __name__ == "__main__":
    # Create a dummy PDF if you don't have one, or manually put one in data/raw_pdfs/
    processor = PDFProcessor()
    
    # Make sure you put a PDF file at data/raw_pdfs/test_doc.pdf first!
    try:
        chunks = processor.process_pdf(TEST_PDF)
        
        print("\n--- EXTRACTED CHUNKS SAMPLE ---")
        for i, chunk in enumerate(chunks[:5]): # Show first 5 chunks
            print(f"[{i}] Type: {chunk['metadata']['type']} | Content: {chunk['text'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")
        print("Did you place a file named 'test_doc.pdf' in 'data/raw_pdfs/'?")