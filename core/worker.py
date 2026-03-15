import streamlit as st
from core.ingestion.pdf_processor import PDFProcessor
from core.retrieval.vector_search import VectorDB

def background_vision_worker(file_path):
    """
    Runs in the background. Embeds the highly specific image coordinates 
    and citation text into LanceDB.
    """
    try:
        processor = PDFProcessor()
        db = VectorDB()

        assets = processor.extract_images_for_vision(file_path)
        vision_chunks = []
        
        for asset in assets:
            rich_context = f"DIAGRAM CAPTION: {asset['caption']} | TEXT EXPLANATION: {asset['citation_context']}"
            
            vision_chunks.append({
                "text": rich_context,
                "metadata": {
                    "source": asset['source'],
                    "page": asset['page'],
                    "type": "image_index",
                    "image_caption": asset['caption'], 
                    "image_rect": asset['rect'],
                    "image_xref": 0, 
                    "image_path": "None" # We strictly set this to None to trigger on-the-fly cropping!
                }
            })

        if vision_chunks:
            print(f"🧠 [BACKGROUND] Adding {len(vision_chunks)} verified diagrams to LanceDB...")
            # We just pass the chunks directly; VectorDB handles the embedding!
            db.add_chunks(vision_chunks)
            print(f"✅ [BACKGROUND] Processed {len(vision_chunks)} diagrams with 0 bytes of extra storage!")
            
    except Exception as e:
        print(f"Background Extraction Error: {e}")
    finally:
        st.session_state.vision_processing = False