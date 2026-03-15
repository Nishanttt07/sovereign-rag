import streamlit as st
import os
import shutil
import lancedb
import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

from core.config import RAW_PDFS_DIR, DB_DIR
from core.ingestion.pdf_processor import PDFProcessor
from core.retrieval.vector_search import VectorDB

def render_sidebar():
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    with st.sidebar:
        st.header("📂 Document Manager")
        
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="main_pdf_uploader")
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.processed_files:
                save_path = RAW_PDFS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_pill = st.empty()
                
                try:
                    status_pill.info(f"Extracting Text for {uploaded_file.name}...", icon="🧠")
                    
                    ingest_file(save_path, lambda msg: status_pill.info(msg, icon="🧠"))
                    
                    from core.worker import background_vision_worker
                    st.session_state.vision_processing = True
                    
                    vision_thread = threading.Thread(target=background_vision_worker, args=(save_path,))
                    add_script_run_ctx(vision_thread) 
                    vision_thread.start()
                    
                    status_pill.success("Text Ready! Diagrams indexing in background...", icon="✨")
                    st.session_state.processed_files.add(uploaded_file.name)
                    time.sleep(2) 
                    status_pill.empty()
                    st.rerun()
                    
                except Exception as e:
                    status_pill.error(f"Error: {e}", icon="❌")

        st.divider()

        with st.expander("🛠️ System Health & Debug", expanded=True):
            try:
                db = lancedb.connect(str(DB_DIR))
                if "vectors" in db.table_names():
                    tbl = db.open_table("vectors")
                    st.success(f"✅ DB Connected: {len(tbl)} chunks")
                else:
                    st.warning("⚠️ DB Empty")
            except Exception as e:
                st.error(f"❌ Error: {e}")

            if st.button("🔄 Full Reset"):
                status_pill = st.empty()
                status_pill.warning("Deleting Database...", icon="⚠️")
                
                if os.path.exists(DB_DIR):
                    try:
                        shutil.rmtree(DB_DIR)
                    except PermissionError:
                        status_pill.error("❌ Close other apps using this DB!")
                        return

                status_pill.info("System Wiped.", icon="🗑️")
                time.sleep(1)
                
                if os.path.exists(RAW_PDFS_DIR):
                    files = [f for f in os.listdir(RAW_PDFS_DIR) if f.endswith(".pdf")]
                    
                    from core.worker import background_vision_worker
                    for f in files:
                        file_path = RAW_PDFS_DIR / f
                        status_pill.info(f"♻️ Fast-reading {f}...", icon="🔄")
                        
                        ingest_file(file_path, lambda msg: status_pill.info(msg, icon="♻️"))
                        
                        st.session_state.vision_processing = True
                        vision_thread = threading.Thread(target=background_vision_worker, args=(file_path,))
                        add_script_run_ctx(vision_thread)  
                        vision_thread.start()
                    
                    status_pill.success("All Ready! Diagrams queueing.", icon="🚀")
                    time.sleep(2)
                    st.rerun()

        st.subheader("📚 Active Files")
        if os.path.exists(RAW_PDFS_DIR):
            for f in os.listdir(RAW_PDFS_DIR):
                st.caption(f"📄 {f}")

def ingest_file(file_path, status_callback=None):
    processor = PDFProcessor()
    db = VectorDB()
    total = 0
    
    for batch in processor.process_text_fast(file_path, status_callback):
        total += len(batch)
        db.add_chunks(batch)