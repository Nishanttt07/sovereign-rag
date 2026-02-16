import streamlit as st
import os
import shutil
import lancedb
import time
from core.config import RAW_PDFS_DIR, DB_DIR
from core.ingestion.pdf_processor import PDFProcessor
from core.retrieval.vector_search import VectorDB

def render_sidebar():
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    with st.sidebar:
        st.header("📂 Document Manager")
        
        # --- UPLOAD SECTION ---
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.processed_files:
                save_path = RAW_PDFS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_pill = st.empty()
                
                try:
                    status_pill.info(f"Starting {uploaded_file.name}...", icon="🧠")
                    # Pass the lambda to update the pill
                    ingest_file(save_path, lambda msg: status_pill.info(msg, icon="🧠"))
                    
                    status_pill.success("Done!", icon="✨")
                    st.session_state.processed_files.add(uploaded_file.name)
                    time.sleep(1) 
                    status_pill.empty()
                    st.rerun()
                    
                except Exception as e:
                    status_pill.error(f"Error: {e}", icon="❌")

        st.divider()

        # --- SYSTEM HEALTH ---
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

                status_pill.info("Database Wiped.", icon="🗑️")
                time.sleep(1)
                
                if os.path.exists(RAW_PDFS_DIR):
                    files = [f for f in os.listdir(RAW_PDFS_DIR) if f.endswith(".pdf")]
                    
                    for f in files:
                        status_pill.info(f"♻️ Re-reading {f}...", icon="🔄")
                        ingest_file(RAW_PDFS_DIR / f, lambda msg: status_pill.info(msg, icon="♻️"))
                    
                    status_pill.success("All Ready!", icon="🚀")
                    time.sleep(1)
                    st.rerun()

        st.subheader("📚 Active Files")
        if os.path.exists(RAW_PDFS_DIR):
            for f in os.listdir(RAW_PDFS_DIR):
                st.caption(f"📄 {f}")

def ingest_file(file_path, status_callback=None):
    processor = PDFProcessor()
    db = VectorDB()
    total = 0
    
    # We pass the status_callback deeper into the processor
    for batch in processor.process_pdf_stream(file_path, status_callback):
        total += len(batch)
        db.add_chunks(batch)