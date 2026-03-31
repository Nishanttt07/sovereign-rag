import streamlit as st
import os
import shutil
import lancedb
import time
import threading
import json # <--- NEW IMPORT
from streamlit.runtime.scriptrunner import add_script_run_ctx

from core.config import RAW_PDFS_DIR, DB_DIR, SQL_DB_PATH, DATA_DIR
from core.ingestion.pdf_processor import PDFProcessor
from core.ingestion.tabular_processor import TabularProcessor
from core.retrieval.vector_search import VectorDB

# --- PERMANENT MEMORY LOGIC ---
INGESTION_LOG_PATH = DATA_DIR / "ingestion_log.json"

def load_ingestion_log():
    if os.path.exists(INGESTION_LOG_PATH):
        with open(INGESTION_LOG_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_ingestion_log(files_set):
    with open(INGESTION_LOG_PATH, "w") as f:
        json.dump(list(files_set), f)

def render_sidebar():
    # Load the permanent log from the hard drive instead of just session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = load_ingestion_log()

    with st.sidebar:
        st.header("📂 Document Manager")
        
        uploaded_files = st.file_uploader(
            "Upload Documents or Spreadsheets", 
            type=["pdf", "csv", "xlsx"], 
            key="main_uploader", 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    save_path = RAW_PDFS_DIR / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    ext = uploaded_file.name.split('.')[-1].lower()
                    status_pill = st.empty()
                    
                    try:
                        # PATH A: PDF
                        if ext == 'pdf':
                            status_pill.info(f"Extracting Text for {uploaded_file.name}...", icon="🧠")
                            ingest_file(save_path, lambda msg: status_pill.info(msg, icon="🧠"))
                            
                            from core.worker import background_vision_worker
                            st.session_state.vision_processing = True
                            vision_thread = threading.Thread(target=background_vision_worker, args=(save_path,))
                            add_script_run_ctx(vision_thread) 
                            vision_thread.start()
                            status_pill.success(f"{uploaded_file.name} Ready! Diagrams indexing in background...", icon="✨")
                        
                        # PATH B: Tabular
                        elif ext in ['csv', 'xlsx']:
                            status_pill.info(f"Converting {uploaded_file.name} to SQL Database...", icon="📊")
                            processor = TabularProcessor()
                            result = processor.process_file(save_path, status_callback=lambda msg: status_pill.info(msg, icon="📊"))
                            
                            if result.get("status") == "success":
                                status_pill.success(f"✅ {uploaded_file.name} added to SQL Database!")
                            else:
                                status_pill.error(f"❌ Failed: {result.get('message')}")

                        # Update memory and save to hard drive!
                        st.session_state.processed_files.add(uploaded_file.name)
                        save_ingestion_log(st.session_state.processed_files)
                        
                        time.sleep(1.5) 
                        status_pill.empty()
                        
                    except Exception as e:
                        status_pill.error(f"Error processing {uploaded_file.name}: {e}", icon="❌")

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
                status_pill.warning("Deleting Databases...", icon="⚠️")
                
                if os.path.exists(DB_DIR):
                    try:
                        shutil.rmtree(DB_DIR)
                    except PermissionError:
                        status_pill.error("❌ Close other apps using this DB!")
                        return
                
                if os.path.exists(SQL_DB_PATH):
                    try:
                        os.remove(SQL_DB_PATH)
                    except Exception as e:
                        pass
                
                # Delete the permanent log book
                if os.path.exists(INGESTION_LOG_PATH):
                    os.remove(INGESTION_LOG_PATH)

                status_pill.info("System Wiped.", icon="🗑️")
                st.session_state.processed_files.clear()
                time.sleep(1)
                st.rerun()

        st.subheader("📚 Active Files")
        if os.path.exists(RAW_PDFS_DIR):
            for f in os.listdir(RAW_PDFS_DIR):
                if f.lower().endswith(('.pdf', '.csv', '.xlsx')):
                    if f in st.session_state.processed_files:
                        st.markdown(f"✅ `{f}`")
                    else:
                        st.caption(f"📄 `{f}` (Pending)")

def ingest_file(file_path, status_callback=None):
    processor = PDFProcessor()
    db = VectorDB()
    for batch in processor.process_text_fast(file_path, status_callback):
        db.add_chunks(batch)