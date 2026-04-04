import streamlit as st
import os
import shutil
import lancedb
import time
import threading
import json 
from streamlit.runtime.scriptrunner import add_script_run_ctx

from core.config import RAW_PDFS_DIR, DB_DIR, SQL_DB_PATH, DATA_DIR
from core.ingestion.pdf_processor import PDFProcessor
from core.ingestion.tabular_processor import TabularProcessor
from core.ingestion.office_processor import OfficeProcessor  # <-- NEW IMPORT
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
        
        # Accepts all necessary file types
        uploaded_files = st.file_uploader(
            "Upload Documents or Spreadsheets", 
            type=["pdf", "csv", "xlsx", "docx"], 
            key="main_uploader", 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    save_path = RAW_PDFS_DIR / uploaded_file.name
                    
                    with st.status(f"🔄 Processing **{uploaded_file.name}**...", expanded=True) as status_box:
                        status_box.write("💾 Saving file to disk...")
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        ext = uploaded_file.name.split('.')[-1].lower()
                        
                        try:
                            # PATH A: PDF
                            if ext == 'pdf':
                                status_box.write("🧠 Extracting text structure...")
                                ingest_file(save_path, lambda msg: status_box.write(f"⏳ {msg}"))
                                
                                from core.worker import background_vision_worker
                                st.session_state.vision_processing = True
                                vision_thread = threading.Thread(target=background_vision_worker, args=(save_path,))
                                add_script_run_ctx(vision_thread) 
                                vision_thread.start()
                                status_box.update(label=f"✅ {uploaded_file.name} Ready!", state="complete", expanded=False)
                                st.toast(f"{uploaded_file.name} text indexed! Diagrams processing in background.", icon="✨")
                            
                            # PATH B: Tabular (CSV / XLSX for SQL database)
                            elif ext in ['csv', 'xlsx']:
                                status_box.write("📊 Converting Tabular data to SQL...")
                                processor = TabularProcessor()
                                result = processor.process_file(save_path, status_callback=lambda msg: status_box.write(f"⏳ {msg}"))
                                
                                if result.get("status") == "success":
                                    status_box.update(label=f"✅ {uploaded_file.name} added to SQL Database!", state="complete", expanded=False)
                                    st.toast(f"Tabular data '{uploaded_file.name}' ready for querying!", icon="📊")
                                else:
                                    status_box.update(label=f"❌ Failed: {result.get('message')}", state="error")
                                    
                            # 🔥 PATH C: Word Document (DOCX for Vector database)
                            elif ext == 'docx':
                                status_box.write("📝 Extracting Markdown from Document...")
                                
                                processor = OfficeProcessor()
                                chunks = processor.process_file(
                                    str(save_path), 
                                    status_callback=lambda msg: status_box.write(f"⏳ {msg}")
                                )
                                
                                if chunks:
                                    status_box.write("💾 Adding chunks to Vector DB...")
                                    db = VectorDB()
                                    db.add_chunks(chunks)
                                    status_box.update(label=f"✅ {uploaded_file.name} added to Vector Database!", state="complete", expanded=False)
                                    st.toast(f"Markdown doc '{uploaded_file.name}' indexed successfully!", icon="🚀")
                                else:
                                    status_box.update(label=f"❌ Failed to process {uploaded_file.name}", state="error")

                            # Update memory and save to hard drive
                            st.session_state.processed_files.add(uploaded_file.name)
                            save_ingestion_log(st.session_state.processed_files)
                            
                        except Exception as e:
                            status_box.update(label=f"❌ Error processing {uploaded_file.name}: {e}", state="error")

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

            if st.button("🔄 Full Reset", use_container_width=True):
                with st.spinner("Deleting databases and clearing memory..."):
                    if os.path.exists(DB_DIR):
                        try:
                            shutil.rmtree(DB_DIR)
                        except PermissionError:
                            st.error("❌ Close other apps using this DB before resetting.")
                            return
                    
                    if os.path.exists(SQL_DB_PATH):
                        try:
                            os.remove(SQL_DB_PATH)
                        except Exception as e:
                            pass
                    
                    # Delete the permanent log book
                    if os.path.exists(INGESTION_LOG_PATH):
                        try:
                            os.remove(INGESTION_LOG_PATH)
                        except Exception:
                            pass

                    st.session_state.processed_files.clear()
                    time.sleep(1.5)
                    
                st.success("System completely wiped. Restarting...", icon="🗑️")
                time.sleep(1)
                st.rerun()

        st.subheader("📚 Active Files")
        if os.path.exists(RAW_PDFS_DIR):
            for f in os.listdir(RAW_PDFS_DIR):
                if f.lower().endswith(('.pdf', '.csv', '.xlsx', '.docx')):
                    if f in st.session_state.processed_files:
                        st.markdown(f"✅ `{f}`")
                    else:
                        st.caption(f"📄 `{f}` (Pending)")

def ingest_file(file_path, status_callback=None):
    processor = PDFProcessor()
    db = VectorDB()
    for batch in processor.process_text_fast(file_path, status_callback):
        db.add_chunks(batch)