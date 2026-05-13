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
from core.ingestion.pptx_processor import PPTXProcessor
from core.ingestion.txt_processor import TXTProcessor
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

def process_document(file_name, source_path=None, file_bytes=None):
    if file_name in st.session_state.processed_files:
        return
        
    save_path = RAW_PDFS_DIR / file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with st.status(f"🔄 Processing **{file_name}**...", expanded=True) as status_box:
        try:
            if file_bytes is not None:
                status_box.write("💾 Saving file to disk...")
                with open(save_path, "wb") as f:
                    f.write(file_bytes)
            elif source_path is not None:
                if str(os.path.abspath(source_path)) != str(os.path.abspath(save_path)):
                    status_box.write("💾 Copying file to working directory...")
                    import shutil
                    shutil.copy2(source_path, save_path)
            
            ext = file_name.split('.')[-1].lower()
            
            # PATH A: PDF
            if ext == 'pdf':
                status_box.write("🧠 Extracting text structure...")
                ingest_file(save_path, lambda msg: status_box.write(f"⏳ {msg}"))
                
                from core.worker import background_vision_worker
                st.session_state.vision_processing = True
                vision_thread = threading.Thread(target=background_vision_worker, args=(save_path,))
                add_script_run_ctx(vision_thread) 
                vision_thread.start()
                status_box.update(label=f"✅ {file_name} Ready!", state="complete", expanded=False)
                st.toast(f"{file_name} text indexed! Diagrams processing in background.", icon="✨")
            
            # PATH B: Tabular (CSV / XLSX for SQL database)
            elif ext in ['csv', 'xlsx']:
                status_box.write("📊 Converting Tabular data to SQL...")
                processor = TabularProcessor()
                result = processor.process_file(save_path, status_callback=lambda msg: status_box.write(f"⏳ {msg}"))
                
                if result.get("status") == "success":
                    status_box.update(label=f"✅ {file_name} added to SQL Database!", state="complete", expanded=False)
                    st.toast(f"Tabular data '{file_name}' ready for querying!", icon="📊")
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
                    status_box.update(label=f"✅ {file_name} added to Vector Database!", state="complete", expanded=False)
                    st.toast(f"Markdown doc '{file_name}' indexed successfully!", icon="🚀")
                else:
                    status_box.update(label=f"❌ Failed to process {file_name}", state="error")

            # 🔥 PATH D: PowerPoint (PPTX for Vector database)
            elif ext == 'pptx':
                status_box.write("📽️ Extracting Text from Presentation...")
                
                processor = PPTXProcessor()
                chunks = processor.process_file(
                    str(save_path), 
                    status_callback=lambda msg: status_box.write(f"⏳ {msg}")
                )
                
                if chunks:
                    status_box.write("💾 Adding chunks to Vector DB...")
                    db = VectorDB()
                    db.add_chunks(chunks)
                    status_box.update(label=f"✅ {file_name} added to Vector Database!", state="complete", expanded=False)
                    st.toast(f"Presentation '{file_name}' indexed successfully!", icon="🚀")
                else:
                    status_box.update(label=f"❌ Failed to process {file_name}", state="error")

            # 🔥 PATH E: Text Files (TXT for Vector database)
            elif ext == 'txt':
                status_box.write("📄 Processing Raw Text...")
                
                processor = TXTProcessor()
                chunks = processor.process_file(
                    str(save_path), 
                    status_callback=lambda msg: status_box.write(f"⏳ {msg}")
                )
                
                if chunks:
                    status_box.write("💾 Adding chunks to Vector DB...")
                    db = VectorDB()
                    db.add_chunks(chunks)
                    status_box.update(label=f"✅ {file_name} added to Vector Database!", state="complete", expanded=False)
                    st.toast(f"Text file '{file_name}' indexed successfully!", icon="🚀")
                else:
                    status_box.update(label=f"❌ Failed to process {file_name}", state="error")

            # Update memory and save to hard drive
            st.session_state.processed_files.add(file_name)
            save_ingestion_log(st.session_state.processed_files)
            
        except Exception as e:
            status_box.update(label=f"❌ Error processing {file_name}: {e}", state="error")

def render_sidebar():
    # Load the permanent log from the hard drive instead of just session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = load_ingestion_log()

    SUPPORTED_EXTENSIONS = ('.pdf', '.csv', '.xlsx', '.docx', '.pptx', '.txt')

    with st.sidebar:
        st.header("📂 Document Manager")
        
        folder_mode = st.toggle("📂 Enable folder selection")
        
        if folder_mode:
            # --- Folder path mode: user provides a local folder path ---
            folder_path = st.text_input(
                "📁 Enter folder path",
                placeholder=r"C:\Users\you\Documents\my_docs",
            )
            if folder_path and os.path.isdir(folder_path):
                found_files = []
                for root, _, files in os.walk(folder_path):
                    for f in files:
                        if f.lower().endswith(SUPPORTED_EXTENSIONS):
                            found_files.append(os.path.join(root, f))
                
                if found_files:
                    st.caption(f"Found **{len(found_files)}** supported file(s)")
                    if st.button("📥 Ingest all files", use_container_width=True):
                        for fpath in found_files:
                            process_document(
                                os.path.basename(fpath),
                                source_path=fpath
                            )
                else:
                    st.warning("No supported files found in this folder.")
            elif folder_path:
                st.error("❌ Folder not found. Please check the path.")
        else:
            # --- Normal file uploader mode ---
            uploaded_files = st.file_uploader(
                "Upload Documents or Spreadsheets", 
                type=["pdf", "csv", "xlsx", "docx", "pptx", "txt"], 
                key="main_uploader", 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    process_document(uploaded_file.name, file_bytes=uploaded_file.getbuffer())

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
            for root, _, files in os.walk(RAW_PDFS_DIR):
                for f in files:
                    if f.lower().endswith(('.pdf', '.csv', '.xlsx', '.docx', '.pptx', '.txt')):
                        rel_path = os.path.relpath(os.path.join(root, f), RAW_PDFS_DIR)
                        rel_path_fwd = rel_path.replace('\\', '/')
                        
                        if rel_path_fwd in st.session_state.processed_files or f in st.session_state.processed_files:
                            st.markdown(f"✅ `{rel_path_fwd}`")
                        else:
                            st.caption(f"📄 `{rel_path_fwd}` (Pending)")

def ingest_file(file_path, status_callback=None):
    processor = PDFProcessor()
    db = VectorDB()
    for batch in processor.process_text_fast(file_path, status_callback):
        db.add_chunks(batch)