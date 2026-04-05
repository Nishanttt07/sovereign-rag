import os
filepath = r'c:\sovereign_rag_project\ui\sidebar.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# split around 'def render_sidebar():'
parts = content.split('def render_sidebar():', 1)

new_func = '''def process_document(file_name, source_path=None, file_bytes=None):
    if file_name in st.session_state.processed_files:
        return
        
    save_path = RAW_PDFS_DIR / file_name
    
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

def render_sidebar():'''

new_content = parts[0] + new_func + parts[1]

start_marker = '        if uploaded_files:\n            for uploaded_file in uploaded_files:'
end_marker = '        st.divider()'

idx_start = new_content.find(start_marker)
idx_end = new_content.find(end_marker, idx_start)

replacement_block = '''        if uploaded_files:
            for uploaded_file in uploaded_files:
                process_document(uploaded_file.name, file_bytes=uploaded_file.getbuffer())

        st.divider()

        st.subheader("📁 Process Local Folder")
        folder_path = st.text_input("Enter absolute folder path (e.g., C:\\\\Documents):")
        if st.button("Process Folder"):
            if folder_path and os.path.exists(folder_path) and os.path.isdir(folder_path):
                valid_extensions = ('.pdf', '.csv', '.xlsx', '.docx', '.pptx', '.txt')
                processed_any = False
                for root_dir, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(valid_extensions):
                            source_path = os.path.join(root_dir, file)
                            process_document(file, source_path=source_path)
                            processed_any = True
                if not processed_any:
                    st.warning("No supported files found in the specified folder.")
                else:
                    st.success(f"Finished processing folder: {folder_path}")
            else:
                st.error("Invalid folder path or folder does not exist.")

'''

final_content = new_content[:idx_start] + replacement_block + new_content[idx_end + len('        st.divider()'):]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(final_content)

print('Done!')
