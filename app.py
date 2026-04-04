import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import tempfile 

from core.retrieval.hybrid_search import RAGPipeline
from ui.sidebar import render_sidebar
from core.config import DOMAIN_CONFIG 

# Get the domain name, default to Sovereign RAG if not found
app_title = DOMAIN_CONFIG.get("domain_name", "Sovereign RAG")

# 1. PAGE CONFIGURATION
st.set_page_config(page_title=app_title, layout="wide", page_icon="⚡")
st.title(f"⚡ {app_title}") 

def get_pipeline():
    return RAGPipeline()

# Initialize the pipeline early so the sidebar can access it
rag = get_pipeline()

# Sidebar for file uploads and database management
with st.sidebar:
    render_sidebar()
    
    st.divider()


# 2. CHAT HISTORY & STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vision_processing" not in st.session_state:
    st.session_state.vision_processing = False

# Show a UI indicator if images are being processed in the background
if st.session_state.vision_processing:
    st.toast("🖼️ Diagrams are silently indexing in the background...", icon="🤖")

def render_spatial_image(asset):
    try:
        image_path = asset.get("image_path", "None")
        if image_path != "None" and os.path.exists(image_path):
            return Image.open(image_path)
        
        doc = fitz.open(asset["source"])
        page = doc[int(asset["page"]) - 1]
        rect = fitz.Rect(asset["rect"])
        
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.error(f"Error rendering image: {e}")
        return None

# 3. DISPLAY MESSAGE HISTORY
if len(st.session_state.messages) == 0:
    st.markdown(f"""
        <div style="text-align: center; padding: 4rem 2rem; border-radius: 15px; background: rgba(120, 120, 120, 0.05); margin: 2rem 0; border: 1px solid rgba(120, 120, 120, 0.2);">
            <div style="font-size: 3.5rem; display: inline-block; animation: float 3s ease-in-out infinite;">⚡</div>
            <h1 style="font-size: 2.2rem; margin-top: 1rem; color: #4A90E2; font-weight: 600;">{app_title}</h1>
            <p style="font-size: 1.1rem; color: #888; margin-top: 0.5rem;">
                Your intelligent document assistant. Upload files and start asking questions!
            </p>
            <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; color: #aaa;">
                <div style="padding: 10px 20px; border-radius: 20px; background: rgba(120, 120, 120, 0.1);">📄 PDFs</div>
                <div style="padding: 10px 20px; border-radius: 20px; background: rgba(120, 120, 120, 0.1);">📊 Spreadsheets</div>
                <div style="padding: 10px 20px; border-radius: 20px; background: rgba(120, 120, 120, 0.1);">📝 Word Docs</div>
            </div>
        </div>
        <style>
            @keyframes float {{
                0% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-10px); }}
                100% {{ transform: translateY(0px); }}
            }}
        </style>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "spatial_assets" in message and message["spatial_assets"]:
                for asset in message["spatial_assets"]:
                    img = render_spatial_image(asset)
                    if img:
                        st.image(
                            img, 
                            caption=f"Reference: {asset['caption']}", 
                            width=500
                        )
            
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("🔎 View Source Chunks"):
                    for i, s in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1} (Page {s['metadata'].get('page', '?')})**")
                        st.info(s['text'])
                        st.divider()

# 4. USER INTERACTION LOOP
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.status("🚀 Processing...", expanded=True)
        answer_placeholder = st.empty()
        
        full_response = ""
        current_spatial_assets = [] 
        retrieved_chunks = []

        # --- EVENT STREAM HANDLING ---
        for event_type, payload in rag.query_with_feedback(prompt):
            
            if event_type == "status":
                status_box.write(f"🔄 {payload}")
                status_box.update(label=f"🔄 {payload}", state="running")
            
            elif event_type == "spatial_image":
                current_spatial_assets.append(payload)
                img = render_spatial_image(payload)
                if img:
                    st.image(
                        img, 
                        caption=f"📄 {payload['caption']}", 
                        width=500
                    )
            
            elif event_type == "text":
                full_response += payload
                answer_placeholder.markdown(full_response + "▌")
                
            elif event_type == "chunks":
                retrieved_chunks = payload
                
                print("\n" + "="*60)
                print(f"🔍 DEBUG: TOP 5 SEARCH RESULTS FOR '{prompt}'")
                for i, chunk in enumerate(retrieved_chunks[:5]):
                    meta = chunk.get('metadata', {})
                    dist = chunk.get('_distance', 'N/A')
                    score = chunk.get('score', 'N/A') 
                    
                    print(f"\n--- Result {i+1} ---")
                    print(f"Page: {meta.get('page')}")
                    print(f"Type: {meta.get('type')}")
                    print(f"Vector Dist: {dist} | Keyword Score: {score}")
                    print(f"Text Snippet: {chunk.get('text', '').replace(chr(10), ' ')[:250]}...")
                print("="*60 + "\n")

        status_box.update(label="✅ Complete", state="complete", expanded=False)
        answer_placeholder.markdown(full_response)
        
        if retrieved_chunks:
            with st.expander("📚 View Source References"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i+1} (Page {chunk['metadata'].get('page', '?')})**")
                    st.info(chunk['text'])
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "spatial_assets": current_spatial_assets,
            "sources": retrieved_chunks
        })