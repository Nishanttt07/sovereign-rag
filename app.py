import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import os

from core.retrieval.hybrid_search import RAGPipeline
from ui.sidebar import render_sidebar
from core.config import DOMAIN_CONFIG # <-- IMPORT THE CONFIG

# Get the domain name, default to Sovereign RAG if not found
app_title = DOMAIN_CONFIG.get("domain_name", "Sovereign RAG")

# 1. PAGE CONFIGURATION
st.set_page_config(page_title=app_title, layout="wide", page_icon="⚡")
st.title(f"⚡ {app_title}") 

# Sidebar for file uploads and database management
render_sidebar()

# ... (The rest of your app.py remains exactly the same!) ...
# 2. CHAT HISTORY & STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vision_processing" not in st.session_state:
    st.session_state.vision_processing = False

# Show a UI indicator if images are being processed in the background
if st.session_state.vision_processing:
    st.toast("🖼️ Diagrams are silently indexing in the background...", icon="🤖")

# --- CACHE REMOVED ---
# We no longer cache this so it always sees the freshest data!
def get_pipeline():
    return RAGPipeline()

rag = get_pipeline()

def render_spatial_image(asset):
    """
    Loads the pre-extracted Vision image first for speed.
    Falls back to spatial PDF cropping if the image file isn't available.
    """
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
        status_box = st.status("🚀 Processing...", expanded=False)
        answer_placeholder = st.empty()
        
        full_response = ""
        current_spatial_assets = [] 
        retrieved_chunks = []

        # --- EVENT STREAM HANDLING ---
        for event_type, payload in rag.query_with_feedback(prompt):
            
            if event_type == "status":
                status_box.update(label=f"⚡ {payload}", state="running")
            
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
                
                # --- 🛑 DEBUG PRINT ADDED HERE 🛑 ---
                print("\n" + "="*60)
                print(f"🔍 DEBUG: TOP 5 SEARCH RESULTS FOR '{prompt}'")
                for i, chunk in enumerate(retrieved_chunks[:5]):
                    meta = chunk.get('metadata', {})
                    # Get the distance (lower is better) or custom keyword score
                    dist = chunk.get('_distance', 'N/A')
                    score = chunk.get('score', 'N/A') 
                    
                    print(f"\n--- Result {i+1} ---")
                    print(f"Page: {meta.get('page')}")
                    print(f"Type: {meta.get('type')}")
                    print(f"Vector Dist: {dist} | Keyword Score: {score}")
                    print(f"Text Snippet: {chunk.get('text', '').replace(chr(10), ' ')[:250]}...")
                print("="*60 + "\n")
                # ------------------------------------

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