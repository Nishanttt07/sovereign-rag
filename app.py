import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
from core.retrieval.hybrid_search import RAGPipeline
from ui.sidebar import render_sidebar

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Sovereign RAG", layout="wide", page_icon="⚡")
st.title("⚡ Sovereign RAG: Virtual Spatial Mapping")

# Sidebar for file uploads and database management
render_sidebar()

# Initialize Retrieval Pipeline
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

rag = get_pipeline()

# 2. CHAT HISTORY INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_spatial_image(asset):
    """
    Crops an image directly from the source PDF using spatial coordinates.
    This replaces the need for physical image storage.
    """
    try:
        doc = fitz.open(asset["source"])
        # Page numbers in metadata are 1-based, fitz uses 0-based
        page = doc[int(asset["page"]) - 1]
        
        # Define the crop area from the stored rect [x0, y0, x1, y1]
        rect = fitz.Rect(asset["rect"])
        
        # Increase resolution for better UI display (zoom factor 2)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # Convert to PIL Image for Streamlit
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.error(f"Error rendering spatial image: {e}")
        return None

# 3. DISPLAY MESSAGE HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If the message contains a spatial asset, render the crop
        if "spatial_asset" in message and message["spatial_asset"]:
            img = render_spatial_image(message["spatial_asset"])
            if img:
                st.image(
                    img, 
                    caption=f"Reference: {message['spatial_asset']['caption']}", 
                    use_container_width=False, 
                    width=500
                )
        
        st.markdown(message["content"])
        
        # Display source references in an expander
        if "sources" in message and message["sources"]:
            with st.expander("🔎 View Source Chunks"):
                for i, s in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1} (Page {s['metadata'].get('page', '?')})**")
                    st.info(s['text'])
                    st.divider()

# 4. USER INTERACTION LOOP
if prompt := st.chat_input("Ask about your documents..."):
    # Store user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.status("🚀 Processing...", expanded=False)
        answer_placeholder = st.empty()
        
        full_response = ""
        current_spatial_asset = None
        retrieved_chunks = []

        # --- EVENT STREAM HANDLING ---
        for event_type, payload in rag.query_with_feedback(prompt):
            
            if event_type == "status":
                status_box.update(label=f"⚡ {payload}", state="running")
            
            elif event_type == "spatial_image":
                # Received coordinate data for a real-time crop
                current_spatial_asset = payload
                img = render_spatial_image(payload)
                if img:
                    st.image(
                        img, 
                        caption=f"📄 {payload['caption']}", 
                        use_container_width=False, 
                        width=500
                    )
            
            elif event_type == "text":
                # Stream text response from LLM
                full_response += payload
                answer_placeholder.markdown(full_response + "▌")
                
            elif event_type == "chunks":
                # Store raw chunks for references
                retrieved_chunks = payload

        # Final UI Cleanup
        status_box.update(label="✅ Complete", state="complete", expanded=False)
        answer_placeholder.markdown(full_response)
        
        # Render Source References for the current response
        if retrieved_chunks:
            with st.expander("📚 View Source References"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i+1} (Page {chunk['metadata'].get('page', '?')})**")
                    st.info(chunk['text'])
                    st.divider()

        # Save assistant interaction to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "spatial_asset": current_spatial_asset,
            "sources": retrieved_chunks
        })