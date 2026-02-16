import streamlit as st
from core.retrieval.hybrid_search import RAGPipeline
from ui.sidebar import render_sidebar

# Page Config
st.set_page_config(page_title="Sovereign RAG", layout="wide")
st.title("⚡ Sovereign RAG")

# Sidebar
render_sidebar()

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

rag = get_pipeline()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show Images
        if "images" in message and message["images"]:
            for img_path in message["images"]:
                st.image(img_path, caption="Retrieved Context Image", width=400)
        
        st.markdown(message["content"])
        
        # Show Sources (if saved)
        if "sources" in message and message["sources"]:
            with st.expander("🔎 View Source Chunks"):
                for i, s in enumerate(message["sources"]):
                    st.caption(f"Source {i+1} - Page {s['metadata']['page']}")
                    st.text(s['text'][:300] + "...")

# User Input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.status("🚀 Starting...", expanded=False)
        answer_placeholder = st.empty()
        full_response = ""
        retrieved_images = []
        retrieved_sources = []

        # --- EVENT LOOP ---
        for event_type, payload in rag.query_with_feedback(prompt):
            
            if event_type == "status":
                status_box.update(label=f"⚡ {payload}", state="running")
            
            elif event_type == "sources":
                # Capture sources to show them later
                retrieved_sources = payload
                
            elif event_type == "image":
                st.image(payload, caption="📄 Relevant Visual Found", width=400)
                retrieved_images.append(payload)
                
            elif event_type == "text":
                full_response += payload
                answer_placeholder.markdown(full_response + "▌")

        # Final UI Updates
        status_box.update(label="✅ Complete", state="complete", expanded=False)
        answer_placeholder.markdown(full_response)
        
        # SHOW SOURCES in an Expander (Transparency)
        if retrieved_sources:
            with st.expander("🔎 Why did I say this? (View Sources)"):
                for i, s in enumerate(retrieved_sources):
                    st.caption(f"Source {i+1} - Page {s['metadata'].get('page', '?')}")
                    # Show snippets of text so you verify the context
                    st.code(s['text'][:300] + "...", language="text")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "images": retrieved_images,
            "sources": retrieved_sources
        })