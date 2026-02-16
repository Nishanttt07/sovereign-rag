import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Sub-directories
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
DB_DIR = DATA_DIR / "lancedb"
GRAPH_DIR = DATA_DIR / "knowledge_graph"

# Model Names
LLM_MODEL = "phi3:mini"
VISION_MODEL = "moondream"
# THE HEAVYWEIGHT MODEL:
EMBEDDING_MODEL = "mxbai-embed-large" 

# System Settings
# mxbai handles 512 tokens well. 
# We set chunk size to 400 to be safe and leave room for metadata.
CHUNK_SIZE = 400  
CHUNK_OVERLAP = 50
OLLAMA_URL = "http://localhost:11434"

# Rejection Threshold. 
# If the best match is less relevant than this, we return "Not Found".
SEARCH_THRESHOLD = 0.65

# Ensure directories exist
for d in [RAW_PDFS_DIR, EXTRACTED_IMAGES_DIR, DB_DIR, GRAPH_DIR]:
    d.mkdir(parents=True, exist_ok=True)