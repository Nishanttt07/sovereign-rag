import os
from pathlib import Path
import yaml

# --- 1. PATH DEFINITIONS ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Sub-directories
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
DB_DIR = DATA_DIR / "lancedb"
GRAPH_DIR = DATA_DIR / "knowledge_graph"
SQL_DB_DIR = DATA_DIR / "sqldb" # <--- NEW: SQLite Directory

# Specific file paths
SQL_DB_PATH = SQL_DB_DIR / "tabular_knowledge.db" # <--- NEW: SQLite DB File

# Ensure directories exist
for d in [RAW_PDFS_DIR, EXTRACTED_IMAGES_DIR, DB_DIR, GRAPH_DIR, SQL_DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- 2. LOCAL MODEL CONFIGURATION ---
LLM_MODEL = "llama3:8b"
VISION_MODEL = "moondream"
# THE HEAVYWEIGHT MODEL:
EMBEDDING_MODEL = "mxbai-embed-large" 

# System Settings
CHUNK_SIZE = 400  
CHUNK_OVERLAP = 50
OLLAMA_URL = "http://localhost:11434"

# --- 3. DYNAMIC DOMAIN CONFIGURATION (YAML) ---
CONFIG_FILE_PATH = BASE_DIR / "domain_config.yaml"

def load_domain_config():
    """Loads the YAML configuration file. Falls back to defaults if missing."""
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file) or {}
    else:
        print(f"⚠️ Warning: {CONFIG_FILE_PATH} not found. Using hardcoded fallbacks.")
        return {}

DOMAIN_CONFIG = load_domain_config()

# --- 4. EXTRACTED YAML VARIABLES ---
SYSTEM_PROMPT = DOMAIN_CONFIG.get("llm_persona", "You are a helpful assistant. Use ONLY the provided context.")
VISION_LABELS = DOMAIN_CONFIG.get("vision_extraction", {}).get("target_labels", ["Fig", "Figure", "Table", "Diagram"])
VISION_RADIUS = DOMAIN_CONFIG.get("vision_extraction", {}).get("geometric_search_radius", 120)
SEARCH_THRESHOLD = DOMAIN_CONFIG.get("search_tuning", {}).get("dynamic_threshold_percent", 0.65)
STOP_WORDS = set(DOMAIN_CONFIG.get("search_tuning", {}).get("custom_stop_words", ["the", "is", "a", "of", "to"]))
SQL_TRIGGERS = DOMAIN_CONFIG.get("query_routing", {}).get("sql_agent_triggers", ["inventory", "stock", "how many", "cost", "downtime", "sensor"])