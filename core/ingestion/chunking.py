from core.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping chunks.
    Returns a list of strings.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward, but backstep by 'overlap' to keep context
        start += (chunk_size - overlap)
    
    return chunks