import fitz  # PyMuPDF
import os
import time
from core.config import RAW_PDFS_DIR

class PDFProcessor:
    def __init__(self):
        """Initializes with Robust Spatial Indexing."""
        pass

    def process_pdf_stream(self, file_path, status_callback=None):
        doc = fitz.open(file_path)
        batch = []
        BATCH_SIZE = 5

        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("blocks")
            image_list = page.get_images(full=True)
            page_assets = []
            
            for i, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    if not rects: continue
                    img_rect = rects[0]
                    
                    if len(doc.extract_image(xref)["image"]) < 10240: continue 
                    
                    # --- ROBUST CAPTION LINKING ---
                    best_caption = "None"
                    min_v_dist = 1000
                    
                    for block in text_blocks:
                        bx0, by0, bx1, by1, btext = block[:5]
                        clean_text = btext.strip().replace('\n', ' ')
                        
                        # 1. Vertical Check: Text must be BELOW image
                        v_dist = by0 - img_rect.y1 
                        
                        # 2. Horizontal Overlap Check (The Fix)
                        # Instead of strict centering, we check if they overlap horizontally
                        overlap_width = max(0, min(img_rect.x1, bx1) - max(img_rect.x0, bx0))
                        text_width = bx1 - bx0
                        
                        # Calculate overlap ratio (Avoid division by zero)
                        overlap_ratio = overlap_width / text_width if text_width > 0 else 0
                        
                        # Rules:
                        # - Must be within 50px vertically (tight association)
                        # - Must overlap horizontally by at least 50% OR be contained within image width
                        if 0 < v_dist < 50 and overlap_ratio > 0.5:
                            # Prefer the CLOSEST text block
                            if v_dist < min_v_dist:
                                min_v_dist = v_dist
                                best_caption = clean_text

                    # Orphan Cull: Only index if we found a caption
                    if best_caption == "None":
                        continue

                    page_assets.append({
                        "xref": xref,
                        "rect": [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1],
                        "caption": best_caption
                    })
                    print(f"✅ [INDEXED] '{best_caption}' -> Image {xref} (Page {page_num+1})")
                
                except Exception as e:
                    print(f"⚠️ Image Error: {e}")

            # Text Chunks
            raw_text = page.get_text().strip()
            if raw_text:
                batch.append({
                    "text": raw_text,
                    "metadata": {
                        "source": str(file_path), "page": page_num + 1, "type": "text",
                        "image_caption": "None", "image_rect": [0.0]*4, "image_xref": 0, "image_path": "None"
                    }
                })

            # Image Chunks
            for asset in page_assets:
                batch.append({
                    "text": f"DIAGRAM: {asset['caption']} (Ref: Page {page_num+1})",
                    "metadata": {
                        "source": str(file_path), "page": page_num + 1, "type": "image_index",
                        "image_caption": asset['caption'],
                        "image_rect": asset['rect'],
                        "image_xref": asset['xref'],
                        "image_path": "None"
                    }
                })

            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []
                time.sleep(0.01)

        if batch:
            yield batch