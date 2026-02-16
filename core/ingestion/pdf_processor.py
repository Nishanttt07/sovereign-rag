import fitz  # PyMuPDF
import os
import time
from core.config import RAW_PDFS_DIR

class PDFProcessor:
    def __init__(self):
        pass

    def process_pdf_stream(self, file_path, status_callback=None):
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        
        image_folder = RAW_PDFS_DIR / "extracted_images"
        os.makedirs(image_folder, exist_ok=True)
        
        batch = []
        BATCH_SIZE = 5

        for page_num, page in enumerate(doc):
            # 1. Get Text Blocks with Coordinates
            # format: (x0, y0, x1, y1, "text", block_no, block_type)
            text_blocks = page.get_text("blocks")
            
            # 2. Get Images with Coordinates
            image_list = page.get_images(full=True)
            page_images = []
            
            for i, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    # Get image position on page (rect)
                    # We have to search for the image usage on the page to get rect
                    rects = page.get_image_rects(xref)
                    if not rects: continue
                    img_rect = rects[0]  # Use first occurrence
                    
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    
                    if len(image_bytes) < 5120: continue # Skip icons
                    
                    img_name = f"{filename}_p{page_num}_img{i}.{ext}"
                    saved_path = image_folder / img_name
                    
                    with open(saved_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Store image data + location
                    page_images.append({
                        "path": str(saved_path),
                        "rect": img_rect, # (x0, y0, x1, y1)
                        "caption": "Unknown Diagram"
                    })
                    
                except Exception:
                    pass

            # 3. SPATIAL CAPTIONING (The Fix)
            # Find text immediately below the image
            for img in page_images:
                img_x0, img_y0, img_x1, img_y1 = img["rect"]
                
                best_caption = "None"
                min_dist = 1000
                
                for block in text_blocks:
                    bx0, by0, bx1, by1, btext, _, _ = block
                    
                    # Check if text is BELOW image (by0 > img_y1)
                    # And within reasonable distance (e.g., 50 pixels)
                    vertical_dist = by0 - img_y1
                    
                    # Check horizontal overlap (is text roughly centered under image?)
                    # (Simple check: text shouldn't be way off to the left/right)
                    horizontal_overlap = max(0, min(img_x1, bx1) - max(img_x0, bx0))
                    
                    if 0 < vertical_dist < 60 and horizontal_overlap > 0:
                        if vertical_dist < min_dist:
                            min_dist = vertical_dist
                            # Clean text
                            clean_text = btext.strip().replace('\n', ' ')
                            if len(clean_text) > 5: # Ignore page numbers
                                best_caption = clean_text
                
                img["caption"] = best_caption
                if best_caption != "None":
                    print(f"✅ Linked Caption: '{best_caption}' to Image on Page {page_num}")

            # 4. CREATE CHUNKS
            
            # A. Main Text Chunk
            text_content = page.get_text()
            batch.append({
                "text": text_content,
                "metadata": {
                    "source": filename, 
                    "page": page_num + 1,
                    "image_path": page_images[0]["path"] if page_images else "None",
                    "image_caption": "See Image Index",
                    "type": "text"
                }
            })

            # B. Image Index Chunks
            for img in page_images:
                # We inject the Spatial Caption into the searchable text
                # This guarantees retrieval when you search "Salivary Glands"
                desc = f"DIAGRAM/FIGURE: {img['caption']}. Found on page {page_num+1}."
                
                batch.append({
                    "text": desc, 
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "image_path": img["path"],
                        "image_caption": img["caption"],
                        "type": "image_index"
                    }
                })

            if status_callback:
                status_callback(f"Pg {page_num+1}: Processed {len(page_images)} images.")

            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []
                time.sleep(0.05)

        if batch:
            yield batch