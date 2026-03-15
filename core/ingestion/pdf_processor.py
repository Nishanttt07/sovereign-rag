import fitz  # PyMuPDF
import os
import time
import re
import math
from core.config import RAW_PDFS_DIR

def get_shortest_distance(rect1, rect2):
    x_dist = max(0, rect1.x0 - rect2.x1, rect2.x0 - rect1.x1)
    y_dist = max(0, rect1.y0 - rect2.y1, rect2.y0 - rect1.y1)
    return math.sqrt(x_dist**2 + y_dist**2)

class PDFProcessor:
    def __init__(self):
        self.chunk_size = 600  
        self.overlap = 150     

    def process_text_fast(self, file_path, status_callback=None):
        doc = fitz.open(file_path)
        batch = []
        BATCH_SIZE = 15

        for page_num, page in enumerate(doc):
            text = page.get_text("text").replace('\n', ' ').strip()
            if not text: continue
            
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunk_text = text[i : i + self.chunk_size].strip()
                if len(chunk_text) < 60: continue 
                
                batch.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": str(file_path), 
                        "page": page_num + 1, 
                        "type": "text",
                        "image_caption": "None", 
                        "image_rect": [0.0]*4, 
                        "image_xref": 0, 
                        "image_path": "None"
                    }
                })

            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []
                time.sleep(0.01)

        if batch:
            yield batch

    def _cluster_rectangles(self, rects, distance_threshold=45):
        if not rects: return []
        clusters = []
        working_rects = [fitz.Rect(r) for r in rects]

        while working_rects:
            current = working_rects.pop(0)
            merged_this_round = True
            while merged_this_round:
                merged_this_round = False
                for i in range(len(working_rects) - 1, -1, -1):
                    other = working_rects[i]
                    expanded_current = current + (-distance_threshold, -distance_threshold, distance_threshold, distance_threshold)
                    if expanded_current.intersects(other):
                        current = current | other 
                        working_rects.pop(i)
                        merged_this_round = True
            clusters.append(current)
        return clusters

    def extract_images_for_vision(self, file_path):
        doc = fitz.open(file_path)
        extracted_assets = []

        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("blocks")
            image_list = page.get_images(full=True)
            page_area = page.rect.get_area()
            
            raw_img_rects = []
            
            # 1. Catch Raster Images (Photos, JPEGs)
            for img_info in image_list:
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if rects and rects[0].width > 40 and rects[0].height > 40:
                    raw_img_rects.append(rects[0])
                    
            # 2. Catch Vector Graphics (Physics diagrams, lines, shapes)
            try:
                for drawing in page.get_drawings():
                    r = drawing["rect"]
                    # Ignore tiny dots and giant page borders
                    if r.width > 20 and r.height > 20 and r.get_area() < (page_area * 0.8):
                        raw_img_rects.append(r)
            except Exception as e:
                print(f"⚠️ Vector Drawing Error on page {page_num+1}: {e}")

            # Cluster all lines and photos together into solid diagram blocks
            clustered_rects = self._cluster_rectangles(raw_img_rects, distance_threshold=45)
            if not clustered_rects: continue

            page_captions = []
            for block in text_blocks:
                if block[6] == 1: continue 
                text = block[4].strip().replace('\n', ' ')
                
                match = re.search(r'^(?:Fig|Figure|Table|Diagram|Activity|Map)[\s\-\.:]*\d*|^\d+\.\d+', text, re.IGNORECASE)
                if match:
                    cap_id = match.group(0).strip()
                    page_captions.append({
                        "id": cap_id, "text": text, "rect": fitz.Rect(block[:4])
                    })

            for cap in page_captions:
                best_cluster = None
                min_dist = 120 
                
                for cluster in clustered_rects:
                    dist = get_shortest_distance(cap['rect'], cluster)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster

                if not best_cluster or min_dist > 120:
                    continue

                try:
                    final_crop_rect = best_cluster | cap['rect']
                    
                    if final_crop_rect.get_area() > (page_area * 0.6):
                        continue 

                    # Geometric Text Snapping
                    expanded_cluster = final_crop_rect + (-60, -60, 60, 60)
                    citation_context = ""
                    for block in text_blocks:
                        if block[6] == 0:
                            block_rect = fitz.Rect(block[:4])
                            if expanded_cluster.intersects(block_rect):
                                btext = block[4].strip().replace('\n', ' ')
                                if btext not in cap['text']: 
                                    citation_context += btext + " | "

                    extracted_assets.append({
                        "source": str(file_path), "page": page_num + 1, "xref": 0, 
                        "rect": [final_crop_rect.x0, final_crop_rect.y0, final_crop_rect.x1, final_crop_rect.y1],
                        "caption": cap['text'],
                        "citation_context": citation_context.strip()
                    })
                    
                except Exception as e:
                    print(f"⚠️ Coordinate Math Error: {e}")

        return extracted_assets