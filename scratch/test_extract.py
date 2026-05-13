import fitz
import re
from pathlib import Path

pdf_path = "data/raw_pdfs/Aeronautical Engineer's Data Book.pdf"
doc = fitz.open(pdf_path)

# Let's check page 27 (index 26) which contains Fig 2.7
page = doc[26]

print("--- TESTING GET_IMAGES ---")
print(page.get_images(full=True))

print("\n--- TESTING GET_DRAWINGS ---")
paths = page.get_drawings()
for p in paths:
    rect = p["rect"]
    if rect.width > 20 and rect.height > 20 and rect.width < page.rect.width * 0.9 and rect.height < page.rect.height * 0.9:
        print(f"Valid drawing rect: {rect}")

print("\n--- TESTING CAPTIONS WITH DICT ---")
d = page.get_text("dict")
regex_pattern = r'\b(?:Fig|Figure|Table|Diagram)\b[\s\.:\-_]*\d+[\.\da-zA-Z\(\)]*'

page_captions = []
for block in d.get("blocks", []):
    if block.get("type") == 0:  # text block
        for line in block.get("lines", []):
            text = " ".join(span.get("text", "") for span in line.get("spans", [])).strip()
            # print("Line text:", text)
            match = re.search(regex_pattern, text, re.IGNORECASE)
            if match:
                print(f"FOUND CAPTION: {match.group(0)} -> {text}")
                page_captions.append({"text": text, "rect": fitz.Rect(line["bbox"])})
