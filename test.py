import streamlit as st
import fitz
from PIL import Image
import io
import base64
import requests

st.title("PDF Image Explainer")

API_URL = "https://api.moondream.ai/v1/query"
API_KEY = "YOUR_API_KEY"

def extract_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base["image"]))
            images.append(image)

    return images


def explain_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    payload = {
        "image": img_b64,
        "question": "Explain this image."
    }

    headers = {"Authorization": f"Bearer {API_KEY}"}

    r = requests.post(API_URL, json=payload, headers=headers)

    return r.json()["answer"]


uploaded = st.file_uploader("Upload PDF")

if uploaded:

    images = extract_images(uploaded.read())

    if "index" not in st.session_state:
        st.session_state.index = 0

    idx = st.session_state.index

    st.image(images[idx])

    if st.button("Explain"):
        st.write(explain_image(images[idx]))

    col1, col2 = st.columns(2)

    if col1.button("Previous") and idx > 0:
        st.session_state.index -= 1
        st.rerun()

    if col2.button("Next") and idx < len(images)-1:
        st.session_state.index += 1
        st.rerun()