import streamlit as st
import requests
from PIL import Image

HF_TOKEN = "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"
MODEL = "dima806/ai_vs_real_image_detection"   # ← 改成完整模型路徑
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}

st.title("AI vs Human Image Detector — Streamlit Demo")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        img_bytes = uploaded.read()

        with st.spinner("Calling Hugging Face model..."):
            resp = requests.post(API_URL, headers=HEADERS, data=img_bytes, timeout=60)

        if resp.status_code == 200:
            res = resp.json()
            st.subheader("Model Raw Output:")
            st.write(res)

            st.subheader("Parsed Output:")
            if isinstance(res, list):
                for item in res:
                    label = item.get("label", "N/A")
                    score = item.get("score", 0)
                    st.write(f"{label}: {score:.3f}")
            else:
                st.write(res)
        else:
            st.error(f"API Error: {resp.status_code} {resp.text}")
