import streamlit as st
import requests
import os
from PIL import Image

HF_TOKEN = os.getenv("HF_TOKEN")  # 從環境變數讀

if HF_TOKEN is None:
    raise ValueError("請先設定 HF_TOKEN 環境變數")

MODEL = "Ateeqq/ai-vs-human-image-detector"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}

st.title("AI vs Human 圖片識別")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("跑動測試"):
        uploaded.seek(0)
        img_bytes = uploaded.read()

        with st.spinner("Calling Hugging Face model..."):
            resp = requests.post(API_URL, headers=HEADERS, data=img_bytes, timeout=60)

        if resp.status_code == 200:
            res = resp.json()
            st.subheader("建模原始輸出:")
            st.write(res)

            st.subheader("建模原始輸出:")
            if isinstance(res, list):
                for item in res:
                    st.write(f"{item.get('label', 'N/A')}: {item.get('score', 0):.3f}")
            else:
                st.write(res)
        else:
            st.error(f"API Error: {resp.status_code} {resp.text}")
