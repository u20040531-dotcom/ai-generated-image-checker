import streamlit as st
from PIL import Image
from image_detector import ImageDetector
import os

st.set_page_config(page_title="AI vs Human 圖片識別", layout="centered")
st.title("AI vs Human 圖片識別系統")

# ===== 載入模型 =====
MODEL_PATH = "ai_image_detector.pth"

if not os.path.exists(MODEL_PATH):
    st.error("找不到模型檔 ai_image_detector.pth")
    st.stop()

detector = ImageDetector(MODEL_PATH)

# ===== 上傳圖片 =====
uploaded = st.file_uploader(
    "請上傳圖片（jpg / png）",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("開始判斷"):
        with st.spinner("模型推論中..."):
            result = detector.predict(img)

        st.subheader("模型判斷結果")
        st.write(f"📌 **Prediction**：{result['label']}")
        st.write(f"📊 **Confidence**：{result['confidence']:.2%}")
        st.progress(result["confidence"])
