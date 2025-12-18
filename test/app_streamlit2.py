import streamlit as st
from PIL import Image
from image_detector import ImageDetector
import sys
st.title("AI vs Human 圖片識別系統")

detector = ImageDetector("ai_image_detector.pth")

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(sys.argv[1]).convert("RGB")
    st.image(img, use_column_width=True)

    if st.button("跑動測試"):
        prob = detector.predict(img)
        print(prob)
        labels = ["AI Generated", "Real Image"]
        result = labels[prob.argmax()]

        st.subheader("模型判斷結果")
        st.write(f"Prediction: **{result}**")
        st.write(f"Confidence: {prob.max().item():.3f}")

