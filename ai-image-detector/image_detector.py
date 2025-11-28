import requests
import sys

HF_TOKEN = "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"
MODEL = "ai-vs-human-image-detector"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}

def predict_image(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
        resp = requests.post(API_URL, headers=HEADERS, data=img_bytes, timeout=60)
    return resp.json()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hf_image_detector.py image.jpg")
        sys.exit(1)

    result = predict_image(sys.argv[1])
    print("Model Output:")
    print(result)
