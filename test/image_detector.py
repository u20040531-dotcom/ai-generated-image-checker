import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageDetector:
    def __init__(self, model_path):
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, 2)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        self.model.eval()

        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
        ])

    def predict(self, image: Image.Image):
        img = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            out = self.model(img)
            prob = torch.softmax(out, dim=1)[0]
        return prob

