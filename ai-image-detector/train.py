# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# ===============================
# 1. 資料預處理
# ===============================
data_dir = 'dataset'

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# ===============================
# 2. 建立模型 (簡單 CNN)
# ===============================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # AI / Real

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 3. 訓練模型
# ===============================
epochs = 5  # 示範用，正式可改 10~20
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ===============================
# 4. 儲存模型
# ===============================
torch.save(model.state_dict(), "ai_image_detector.pth")
print("訓練完成，ai_image_detector.pth 已產生！")

