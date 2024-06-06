import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np

# 하이퍼파라미터 설정
epochs = 100
batch_size = 16
learning_rate = 0.001
img_size = 640

# 데이터 로드
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, min_x, min_y, max_x, max_y = map(float, line.strip().split())
                boxes.append([class_id, min_x, min_y, max_x, max_y])
        
        boxes = np.array(boxes)
        
        if self.transform:
            image = self.transform(image)

        return image, boxes

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(img_dir='data/train/images', label_dir='data/train/labels', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', path='weights/yolov7.pt', source='local')

import sys
sys.path.append('yolov7')

from models.experimental import attempt_load

# weights 파일 경로 설정
weights_path = 'weights/yolov7.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(weights_path, map_location=device)

# Modify the final layer for 68 classes
num_classes = 68
detect_layer = model.model[-1]
from models.yolo import Detect

if isinstance(detect_layer, Detect):
    detect_layer.nc = num_classes  # number of classes
    detect_layer.no = detect_layer.na * (num_classes + 5)  # number of outputs per anchor
    detect_layer.m = torch.nn.ModuleList([torch.nn.Conv2d(x.in_channels, detect_layer.no, 1) for x in detect_layer.m])
else:
    raise TypeError("The final layer is not an instance of the Detect class.")

# Freeze all layers except the final detection layer
for param in model.parameters():
    if param.is_leaf:
        param.requires_grad = False

for param in detect_layer.parameters():
    if param.is_leaf:
        param.requires_grad = True

learning_rate = 0.001
# Ensure detect_layer parameters are leaf tensors by re-assigning them
for idx, param in enumerate(detect_layer.parameters()):
    if not param.is_leaf:
        detect_layer.m[idx] = param.detach().clone().requires_grad_(True)

# Re-create the optimizer with only the parameters that require gradients
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


# 학습 루프
for epoch in range(epochs):
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = [target.to(device) for target in targets]  # Assuming targets are lists of tensors

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss (ensure your model has a method to compute the loss)
        loss = compute_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # 모델 저장
    torch.save(model.state_dict(), f"yolov7_epoch_{epoch}.pth")

print("Training finished!")
