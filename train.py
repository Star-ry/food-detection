import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import wandb
from PIL import Image
import yaml

# Initialize wandb
wandb.init(project="ssdmobilenetv2_training")

# Transforms for image augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomResizedCrop(112),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.CenterCrop(112),
    transforms.ToTensor()
])

# Custom dataset class to read images and labels
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Read the data.yaml file
        data_yaml_path = os.path.join(root_dir, 'data.yaml')
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        self.classes = data_config['names']
        self.num_classes = data_config['nc']

        # Load image file paths and corresponding labels
        for subdir in ['train', 'valid']:
            subdir_path = os.path.join(root_dir, subdir, 'images')
            label_path = os.path.join(root_dir, subdir, 'labels')

            for img_file in os.listdir(subdir_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(subdir_path, img_file)
                    self.image_files.append(img_path)

                    # Read the corresponding label file
                    label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                    with open(os.path.join(label_path, label_file), 'r') as lf:
                        labels = [line.strip().split() for line in lf.readlines()]
                        self.labels.append(labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        labels = self.labels[idx]
        bboxes = []
        class_labels = []

        for label in labels:
            class_id = int(label[0])
            class_labels.append(class_id)
            x_center, y_center, width, height = map(float, label[1:])
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            bboxes.append([xmin, ymin, xmax, ymax])

        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(class_labels, dtype=torch.int64)
        }

        return image, target

# Create datasets
root_dir = "data"
train_dataset = CustomDataset(root_dir=root_dir, transform=train_transform)
val_dataset = CustomDataset(root_dir=root_dir, transform=val_transform)

# Check the number of classes
print(f"Number of classes: {train_dataset.num_classes}")

# Assign device CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the ssd_mobilenet_v2 model
from torchvision.models.detection import ssd300_vgg16

model = ssd300_vgg16(pretrained=True)
num_classes = train_dataset.num_classes

# Modify the classifier to match the number of classes
model.head.classification_head.num_classes = num_classes
model = model.to(device)

import torch
import sklearn
from sklearn import metrics
import datetime
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import numpy as np

def val_model(model, testset_loader):
    model.eval()
    val_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in testset_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            output = model(images, targets)

            # Calculating loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    val_loss /= len(testset_loader)
    return val_loss

def train_model(model, trainset_loader, valset_loader, optimizer, num_epoch, model_save_path):
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for images, targets in trainset_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            # Backward pass
            losses.backward()
            optimizer.step()

        train_loss /= len(trainset_loader)
        val_loss = val_model(model, valset_loader)

        train_loss_per_epoch.append(train_loss)
        val_loss_per_epoch.append(val_loss)

        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Epoch": epoch + 1
        })

        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), model_save_path)
    return train_loss_per_epoch, val_loss_per_epoch

# Hyperparameters
lr = 0.001
num_epoch = 50
batch_size = 8

# Optimizer, loss, scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Data loader
trainset_loader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=lambda x: tuple(zip(*x)))
valset_loader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=lambda x: tuple(zip(*x)))

# Perform training and validation
save_path = "model_ssd_mobilenet_v2.pth"

train_loss, val_loss = train_model(model, trainset_loader, valset_loader, optimizer, num_epoch, save_path)

with open('train_loss_ssd_mobilenetv2.json', 'w') as jf:
    json.dump(train_loss, jf)

with open('val_loss_ssd_mobilenetv2.json', 'w') as jf:
    json.dump(val_loss, jf)
