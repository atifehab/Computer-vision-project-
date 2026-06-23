import os
import shutil
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import opendatasets as od

od.download('https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset/data', data_dir="data")


source_dir = "data/binary-cropped-crown-of-thorns-dataset"
output_dir = "data/dataset_splits"  
classes = ["cots_crops", "notcots_crops"]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    
    all_images = [img for img in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, img))]
    
    train_imgs, remaining_imgs = train_test_split(
        all_images, train_size=0.70, random_state=42, shuffle=True
    )
    
    val_imgs, test_imgs = train_test_split(
        remaining_imgs, train_size=0.50, random_state=42, shuffle=True
    )
    
    def copy_images(img_list, split_name):
        for img in img_list:
            src = os.path.join(cls_path, img)
            dst = os.path.join(output_dir, split_name, cls, img)
            shutil.copy(src, dst) 

    copy_images(train_imgs, "train")
    copy_images(val_imgs, "val")
    copy_images(test_imgs, "test")

print(f"Done splitting data into '{output_dir}'")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(root='data/dataset_splits/train', transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root='data/dataset_splits/val', transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(root='data/dataset_splits/test', transform=data_transforms['val']) 

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Number of test images: {len(test_dataset)}")
print("Classes identified:", train_dataset.class_to_idx)

weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. دالة التدريب
def train_model(model, criterion, optimizer, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Starting Training...")
    print("-" * 30)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('=' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
                dataloader = train_loader
                dataset_size = len(train_dataset)
            else:
                model.eval()   
                dataloader = val_loader
                dataset_size = len(val_dataset)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model

trained_model = train_model(model, criterion, optimizer, num_epochs=20)

trained_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = trained_model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(f"Test Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
print(f"Precision     : {precision_score(all_labels, all_preds):.4f}")
print(f"Recall        : {recall_score(all_labels, all_preds):.4f}")
print(f"F1-Score      : {f1_score(all_labels, all_preds):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

os.makedirs("models", exist_ok=True)
torch.save(trained_model.state_dict(), "models/cots_model.pth")
print("Model saved successfully at 'models/cots_model.pth'")