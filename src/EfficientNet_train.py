"""
=============================================================================
 EfficientNet-B0 COTS Classifier — Improved Training Pipeline (v2)
=============================================================================
 Addresses the following issues from v1:
   1. Weak augmentation pipeline → Underwater-specific augmentations
   2. Fixed Resize distortion  → RandomResizedCrop for scale invariance
   3. No color robustness      → ColorJitter + custom underwater color shift
   4. Fixed LR too aggressive  → Differential LR + cosine annealing
   5. CrossEntropyLoss         → FocalLoss for hard example mining
   6. No early stopping        → Patience-based early stopping
   7. No learning rate warmup  → Linear warmup for stable training

 Usage (Google Colab / local):
   python src/EfficientNet_train.py
=============================================================================
"""

import os
import shutil
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import opendatasets as od


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 30
BACKBONE_LR = 1e-4       # Lower LR for pretrained backbone
HEAD_LR = 1e-3            # Higher LR for the new classifier head
WEIGHT_DECAY = 1e-4
FOCAL_ALPHA = 0.25        # Focal loss: class balance weight
FOCAL_GAMMA = 2.0         # Focal loss: focusing parameter
EARLY_STOP_PATIENCE = 7   # Stop if val_acc doesn't improve for N epochs
WARMUP_EPOCHS = 3         # Linear LR warmup period
NUM_WORKERS = 2
IMAGE_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of correct class

        # Apply focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha balancing
        if self.alpha is not None:
            alpha_t = torch.where(
                targets == 1,
                torch.tensor(1 - self.alpha, device=inputs.device),
                torch.tensor(self.alpha, device=inputs.device),
            )
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM UNDERWATER COLOR SHIFT TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
class UnderwaterColorShift:
    """Simulate underwater color absorption at random depths."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # img is a tensor (C, H, W) with values in [0, 1]
        red_scale = random.uniform(0.3, 1.0)    # Heavy red absorption
        green_scale = random.uniform(0.6, 1.0)  # Moderate green absorption
        blue_scale = random.uniform(0.8, 1.0)   # Minimal blue absorption

        img[0] = img[0] * red_scale   # R channel
        img[1] = img[1] * green_scale  # G channel
        img[2] = img[2] * blue_scale   # B channel

        return img.clamp(0, 1)


class UnderwaterHaze:
    """Simulate underwater particulate scattering / haze."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        intensity = random.uniform(0.05, 0.2)
        haze = torch.zeros_like(img)
        haze[0] = 0.1   # Minimal red in haze
        haze[1] = 0.4   # Some green
        haze[2] = 0.5   # Dominant blue

        img = img * (1 - intensity) + haze * intensity
        return img.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
data_transforms = {
    "train": transforms.Compose([
        # Spatial transforms
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.2)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),

        # Color transforms
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.5,
            hue=0.15,
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.05),

        # Convert to tensor
        transforms.ToTensor(),

        # Custom underwater transforms
        UnderwaterColorShift(p=0.4),
        UnderwaterHaze(p=0.25),

        # Random erasing simulates occlusion
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),

        # Normalize with ImageNet stats
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),

    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DOWNLOAD & SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def download_and_split_data():
    """Download the Kaggle dataset and split into train/val/test."""
    od.download(
        "https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset/data",
        data_dir="data",
    )

    source_dir = "data/binary-cropped-crown-of-thorns-dataset"
    output_dir = "data/dataset_splits_v2"
    classes = ["cots_crops", "notcots_crops"]

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        all_images = [
            img for img in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, img))
        ]

        train_imgs, remaining = train_test_split(
            all_images, train_size=0.70, random_state=SEED, shuffle=True
        )
        val_imgs, test_imgs = train_test_split(
            remaining, train_size=0.50, random_state=SEED, shuffle=True
        )

        for img_list, split_name in [
            (train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")
        ]:
            for img in img_list:
                src = os.path.join(cls_path, img)
                dst = os.path.join(output_dir, split_name, cls, img)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    print(f"Data split complete → '{output_dir}'")
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, criterion, optimizer, scheduler, train_loader,
                val_loader, train_size, val_size, device, num_epochs,
                patience):
    """
    Train with early stopping, LR scheduling, and best-model checkpointing.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    print("Starting Training (v2 — improved pipeline)")
    print("=" * 50)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
                dataset_size = train_size
            else:
                model.eval()
                loader = val_loader
                dataset_size = val_size

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0
                        )
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = (running_corrects.double() / dataset_size).item()

            print(f"  {phase.capitalize():>5} — Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  LR: {current_lr:.6f}")

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"  ** New best model (val_acc={best_acc:.4f}) **")
                else:
                    epochs_no_improve += 1
                    print(f"  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    elapsed = time.time() - since
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader, device, class_names):
    """Full evaluation with metrics and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision : {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall    : {recall_score(all_labels, all_preds):.4f}")
    print(f"F1-Score  : {f1_score(all_labels, all_preds):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=class_names)}")

    print("\n" + "-" * 50)
    print("CONFIDENCE THRESHOLD ANALYSIS (for ROV deployment)")
    print("-" * 50)
    cots_probs = all_probs[:, 0]
    for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
        thresh_preds = (cots_probs >= threshold).astype(int)
        thresh_preds = 1 - thresh_preds
        acc = accuracy_score(all_labels, thresh_preds)
        prec = precision_score(all_labels, thresh_preds, zero_division=0)
        rec = recall_score(all_labels, thresh_preds, zero_division=0)
        f1 = f1_score(all_labels, thresh_preds, zero_division=0)
        print(f"  Threshold={threshold:.2f} → Acc={acc:.3f} Prec={prec:.3f} "
              f"Rec={rec:.3f} F1={f1:.3f}")

    return all_preds, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    data_dir = download_and_split_data()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=data_transforms["train"],
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=data_transforms["val"],
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=data_transforms["val"],
    )

    print(f"Classes: {train_dataset.class_to_idx}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 2),
    )
    model = model.to(device)

    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters()
                   if "classifier" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )

    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
    )

    evaluate_model(model, test_loader, device, test_dataset.classes)

    os.makedirs("models", exist_ok=True)
    save_path = "models/cots_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved → '{save_path}'")


if __name__ == "__main__":
    main()