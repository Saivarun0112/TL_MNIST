# Transfer Learning on MNIST

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image
import os

# ─────────────────────────────────────────────
# 1. TRANSFORMS
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # ResNet18 expects 224×224
    transforms.Grayscale(num_output_channels=3),  # MNIST is 1-ch; copy to 3
    transforms.ToTensor(),                  # HWC → CHW, divide by 255
    transforms.Normalize(                   # same mean/std for all 3 channels
        mean=[0.1307, 0.1307, 0.1307],
        std=[0.3081, 0.3081, 0.3081]
    )
])

# ─────────────────────────────────────────────
# 2. DATASETS  →  train / val / test split
# ─────────────────────────────────────────────
# MNIST gives us 60 000 train + 10 000 test images.
# We carve a validation set out of the training split:
#   train  : 48 000   (80 % of 60 000)
#   val    : 12 000   (20 % of 60 000)
#   test   : 10 000   (the official held-out test set, untouched until final eval)

full_train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

val_size   = int(0.20 * len(full_train_dataset))  # 12 000
train_size = len(full_train_dataset) - val_size    # 48 000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)    # reproducible split
)

print(f"Dataset sizes  →  train: {len(train_dataset)}, "
      f"val: {len(val_dataset)}, test: {len(test_dataset)}")

from torch.utils.data import Subset
train_dataset = Subset(train_dataset, range(5000))
val_dataset = Subset(val_dataset, range(1000))

# ─────────────────────────────────────────────
# 3. DATA LOADERS
# ─────────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# 4. MODEL  (frozen ResNet18 + new head)
# ─────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all pretrained layers — keeps ImageNet knowledge intact
for param in model.parameters():
    param.requires_grad = False

# Replace the 1000-class head with a 10-class head for MNIST digits
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Only the new head's parameters are passed to the optimiser
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    """Single pass over a DataLoader.  Returns (avg_loss, accuracy_%)."""
    model.train() if is_train else model.eval()
    total_loss, correct = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()

            print(f"  [{'train' if is_train else 'val'}] batch {batch_idx+1}/{len(loader)}", flush=True)

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

# ─────────────────────────────────────────────
# 6. TRAINING LOOP  (train + val each epoch)
# ─────────────────────────────────────────────
EPOCHS = 5
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

print("\nStarting Transfer Learning Training...")
print("-" * 65)

for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
    val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
          f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

# ─────────────────────────────────────────────
# 7. FINAL EVALUATION ON THE TEST SET
#    (run ONCE, only after all training is done)
# ─────────────────────────────────────────────
print("\nEvaluating on the held-out test set...")
test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, is_train=False)
print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")

# ─────────────────────────────────────────────
# SAVE MODEL FOR DEPLOYMENT
# ─────────────────────────────────────────────
torch.save(model.state_dict(), "resnet18_mnist.pth")
print("Model saved as resnet18_mnist.pth")

# ─────────────────────────────────────────────
# 8. CONFUSION MATRIX  (on test set)
# ─────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
disp.plot(cmap='Blues')
plt.title('Transfer Learning (ResNet18) — Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ─────────────────────────────────────────────
# 9. TRAINING CURVES
# ─────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"],   label="Val Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["train_acc"], label="Train Acc")
plt.plot(epochs_range, history["val_acc"],   label="Val Acc")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()

plt.suptitle("Transfer Learning — ResNet18 on MNIST", fontsize=13)
plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# ─────────────────────────────────────────────
# 10. COMPARISON BAR CHART  (swap in your CNN number)
# ─────────────────────────────────────────────
cnn_accuracy = 85.31  # ← replace with your Phase-1 CNN result

plt.figure(figsize=(5, 4))
plt.bar(['Normal CNN', 'Transfer Learning (ResNet18)'],
        [cnn_accuracy, test_acc], color=['steelblue', 'orange'])
plt.title('Final Test Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(95, 100)
plt.tight_layout()
plt.savefig("comparison_chart.png")
plt.show()

print("=" * 45)
print(f"Normal CNN Accuracy       : {cnn_accuracy:.2f}%")
print(f"Transfer Learning Accuracy: {test_acc:.2f}%")
print("=" * 45)

# ─────────────────────────────────────────────
# 11. SINGLE-IMAGE UPLOAD & PREDICTION
#     Drop any handwritten-digit image here to test the model.
# ─────────────────────────────────────────────
def predict_single_image(image_path: str, model, transform, device) -> None:
    """
    Load an image from disk, run it through the trained model,
    and display the image alongside the predicted digit.

    Parameters
    ----------
    image_path : str
        Path to any PNG/JPG image of a handwritten digit.
    model      : trained PyTorch model
    transform  : same transform pipeline used for training
    device     : torch.device
    """
    if not os.path.isfile(image_path):
        print(f"[predict_single_image] File not found: {image_path}")
        return

    # --- Load & preprocess -------------------------------------------------
    img   = Image.open(image_path).convert("L")   # force greyscale (1 channel)
    input_tensor = transform(img).unsqueeze(0).to(device)  # add batch dim → [1,3,224,224]

    # --- Inference ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        output      = model(input_tensor)          # raw logits  [1, 10]
        probs       = torch.softmax(output, dim=1) # probabilities
        pred_class  = probs.argmax(1).item()
        confidence  = probs[0, pred_class].item() * 100

    # --- Display -----------------------------------------------------------
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted digit: {pred_class}  ({confidence:.1f}% confidence)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"→ Predicted digit : {pred_class}")
    print(f"→ Confidence      : {confidence:.2f}%")
    print(f"→ All class probs : {[f'{p*100:.1f}%' for p in probs[0].tolist()]}")


# ── HOW TO USE ──────────────────────────────────────────────────────────────
# After training, call this function with the path to your image:
predict_single_image("digit.jfif", model, transform, device)
#
# The image can be any size; it will be resized to 224×224 automatically.
# For best results, use a white digit on a black background (like MNIST).
# ────────────────────────────────────────────────────────────────────────────