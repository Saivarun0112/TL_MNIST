import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import numpy as np
import io
import os

app = FastAPI(title="MNIST Digit Classifier")

FRONTEND_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend",
    "index.html"
)

@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to model saved by the notebook
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ML", "resnet18_mnist.pth")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"WARNING: {MODEL_PATH} not found. Run the notebook first.")

model.to(device)
model.eval()

# FIX 1: Use ImageNet normalization stats (correct for ResNet18)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale → 3-channel for ResNet
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean — correct for ResNet18
        std=[0.229, 0.224, 0.225]     # ImageNet std  — correct for ResNet18
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert to grayscale
    img = Image.open(io.BytesIO(contents)).convert("L")

    # FIX 2: Auto-invert if image has a light background (black digit on white paper).
    # MNIST trains on WHITE digit on BLACK background, so we must match that convention.
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(1).item()
        confidence = probs[0, pred_class].item() * 100

    return {
        "predicted_digit": pred_class,
        "confidence": round(confidence, 2),
        "all_probs": [round(p * 100, 2) for p in probs[0].tolist()]
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": os.path.exists(MODEL_PATH)
    }
