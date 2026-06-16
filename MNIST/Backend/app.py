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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ML", "resnet18_mnist.pth")

# FIX 1: Must use pretrained ImageNet weights first (matches training),
# then replace the head — exactly like the training notebook did.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)

# FIX 2: Load your saved state_dict (only the fc head was trained & saved)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"WARNING: {MODEL_PATH} not found. Run the notebook first.")

model.to(device)
model.eval()

# FIX 3: Use the SAME transform as training — MNIST mean/std, same order
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1307, 0.1307, 0.1307],
        std=[0.3081, 0.3081, 0.3081]
    )
])


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Normalize uploaded image to MNIST format:
    white digit on black background, tightly cropped.
    """
    img = img.convert("L")
    arr = np.array(img)

    # Auto-invert: if background is light (black digit on white paper), invert it
    if arr.mean() > 127:
        arr = 255 - arr

    # Crop to bounding box of digit with padding
    binary = arr > 50
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 20
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1], cmax + pad)
        arr = arr[rmin:rmax, cmin:cmax]

    # Pad to square
    h, w = arr.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = arr

    return Image.fromarray(square)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = preprocess_image(img)

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
