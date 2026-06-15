import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

app = FastAPI(title="MNIST Digit Classifier")

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1307, 0.1307, 0.1307],
        std=[0.3081, 0.3081, 0.3081]
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L")

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