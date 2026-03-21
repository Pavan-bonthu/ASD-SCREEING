from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import io
import json

router = APIRouter()

# ── Load metadata ──────────────────────────────────────────────
with open("model/saved/image_model_meta.json") as f:
    meta = json.load(f)

CLASSES  = meta["classes"]
IMG_SIZE = meta["img_size"]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ❗ Lazy model loading
image_model = None

def load_image_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )
    m.load_state_dict(
        torch.load("model/saved/image_model_best.pth", map_location=DEVICE)
    )
    m.eval()
    return m.to(DEVICE)

def get_model():
    global image_model
    if image_model is None:
        print("🔄 Loading image model...")
        image_model = load_image_model()
        print("✅ Image model loaded")
    return image_model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    model = get_model()   # ✅ lazy load here

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error":   "Invalid image file."
        })

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    asd_idx     = CLASSES.index("Autistic")
    nonasd_idx  = CLASSES.index("Non_Autistic")

    asd_prob    = float(probs[asd_idx])
    nonasd_prob = float(probs[nonasd_idx])

    prediction  = "ASD" if asd_prob > nonasd_prob else "Non-ASD"

    if asd_prob >= 0.7:   risk = "High"
    elif asd_prob >= 0.4: risk = "Medium"
    else:                 risk = "Low"

    return {
        "success": True,
        "prediction": prediction,
        "asd_probability": round(asd_prob * 100, 2),
        "nonasd_probability": round(nonasd_prob * 100, 2),
        "risk_level": risk,
        "model_accuracy": meta["accuracy"],
    }