from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import io
import json
import numpy as np
from pydantic import BaseModel
router = APIRouter()

# ── Load model ──────────────────────────────────────────────────────────────
with open("model/saved/image_model_meta.json") as f:
    meta = json.load(f)

CLASSES  = meta["classes"]
IMG_SIZE = meta["img_size"]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

image_model = load_image_model()
print(f"✅ Image model loaded | Classes: {CLASSES} | Device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
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
        outputs = image_model(tensor)
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
        "success":          True,
        "prediction":       prediction,
        "asd_probability":  round(asd_prob    * 100, 2),
        "nonasd_probability": round(nonasd_prob * 100, 2),
        "risk_level":       risk,
        "model_accuracy":   meta["accuracy"],
    }
    
   

class FusionInput(BaseModel):
    tabular_asd_prob: float   # from /api/predict
    image_asd_prob:   float   # from /api/predict-image
    tabular_weight:   float = 0.6
    image_weight:     float = 0.4

@router.post("/predict-combined")
def predict_combined(data: FusionInput):
    # Weighted average fusion
    fused_prob = (
        data.tabular_asd_prob * data.tabular_weight +
        data.image_asd_prob   * data.image_weight
    )

    prediction = "ASD" if fused_prob >= 50 else "Non-ASD"

    if fused_prob >= 70:   risk = "High"
    elif fused_prob >= 40: risk = "Medium"
    else:                  risk = "Low"

    return {
        "prediction":      prediction,
        "fused_probability": round(fused_prob, 2),
        "risk_level":       risk,
        "breakdown": {
            "tabular_contribution": round(data.tabular_asd_prob * data.tabular_weight, 2),
            "image_contribution":   round(data.image_asd_prob   * data.image_weight,   2),
            "tabular_weight":       data.tabular_weight,
            "image_weight":         data.image_weight,
        }
    }