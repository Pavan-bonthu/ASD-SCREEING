from fastapi import APIRouter
import joblib
import pandas as pd
import numpy as np
from schemas import ScreeningInput, PredictionOutput

router = APIRouter()

model        = joblib.load("model/saved/model.pkl")
feature_cols = joblib.load("model/saved/feature_cols.pkl")

@router.post("/predict", response_model=PredictionOutput)
def predict(data: ScreeningInput):
    input_dict = data.dict()
    print("📥 Received payload:", input_dict)        # debug line
    print("📋 Expected features:", feature_cols)     # debug line

    df = pd.DataFrame([input_dict]).reindex(columns=feature_cols, fill_value=0)

    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    asd_prob = float(proba[1])

    if asd_prob >= 0.7:
        risk = "High"
    elif asd_prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    top_features = {}
    if hasattr(model, "feature_importances_"):
        importances  = dict(zip(feature_cols, model.feature_importances_))
        top_features = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6]
        )

    return PredictionOutput(
        prediction          = "ASD" if pred == 1 else "Non-ASD",
        asd_probability     = round(asd_prob * 100, 2),
        non_asd_probability = round(float(proba[0]) * 100, 2),
        risk_level          = risk,
        top_features        = top_features,
    )