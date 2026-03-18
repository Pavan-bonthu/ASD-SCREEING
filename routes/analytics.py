from fastapi import APIRouter
import joblib

router = APIRouter()

@router.get("/analytics/model-info")
def model_info():
    accuracy = joblib.load("model/saved/accuracy.pkl")
    return {
        "accuracy": round(float(accuracy) * 100, 2),
        "models_tested": ["Logistic Regression", "Random Forest", "XGBoost"],
        "best_model": "XGBoost",
        "dataset": "UCI Autism Screening Dataset",
        "features_used": 16,
    }