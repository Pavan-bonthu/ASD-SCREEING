from fastapi import APIRouter
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io
import os

from schemas import ScreeningInput, PredictionOutput
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

router = APIRouter()

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "saved", "model.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "saved", "feature_cols.pkl"))
train_data_path = os.path.join(BASE_DIR, "saved", "train_data.pkl")

# ─────────────────────────────────────────────
# EXPLAINER CACHE
# ─────────────────────────────────────────────
_shap_explainer = None
_explainer_type = None
_train_df_cache = None


def get_train_df():
    global _train_df_cache

    if _train_df_cache is None:
        if os.path.exists(train_data_path):
            try:
                _train_df_cache = pd.read_pickle(train_data_path)
            except Exception as e:
                print(f"⚠ Could not load train_data.pkl: {e}")
        else:
            _train_df_cache = None

    if _train_df_cache is None:
        _train_df_cache = pd.DataFrame(
            np.zeros((50, len(feature_cols))),
            columns=feature_cols
        )

    return _train_df_cache


def get_explainer():
    global _shap_explainer, _explainer_type

    if _shap_explainer is not None:
        return _shap_explainer, _explainer_type

    try:
        from xgboost import XGBClassifier
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            _shap_explainer = shap.TreeExplainer(model)
            _explainer_type = "tree"
            return _shap_explainer, _explainer_type
    except Exception:
        pass

    if isinstance(model, LogisticRegression):
        try:
            train_df = get_train_df()
            masker = shap.maskers.Independent(train_df)
            _shap_explainer = shap.LinearExplainer(model, masker)
            _explainer_type = "linear"
            return _shap_explainer, _explainer_type
        except Exception as e:
            print(f"⚠ LinearExplainer failed: {e}")

    try:
        train_df = get_train_df()
        _shap_explainer = shap.KernelExplainer(
            model.predict_proba, shap.sample(train_df, 50)
        )
        _explainer_type = "kernel"
    except Exception as e:
        print(f"⚠ KernelExplainer failed: {e}")
        _shap_explainer = None
        _explainer_type = None

    return _shap_explainer, _explainer_type


# ─────────────────────────────────────────────
# FIG → BASE64
# ─────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#060a0f", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────
def get_shap_plot(input_df):
    explainer, explainer_type = get_explainer()
    if explainer is None:
        return None

    shap_values = explainer(input_df)

    try:
        shap_values = explainer(input_df)
        if explainer_type == "tree":
            vals = shap_values[:, :, 1].values[0] if len(np.array(shap_values).shape) == 3 else shap_values.values[0]
        elif explainer_type == "linear":
            raw = shap_values.values
            vals = raw[0, :, 1] if len(raw.shape) == 3 else raw[0]
        else:
            vals = np.array(shap_values)[1][0] if isinstance(shap_values, list) else shap_values.values[0]
    except Exception:
        return None

    feats = list(feature_cols)
    sorted_idx = np.argsort(np.abs(vals))[-8:]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([feats[i] for i in sorted_idx],
            [float(vals[i]) for i in sorted_idx])

    ax.axvline(0)
    ax.set_title("SHAP — Feature Impact")

    return fig_to_base64(fig)


# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────
def get_lime_plot(input_df):
    try:
        
        train_df = get_train_df()

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_df.values,
            feature_names=feature_cols,
            class_names=["Non-ASD", "ASD"],
            mode="classification"
        )

        explanation = explainer.explain_instance(
            input_df.values[0],
            model.predict_proba,
            num_features=8
        )

        fig = explanation.as_pyplot_figure()
        return fig_to_base64(fig)

    except Exception as e:
        print("⚠ LIME error:", e)
        return None


# ─────────────────────────────────────────────
# 🚀 MAIN API
# ─────────────────────────────────────────────
@router.post("/predict", response_model=PredictionOutput)
def predict(data: ScreeningInput):

    input_dict = data.dict()

    # YES / NO → numeric
    for k, v in input_dict.items():
        if isinstance(v, str):
            if v.lower() == "yes":
                input_dict[k] = 1
            elif v.lower() == "no":
                input_dict[k] = 0

    # Feature engineering
    behavior_score = sum(input_dict.get(f"A{i}", 0) for i in range(1, 11))
    doctor_score = sum(input_dict.get(f"D{i}", 0) for i in range(1, 11))

    input_dict["behavior_score"] = behavior_score
    input_dict["doctor_score"] = doctor_score

    df = pd.DataFrame([input_dict]).reindex(columns=feature_cols, fill_value=0)

    pred = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    proba = [float(x) for x in proba]

    asd_prob = proba[1]
    non_asd_prob = proba[0]

    # Risk
    if asd_prob >= 0.65:
        risk = "High"
        risk_level_num = 3
    elif asd_prob >= 0.30:
        risk = "Medium"
        risk_level_num = 2
    else:
        risk = "Low"
        risk_level_num = 1

    # Feature importance
    top_features = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_cols, model.feature_importances_))
        top_features = {
            str(k): float(v)
            for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6]
        }

    # SHAP / LIME
    shap_plot = get_shap_plot(df)
    lime_plot = get_lime_plot(df)
    print("DEBUG TYPES:")
    print(type(pred))
    print(type(asd_prob))
    print(type(non_asd_prob))

    return PredictionOutput(
        prediction="ASD" if pred == 1 else "Non-ASD",
        asd_probability=float(round(asd_prob * 100, 2)),
        non_asd_probability=float(round(non_asd_prob * 100, 2)),
        risk_level=str(risk),
        risk_level_num=int(risk_level_num) if 'risk_level_num' in locals() else 1,
        top_features=top_features,
        shap_plot=str(shap_plot) if shap_plot else None,
        lime_plot=str(lime_plot) if lime_plot else None,
        
    
    )
    
   