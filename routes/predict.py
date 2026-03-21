from fastapi import APIRouter
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import base64
import io
import os
from schemas import ScreeningInput, PredictionOutput
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

router = APIRouter()

model        = joblib.load("model/saved/model.pkl")
feature_cols = joblib.load("model/saved/feature_cols.pkl")
train_data_path = "model/saved/train_data.pkl"

# ── Pick the right SHAP explainer based on model type ─────────────────────
def get_shap_explainer(model):
    import shap
    try:
        from xgboost import XGBClassifier
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            return shap.TreeExplainer(model), "tree"
    except ImportError:
        pass
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, shap.maskers.Independent(
            pd.read_pickle(train_data_path) if os.path.exists(train_data_path)
            else pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
        )), "linear"
    # fallback — works for any model but slower
    train_bg = pd.read_pickle(train_data_path) if os.path.exists(train_data_path) \
               else pd.DataFrame(np.zeros((50, len(feature_cols))), columns=feature_cols)
    return shap.KernelExplainer(model.predict_proba, shap.sample(train_bg, 50)), "kernel"

shap_explainer = None
explainer_type = None

def get_explainer():
    global shap_explainer, explainer_type
    if shap_explainer is None:
        print("🔄 Initializing SHAP explainer...")
        shap_explainer, explainer_type = get_shap_explainer(model)
    return shap_explainer, explainer_type

# ── Helpers ────────────────────────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#060a0f", edgecolor="none", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def get_shap_plot(input_df):
    import shap
    import matplotlib.pylot as plt
    explainer, explainer_type = get_explainer()
    shap_values = shap_explainer(input_df)

    # Extract values for class 1 (ASD) depending on explainer type
    if explainer_type == "tree":
        if len(np.array(shap_values).shape) == 3:
            vals = shap_values[:, :, 1].values[0]
        else:
            vals = shap_values.values[0]
    elif explainer_type == "linear":
        raw = shap_values.values
        # LinearExplainer returns (n, features, classes) or (n, features)
        if len(raw.shape) == 3:
            vals = raw[0, :, 1]
        else:
            vals = raw[0]
    else:
        # KernelExplainer returns list per class
        vals = np.array(shap_values)[1][0] if isinstance(shap_values, list) else shap_values.values[0]

    feats   = list(feature_cols)
    colors  = ["#ef4444" if v > 0 else "#00ffc8" for v in vals]
    sorted_idx = np.argsort(np.abs(vals))[-8:]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#060a0f")
    ax.set_facecolor("#0d1117")
    ax.barh(
        [feats[i] for i in sorted_idx],
        [vals[i]  for i in sorted_idx],
        color=[colors[i] for i in sorted_idx],
        height=0.6,
    )
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.2)
    ax.set_title("SHAP — Feature Impact on Prediction",
                 color="#00ffc8", fontsize=11, pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("SHAP Value  (red = pushes toward ASD, cyan = pushes toward Non-ASD)",
              color="#64748b", fontsize=8)
    return fig_to_base64(fig)

def get_lime_plot(input_df, train_df):
    import lime
    import lime.lime_tabular
    import matplotlib.pyplot as plt
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data        = train_df.values,
        feature_names        = feature_cols,
        class_names          = ["Non-ASD", "ASD"],
        mode                 = "classification",
        discretize_continuous= True,
        random_state         = 42,
    )
    explanation = lime_explainer.explain_instance(
        input_df.values[0],
        model.predict_proba,
        num_features=8,
        top_labels=1,
    )
    label    = list(explanation.as_map().keys())[0]
    exp_list = explanation.as_list(label=label)
    features = [e[0] for e in exp_list]
    values   = [e[1] for e in exp_list]
    colors   = ["#ef4444" if v > 0 else "#a78bfa" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#060a0f")
    ax.set_facecolor("#0d1117")
    ax.barh(features, values, color=colors, height=0.6)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_title("LIME — Local Explanation for This Prediction",
                 color="#a78bfa", fontsize=11, pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("LIME Weight  (red = ASD indicator, purple = Non-ASD indicator)",
              color="#64748b", fontsize=8) 
    return fig_to_base64(fig)

# ── Main predict endpoint ──────────────────────────────────────────────────
@router.post("/predict", response_model=PredictionOutput)
def predict(data: ScreeningInput):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict]).reindex(columns=feature_cols, fill_value=0)

    pred     = model.predict(df)[0]
    proba    = model.predict_proba(df)[0]
    asd_prob = float(proba[1])

    if asd_prob >= 0.7:   risk = "High"
    elif asd_prob >= 0.4: risk = "Medium"
    else:                 risk = "Low"

    top_features = {}
    if hasattr(model, "feature_importances_"):
        importances  = dict(zip(feature_cols, model.feature_importances_))
        top_features = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6]
        )

    # SHAP
    shap_plot = None
    try:
        shap_plot = get_shap_plot(df)
    except Exception as e:
        print(f"⚠ SHAP error: {e}")

    # LIME
    lime_plot = None
    try:
        if os.path.exists(train_data_path):
            train_df  = pd.read_pickle(train_data_path)
            lime_plot = get_lime_plot(df, train_df)
    except Exception as e:
        print(f"⚠ LIME error: {e}")

    return PredictionOutput(
        prediction          = "ASD" if pred == 1 else "Non-ASD",
        asd_probability     = round(asd_prob * 100, 2),
        non_asd_probability = round(float(proba[0]) * 100, 2),
        risk_level          = risk,
        top_features        = top_features,
        shap_plot           = shap_plot,
        lime_plot           = lime_plot,
    )