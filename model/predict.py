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

model        = joblib.load("model/saved/model.pkl")
feature_cols = joblib.load("model/saved/feature_cols.pkl")
train_data_path = "model/saved/train_data.pkl"

# ── Lazy load — explainer created on first request, not at startup ─────────
_shap_explainer  = None
_explainer_type  = None
_train_df_cache  = None

def get_train_df():
    global _train_df_cache
    if _train_df_cache is None and os.path.exists(train_data_path):
        try:
            _train_df_cache = pd.read_pickle(train_data_path)
        except Exception as e:
            print(f"⚠ Could not load train_data.pkl: {e}")
            # fallback — create dummy background from zeros
            _train_df_cache = pd.DataFrame(
                np.zeros((50, len(feature_cols))),
                columns=feature_cols
            )
    elif _train_df_cache is None:
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
            print("✅ SHAP TreeExplainer ready")
            return _shap_explainer, _explainer_type
    except Exception:
        pass

    if isinstance(model, LogisticRegression):
        try:
            train_df = get_train_df()
            masker   = shap.maskers.Independent(train_df)
            _shap_explainer = shap.LinearExplainer(model, masker)
            _explainer_type = "linear"
            print("✅ SHAP LinearExplainer ready")
            return _shap_explainer, _explainer_type
        except Exception as e:
            print(f"⚠ LinearExplainer failed: {e}")

    # Universal fallback
    try:
        train_df = get_train_df()
        _shap_explainer = shap.KernelExplainer(
            model.predict_proba, shap.sample(train_df, 50)
        )
        _explainer_type = "kernel"
        print("✅ SHAP KernelExplainer ready (fallback)")
    except Exception as e:
        print(f"⚠ KernelExplainer failed: {e}")
        _shap_explainer = None
        _explainer_type = None

    return _shap_explainer, _explainer_type

# ── Plot helpers ───────────────────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#060a0f", edgecolor="none", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def get_shap_plot(input_df):
    explainer, explainer_type = get_explainer()
    if explainer is None:
        return None

    shap_values = explainer(input_df)

    if explainer_type == "tree":
        vals = shap_values[:, :, 1].values[0] if len(np.array(shap_values).shape) == 3 else shap_values.values[0]
    elif explainer_type == "linear":
        raw  = shap_values.values
        vals = raw[0, :, 1] if len(raw.shape) == 3 else raw[0]
    else:
        vals = np.array(shap_values)[1][0] if isinstance(shap_values, list) else shap_values.values[0]

    feats      = list(feature_cols)
    colors     = ["#ef4444" if v > 0 else "#00ffc8" for v in vals]
    sorted_idx = np.argsort(np.abs(vals))[-8:]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#060a0f")
    ax.set_facecolor("#0d1117")
    ax.barh([feats[i] for i in sorted_idx], [vals[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx], height=0.6)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_title("SHAP — Feature Impact on Prediction", color="#00ffc8", fontsize=11, pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("SHAP Value  (red = ASD, cyan = Non-ASD)", color="#64748b", fontsize=8)
    return fig_to_base64(fig)

def get_lime_plot(input_df):
    train_df = get_train_df()
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data        = train_df.values,
            feature_names        = feature_cols,
            class_names          = ["Non-ASD", "ASD"],
            mode                 = "classification",
            discretize_continuous= True,
            random_state         = 42,
        )
        explanation = lime_explainer.explain_instance(
            input_df.values[0], model.predict_proba,
            num_features=8, top_labels=1,
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
        ax.set_title("LIME — Local Explanation", color="#a78bfa", fontsize=11, pad=10)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.set_xlabel("LIME Weight  (red = ASD, purple = Non-ASD)", color="#64748b", fontsize=8)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"⚠ LIME error: {e}")
        return None

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

    shap_plot = None
    try:
        shap_plot = get_shap_plot(df)
    except Exception as e:
        print(f"⚠ SHAP error: {e}")

    lime_plot = None
    try:
        lime_plot = get_lime_plot(df)
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