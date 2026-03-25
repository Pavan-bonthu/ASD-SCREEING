import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# 1. PATH SETUP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "synthetic_asd_dataset.csv")

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
print("📥 Loading dataset...")

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)

# ─────────────────────────────────────────────
# 3. CLEAN DATA
# ─────────────────────────────────────────────
df.replace("?", np.nan, inplace=True)
df.fillna(0, inplace=True)

# Ensure numeric
df = df.apply(pd.to_numeric, errors="coerce")
df.fillna(0, inplace=True)

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df["behavior_score"] = df[[f"A{i}" for i in range(1, 11)]].sum(axis=1)
df["doctor_score"]   = df[[f"D{i}" for i in range(1, 11)]].sum(axis=1)

# ─────────────────────────────────────────────
# 5. TARGET
# ─────────────────────────────────────────────
if "Class/ASD" not in df.columns:
    raise Exception("❌ Target column missing")

X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

print("\n🎯 TARGET DISTRIBUTION:")
print(y.value_counts())

# ─────────────────────────────────────────────
# 6. TRAIN TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────────
print("\n🤖 Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\n" + "="*40)
print(f"🎯 Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, preds, target_names=["Non-ASD", "ASD"]))
print("="*40)

# ─────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────
save_dir = os.path.join(BASE_DIR, "saved")
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "model.pkl"))
joblib.dump(list(X.columns), os.path.join(save_dir, "feature_cols.pkl"))
joblib.dump(acc, os.path.join(save_dir, "accuracy.pkl"))

X_train.to_pickle(os.path.join(save_dir, "train_data.pkl"))

model_info = {
    "model": "RandomForest",
    "accuracy": float(acc),
    "features": list(X.columns)
}

with open(os.path.join(save_dir, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=4)

print("\n💾 Model saved successfully!")

# ─────────────────────────────────────────────
# 10. TEST SAMPLE
# ─────────────────────────────────────────────
sample = X_test.iloc[0:1]

pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

print("\n🧪 TEST SAMPLE:")
print("Prediction:", "ASD" if pred == 1 else "Non-ASD")
print("Confidence:", f"{prob*100:.2f}%")