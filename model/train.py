import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ── 1. Load CSV (skip the malformed first row using header=1) ──────────────
df = pd.read_csv("model/data/Autism_Data.csv", header=1)

# ── 2. Fix column names (first score col is 'ore', id col is '1') ──────────
df = df.rename(columns={"ore": "A1_Score", "1": "id"})

# ── 3. Drop irrelevant columns ─────────────────────────────────────────────
df.drop(columns=["id", "age_desc", "relation", "used_app_before"], errors="ignore", inplace=True)

# ── 4. Handle missing values ───────────────────────────────────────────────
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# ── 5. Encode categorical columns ─────────────────────────────────────────
# Note: your CSV uses 'jundice' and 'austim' (typos in the dataset — keep as-is)
cat_cols = ["gender", "ethnicity", "jundice", "austim", "contry_of_res", "Class/ASD"]
label_encoders = {}
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
# ── 6. Force all columns to numeric (fixes mixed type errors) ──────────────
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

# ── 7. Split features and target ───────────────────────────────────────────
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

print(f"✅ Dataset ready — {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 7. Train all 3 models, pick the best ───────────────────────────────────
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":              XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

best_model, best_acc, best_name = None, 0, ""

for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{'─'*40}")
    print(f"  {name}: {acc*100:.2f}% accuracy")
    print(classification_report(y_test, preds, target_names=["Non-ASD", "ASD"]))
    if acc > best_acc:
        best_acc, best_model, best_name = acc, model, name

# ── 8. Save best model and metadata ───────────────────────────────────────
os.makedirs("model/saved", exist_ok=True)
joblib.dump(best_model,           "model/saved/model.pkl")
joblib.dump(list(X.columns),      "model/saved/feature_cols.pkl")
joblib.dump(best_acc,             "model/saved/accuracy.pkl")
joblib.dump(label_encoders,       "model/saved/label_encoders.pkl")

print(f"\n{'='*40}")
print(f"🏆 Best Model : {best_name}")
print(f"   Accuracy   : {best_acc*100:.2f}%")
print(f"   Saved to   : model/saved/")
print(f"{'='*40}")

# Save training data for LIME background
X_train.to_pickle("model/saved/train_data.pkl")
print("✅ Training data saved for LIME explainer")