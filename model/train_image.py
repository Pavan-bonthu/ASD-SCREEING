import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from PIL import Image, UnidentifiedImageError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json, shutil

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR  = "model/image_data"
SAVE_DIR  = "model/saved"
IMG_SIZE  = 224
BATCH     = 16
EPOCHS    = 20
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"✅ Device : {DEVICE}")

# ── Remove corrupt/unreadable images ───────────────────────────────────────
print("🔍 Scanning for corrupt images...")
removed = 0
for root, dirs, files in os.walk(BASE_DIR):
    for fname in files:
        fpath = os.path.join(root, fname)
        try:
            img = Image.open(fpath)
            img.verify()
        except (UnidentifiedImageError, Exception):
            os.remove(fpath)
            removed += 1
print(f"   Removed {removed} corrupt files\n")

# ── Transforms ──────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset loader ──────────────────────────────────────────────────────────
train_data = datasets.ImageFolder(os.path.join(BASE_DIR, "train"), train_tf)
val_data   = datasets.ImageFolder(os.path.join(BASE_DIR, "valid"), val_tf)
test_data  = datasets.ImageFolder(os.path.join(BASE_DIR, "test"),  val_tf)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_data,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=BATCH, shuffle=False, num_workers=0)

print(f"   Classes : {train_data.classes}")
print(f"   Train   : {len(train_data)}")
print(f"   Val     : {len(val_data)}")
print(f"   Test    : {len(test_data)}\n")

# ── Model: MobileNetV2 ──────────────────────────────────────────────────────
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2),
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

history  = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_val = 0.0

# ── Phase 1: Train classifier head ─────────────────────────────────────────
print("── Phase 1: Training classifier head ──────────────────────────────")
for epoch in range(EPOCHS):
    model.train()
    t_loss = t_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * imgs.size(0)
        t_correct += (out.argmax(1) == labels).sum().item()

    model.eval()
    v_loss = v_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out    = model(imgs)
            loss   = criterion(out, labels)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (out.argmax(1) == labels).sum().item()

    t_acc = t_correct / len(train_data) * 100
    v_acc = v_correct / len(val_data)   * 100
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)
    history["train_loss"].append(t_loss / len(train_data))
    history["val_loss"].append(v_loss / len(val_data))
    scheduler.step()

    tag = ""
    if v_acc > best_val:
        best_val = v_acc
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "image_model_best.pth"))
        tag = " ← best saved"

    print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train {t_acc:.1f}% loss={t_loss/len(train_data):.4f} | "
          f"Val {v_acc:.1f}%{tag}")

# ── Phase 2: Fine-tune last 3 layers ───────────────────────────────────────
print("\n── Phase 2: Fine-tuning last 3 layers ─────────────────────────────")
for param in list(model.features[-3:].parameters()):
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

for epoch in range(10):
    model.train()
    t_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_correct += (out.argmax(1) == labels).sum().item()

    model.eval()
    v_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            v_correct += (model(imgs).argmax(1) == labels.to(DEVICE)).sum().item()

    t_acc = t_correct / len(train_data) * 100
    v_acc = v_correct / len(val_data)   * 100

    tag = ""
    if v_acc > best_val:
        best_val = v_acc
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "image_model_best.pth"))
        tag = " ← best saved"

    print(f"  FT {epoch+1:02d}/10 | Train {t_acc:.1f}% | Val {v_acc:.1f}%{tag}")

# ── Test evaluation ─────────────────────────────────────────────────────────
print("\n── Test Evaluation ─────────────────────────────────────────────────")
model.load_state_dict(
    torch.load(os.path.join(SAVE_DIR, "image_model_best.pth"),
               map_location=DEVICE)
)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        preds  = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
print(f"\n🏆 Test Accuracy : {test_acc:.2f}%")
print(classification_report(all_labels, all_preds,
                             target_names=train_data.classes))

# ── Save metadata ────────────────────────────────────────────────────────────
meta = {
    "classes":  train_data.classes,
    "accuracy": round(test_acc, 2),
    "img_size": IMG_SIZE,
    "model":    "MobileNetV2",
}
with open(os.path.join(SAVE_DIR, "image_model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#060a0f")
for ax in [ax1, ax2]:
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

ax1.plot(history["train_acc"],  color="#00ffc8", label="Train")
ax1.plot(history["val_acc"],    color="#00c8ff", label="Val", linestyle="--")
ax1.set_title("Accuracy",  color="#00ffc8", pad=8)
ax1.legend(facecolor="#0d1117", labelcolor="white")

ax2.plot(history["train_loss"], color="#ef4444", label="Train")
ax2.plot(history["val_loss"],   color="#fb923c", label="Val",  linestyle="--")
ax2.set_title("Loss", color="#ef4444", pad=8)
ax2.legend(facecolor="#0d1117", labelcolor="white")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "image_training_history.png"),
            facecolor="#060a0f", dpi=120)

print(f"\n✅ Saved: model/saved/image_model_best.pth")
print(f"✅ Saved: model/saved/image_model_meta.json")
print(f"✅ Saved: model/saved/image_training_history.png")