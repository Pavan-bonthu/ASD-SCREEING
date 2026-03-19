import os
import shutil
import random

# ── UPDATE THESE 2 PATHS TO YOUR ACTUAL IMAGE FOLDERS ─────────────────────
AUTISTIC_SOURCE     = r"E:\asd\Asd-screening/Backend/model/Autistic"       # ← change this
NON_AUTISTIC_SOURCE = r"E:\asd\Asd-screening/Backend/model/Non-Autistic"   # ← change this
# ──────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = r"E:\asd\Asd-screening\Backend\model\image_data"
TRAIN_RATIO  = 0.70
VALID_RATIO  = 0.15
# TEST gets the remaining 15%
SEED         = 42
random.seed(SEED)

# Accepts all common image formats including mixed types
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

def split_and_copy(source_dir, class_name):
    images = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(VALID_EXTS)
    ]
    random.shuffle(images)

    total   = len(images)
    train_n = int(total * TRAIN_RATIO)
    valid_n = int(total * VALID_RATIO)

    splits = {
        "train": images[:train_n],
        "valid": images[train_n:train_n + valid_n],
        "test":  images[train_n + valid_n:],
    }

    print(f"\n📁 {class_name}: {total} total images")
    for split_name, split_files in splits.items():
        dest = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(dest, exist_ok=True)
        for fname in split_files:
            shutil.copy2(
                os.path.join(source_dir, fname),
                os.path.join(dest, fname)
            )
        print(f"   {split_name:6s} → {len(split_files):4d} images")

split_and_copy(AUTISTIC_SOURCE,     "Autistic")
split_and_copy(NON_AUTISTIC_SOURCE, "Non_Autistic")

# ── Verify ─────────────────────────────────────────────────────────────────
print("\n✅ Final structure:")
for split in ["train", "valid", "test"]:
    for cls in ["Autistic", "Non_Autistic"]:
        path  = os.path.join(OUTPUT_DIR, split, cls)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"   {split:6s}/{cls:15s}: {count} images")