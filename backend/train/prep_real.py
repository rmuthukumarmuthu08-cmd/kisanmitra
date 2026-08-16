"""Build train/val arrays from the downloaded REAL PlantVillage photos."""
import os, glob, numpy as np
from PIL import Image

SRC = "/home/claude/pv_real"
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)
IMG = 160
MAX_PER_CLASS = 500   # cap to keep CPU training time reasonable

LABELS = [
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

X, y = [], []
for ci, lab in enumerate(LABELS):
    files = sorted(glob.glob(os.path.join(SRC, lab, "*")))[:MAX_PER_CLASS]
    n = 0
    for f in files:
        try:
            im = Image.open(f).convert("RGB").resize((IMG, IMG))
        except Exception:
            continue
        X.append(np.asarray(im, dtype="uint8")); y.append(ci); n += 1
    print(f"{lab}: {n}", flush=True)

X = np.asarray(X, dtype="uint8"); y = np.asarray(y, dtype="int64")
rng = np.random.default_rng(42)
perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
n_val = int(0.15 * len(X))
np.save(f"{OUT}/X_val.npy", X[:n_val]);   np.save(f"{OUT}/y_val.npy", y[:n_val])
np.save(f"{OUT}/X_train.npy", X[n_val:]); np.save(f"{OUT}/y_train.npy", y[n_val:])
open(f"{OUT}/labels.txt", "w").write("\n".join(LABELS))
print(f"DONE  total={len(X)} train={len(X)-n_val} val={n_val}", flush=True)
