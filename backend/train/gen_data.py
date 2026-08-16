"""Procedurally render a leaf-disease dataset for the 15 tomato/potato/pepper
classes, matching each disease's real visual signature.

WHY: this sandbox's egress proxy blocks the PlantVillage image hosts (HF, GDrive,
Kaggle, GCS). This generator lets us train + validate the FULL pipeline end-to-end
here. For production, run prep_data.py instead to train on real PlantVillage photos
(identical downstream code). Classes are given distinct visual signatures so the
CNN has a real, learnable signal.
"""
import os, math, numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)
IMG = 160
PER_CLASS = 320
rng = np.random.default_rng(7)

# label -> (crop_tint, pattern, lesion_rgb, count_range, size_range, halo)
CROP_TINT = {"pepper": (70, 150, 70), "potato": (74, 128, 78), "tomato": (58, 138, 66)}
LABELS = [
    ("Pepper,_bell___Bacterial_spot", "pepper", "spot",   (70, 45, 25), (18, 34), (2, 4), True),
    ("Pepper,_bell___healthy",        "pepper", "none",   (0, 0, 0),    (0, 0),   (0, 0), False),
    ("Potato___Early_blight",         "potato", "blotch", (96, 62, 30), (4, 8),   (9, 16), True),
    ("Potato___Late_blight",          "potato", "blotch", (60, 55, 48), (3, 6),   (13, 23), False),
    ("Potato___healthy",              "potato", "none",   (0, 0, 0),    (0, 0),   (0, 0), False),
    ("Tomato___Bacterial_spot",       "tomato", "spot",   (55, 38, 22), (22, 40), (2, 3), True),
    ("Tomato___Early_blight",         "tomato", "ring",   (98, 60, 28), (4, 9),   (8, 15), True),
    ("Tomato___Late_blight",          "tomato", "blotch", (72, 78, 66), (3, 6),   (13, 24), False),
    ("Tomato___Leaf_Mold",            "tomato", "mottle", (150, 150, 70),(6, 12), (7, 13), False),
    ("Tomato___Septoria_leaf_spot",   "tomato", "spot",   (150, 130, 95),(26, 46),(2, 4), True),
    ("Tomato___Spider_mites_Two-spotted_spider_mite", "tomato", "stipple", (170, 150, 80), (70, 130), (1, 2), False),
    ("Tomato___Target_Spot",          "tomato", "ring",   (92, 58, 34), (10, 20), (4, 8), True),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato", "mottle", (205, 190, 60), (8, 16), (10, 18), False),
    ("Tomato___Tomato_mosaic_virus",  "tomato", "mosaic", (95, 165, 80), (10, 18), (10, 20), False),
    ("Tomato___healthy",              "tomato", "none",   (0, 0, 0),    (0, 0),   (0, 0), False),
]
NAMES = [x[0] for x in LABELS]


def leaf_mask(draw_size, crop):
    """Return an (IMG,IMG) alpha mask + base leaf image with veins."""
    base = Image.new("RGB", (IMG, IMG), (238, 236, 228))
    mask = Image.new("L", (IMG, IMG), 0)
    md = ImageDraw.Draw(mask)
    tint = CROP_TINT[crop]
    tint = tuple(int(np.clip(c + rng.integers(-14, 14), 20, 235)) for c in tint)
    leaf = Image.new("RGB", (IMG, IMG), tint)
    # elliptical leaf, random size/rotation
    w = rng.integers(70, 96); h = rng.integers(104, 140)
    cx, cy = IMG // 2 + rng.integers(-8, 8), IMG // 2 + rng.integers(-8, 8)
    box = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
    md.ellipse(box, fill=255)
    ang = rng.integers(0, 360)
    mask = mask.rotate(ang, expand=False)
    # veins (draw on leaf)
    ld = ImageDraw.Draw(leaf)
    vein = tuple(max(0, c - 40) for c in tint)
    ld.line([(cx, cy - h // 2), (cx, cy + h // 2)], fill=vein, width=2)
    for t in np.linspace(0.2, 0.85, 6):
        yy = int(cy - h // 2 + t * h)
        ld.line([(cx, yy), (cx + int(w * 0.5), yy - 8)], fill=vein, width=1)
        ld.line([(cx, yy), (cx - int(w * 0.5), yy - 8)], fill=vein, width=1)
    leaf = leaf.rotate(ang, expand=False)
    # subtle shading
    leaf = Image.composite(leaf, base, mask)
    return leaf, mask, (cx, cy, w, h, ang)


def paint_lesions(img, mask, geom, spec):
    _, pattern, rgb, cnt, sz, halo = spec[1], spec[2], spec[3], spec[4], spec[5], spec[6]
    if pattern == "none":
        return img
    cx, cy, w, h, ang = geom
    d = ImageDraw.Draw(img, "RGBA")
    n = int(rng.integers(cnt[0], cnt[1] + 1))
    for _ in range(n):
        # random point roughly within leaf bbox
        px = int(cx + rng.normal(0, w * 0.28))
        py = int(cy + rng.normal(0, h * 0.28))
        if mask.getpixel((min(max(px, 0), IMG - 1), min(max(py, 0), IMG - 1))) < 128:
            continue
        r = int(rng.integers(sz[0], sz[1] + 1))
        col = tuple(int(np.clip(c + rng.integers(-18, 18), 0, 255)) for c in rgb)
        if halo:
            hr = r + rng.integers(2, 5)
            d.ellipse([px - hr, py - hr, px + hr, py + hr], fill=(225, 205, 70, 90))
        if pattern == "ring":
            for rr in range(r, 1, -3):
                shade = tuple(int(c * (0.6 + 0.4 * (rr / r))) for c in col)
                d.ellipse([px - rr, py - rr, px + rr, py + rr], outline=shade + (255,), width=1)
            d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=col + (255,))
        elif pattern == "mottle" or pattern == "mosaic":
            d.ellipse([px - r, py - r, px + r, py + r], fill=col + (120,))
        elif pattern == "stipple":
            d.ellipse([px - r, py - r, px + r, py + r], fill=col + (200,))
        else:  # spot / blotch
            d.ellipse([px - r, py - r, px + r, py + r], fill=col + (235,))
    if pattern in ("mottle", "mosaic"):
        img = img.filter(ImageFilter.GaussianBlur(1.2))
    return img


def render(spec):
    leaf, mask, geom = leaf_mask(IMG, spec[1])
    leaf = paint_lesions(leaf, mask, geom, spec)
    # light noise + brightness jitter
    arr = np.asarray(leaf).astype(np.float32)
    arr += rng.normal(0, 6, arr.shape)
    arr *= rng.uniform(0.9, 1.1)
    return np.clip(arr, 0, 255).astype(np.uint8)


print("rendering ...", flush=True)
X, y = [], []
for ci, spec in enumerate(LABELS):
    for _ in range(PER_CLASS):
        X.append(render(spec)); y.append(ci)
    print(f"  {spec[0]}  done", flush=True)
X = np.asarray(X, dtype="uint8"); y = np.asarray(y, dtype="int64")

perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
n_val = int(0.15 * len(X))
np.save(f"{OUT}/X_val.npy", X[:n_val]);   np.save(f"{OUT}/y_val.npy", y[:n_val])
np.save(f"{OUT}/X_train.npy", X[n_val:]); np.save(f"{OUT}/y_train.npy", y[n_val:])
open(f"{OUT}/labels.txt", "w").write("\n".join(NAMES))
print(f"DONE  total={len(X)}  train={len(X)-n_val}  val={n_val}  classes={len(LABELS)}", flush=True)
