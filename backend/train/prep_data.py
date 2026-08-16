"""Download PlantVillage, subsample the 15 tomato/potato/pepper classes, save arrays."""
import os, numpy as np, tensorflow as tf, tensorflow_datasets as tfds

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)
IMG = 160          # input size
PER_CLASS = 260    # max images per class (keeps CPU training fast)

TARGET = [
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

print("[1/4] download_and_prepare ...", flush=True)
builder = tfds.builder('plant_village')
builder.download_and_prepare()
names = builder.info.features['label'].names
target_idx = {names.index(t): t for t in TARGET}          # global idx -> name
local_id = {t: i for i, t in enumerate(TARGET)}           # name -> 0..14
print("    ready. target classes:", len(TARGET), flush=True)

print("[2/4] collecting subset ...", flush=True)
ds = builder.as_dataset(split='train', shuffle_files=True)
counts = {t: 0 for t in TARGET}
X, y = [], []
done = 0
for ex in tfds.as_numpy(ds):
    gi = int(ex['label'])
    if gi not in target_idx:
        continue
    name = target_idx[gi]
    if counts[name] >= PER_CLASS:
        continue
    img = tf.image.resize(ex['image'], (IMG, IMG)).numpy().astype('uint8')
    X.append(img); y.append(local_id[name])
    counts[name] += 1
    done += 1
    if done % 500 == 0:
        print(f"    collected {done}", flush=True)
    if all(c >= PER_CLASS for c in counts.values()):
        break

X = np.asarray(X, dtype='uint8'); y = np.asarray(y, dtype='int64')
print("    total:", X.shape, "per-class:", dict(counts), flush=True)

print("[3/4] shuffle + split ...", flush=True)
rng = np.random.default_rng(42)
perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
n_val = int(0.15 * len(X))
Xv, yv, Xt, yt = X[:n_val], y[:n_val], X[n_val:], y[n_val:]

print("[4/4] saving ...", flush=True)
np.save(f"{OUT}/X_train.npy", Xt); np.save(f"{OUT}/y_train.npy", yt)
np.save(f"{OUT}/X_val.npy", Xv);   np.save(f"{OUT}/y_val.npy", yv)
with open(f"{OUT}/labels.txt", "w") as f:
    f.write("\n".join(TARGET))
print(f"DONE  train={len(Xt)} val={len(Xv)} img={IMG}", flush=True)
