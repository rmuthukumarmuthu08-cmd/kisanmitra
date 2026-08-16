"""Export the trained CNN so it runs fully in-browser (on-device) with tf.js.

We rebuild a clean inference 'core' (96x96 normalized input -> conv blocks -> dense),
copy the trained weights, validate it matches the original model, then dump the
weights as base64 + sample images + knowledge base into web_assets.json for the app.
"""
import os, json, base64, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model

HERE = os.path.dirname(__file__)
MODELS = os.path.join(os.path.dirname(HERE), "models")
DATA = os.path.join(HERE, "data")
APPDATA = os.path.join(os.path.dirname(HERE), "app", "data")

full = tf.keras.models.load_model(f"{MODELS}/model.keras")
labels = json.load(open(f"{MODELS}/labels.json"))
n = len(labels)
FILTERS = (32, 64, 96, 128)
DENSE = 128


def build_core():
    inp = layers.Input((96, 96, 3))          # already normalized [0,1] in JS
    x = inp
    for f in FILTERS:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(DENSE, activation="relu")(x)
    x = layers.Dense(n, activation="softmax")(x)
    return Model(inp, x)


core = build_core()
# copy weights from layers that have them, in order (conv, bn, dense)
src = [l for l in full.layers if l.get_weights()]
dst = [l for l in core.layers if l.get_weights()]
assert len(src) == len(dst), f"layer mismatch {len(src)} vs {len(dst)}"
for s, d in zip(src, dst):
    d.set_weights(s.get_weights())

# ---- validate: core(normalized 96) must match full(raw 160) ----
Xv = np.load(f"{DATA}/X_val.npy"); yv = np.load(f"{DATA}/y_val.npy")
Xs = Xv[:200]
full_pred = full.predict(Xs.astype("float32"), verbose=0).argmax(1)
norm96 = tf.image.resize(Xs.astype("float32") / 255.0, (96, 96)).numpy()
core_pred = core.predict(norm96, verbose=0).argmax(1)
agree = float((full_pred == core_pred).mean())
core_acc = float((core_pred == yv[:200]).mean())
print(f"core vs full agreement: {agree:.3f} | core acc on 200 val: {core_acc:.3f}", flush=True)
assert agree > 0.97, "core model diverged from trained model"


def b64(a):
    return base64.b64encode(np.asarray(a, dtype=np.float32).tobytes()).decode()


# ---- export weights in forward order ----
blocks = []
for l in core.layers:
    if isinstance(l, layers.Conv2D):
        blocks.append({"type": "conv", "kernel": b64(l.get_weights()[0]),
                       "shape": list(l.get_weights()[0].shape)})
    elif isinstance(l, layers.BatchNormalization):
        g, be, m, v = l.get_weights()
        blocks.append({"type": "bn", "gamma": b64(g), "beta": b64(be),
                       "mean": b64(m), "var": b64(v), "eps": float(l.epsilon),
                       "c": int(g.shape[0])})
    elif isinstance(l, layers.Dense):
        w, bi = l.get_weights()
        blocks.append({"type": "dense", "w": b64(w), "shape": list(w.shape),
                       "b": b64(bi), "act": l.get_config()["activation"]})

# ---- sample images (real val images) for a reliable demo ----
from PIL import Image
import io
want = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Potato___Late_blight", "Pepper,_bell___Bacterial_spot", "Tomato___healthy"]
samples = []
for lab in want:
    ci = labels.index(lab)
    idx = np.where(yv == ci)[0][0]
    buf = io.BytesIO(); Image.fromarray(Xv[idx]).save(buf, format="PNG")
    samples.append({"label": lab, "png": base64.b64encode(buf.getvalue()).decode()})

kb = json.load(open(f"{APPDATA}/diseases.json"))
meta = json.load(open(f"{MODELS}/metrics.json"))

assets = {"labels": labels, "input": 96, "blocks": blocks, "samples": samples,
          "kb": kb, "meta": {k: meta[k] for k in ("model_version", "val_accuracy", "num_classes")}}
out = f"{HERE}/web_assets.json"
json.dump(assets, open(out, "w"), ensure_ascii=False)
print(f"wrote {out}  ({os.path.getsize(out)//1024} KB)  blocks={len(blocks)} samples={len(samples)}", flush=True)
