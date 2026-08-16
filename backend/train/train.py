"""Train the KisanMitra crop-disease classifier, evaluate, and export a
quantized TFLite edge model + labels + metrics.

Compact CNN trained from scratch (no external pretrained weights — runs in a
firewalled sandbox). Deeper/wider net + class weights + more data for accuracy.
"""
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(os.path.dirname(HERE), "models")
os.makedirs(OUT, exist_ok=True)
IMG, TRAIN_RES = 160, 96
FILTERS = (32, 64, 96, 128)
DENSE = 128

print("loading arrays ...", flush=True)
Xt = np.load(f"{DATA}/X_train.npy"); yt = np.load(f"{DATA}/y_train.npy")
Xv = np.load(f"{DATA}/X_val.npy");   yv = np.load(f"{DATA}/y_val.npy")
labels = open(f"{DATA}/labels.txt").read().splitlines()
n_classes = len(labels)
print(f"train={len(Xt)} val={len(Xv)} classes={n_classes}", flush=True)

# class weights (handles imbalance, e.g. potato_healthy)
counts = np.bincount(yt, minlength=n_classes)
cw = {i: float(len(yt) / (n_classes * max(counts[i], 1))) for i in range(n_classes)}


def build():
    inp = layers.Input(shape=(IMG, IMG, 3), name="image")
    x = layers.Rescaling(1 / 255.0)(inp)
    x = layers.Resizing(TRAIN_RES, TRAIN_RES)(x)
    x = tf.keras.Sequential([layers.RandomFlip("horizontal"),
                             layers.RandomRotation(0.12),
                             layers.RandomZoom(0.12),
                             layers.RandomContrast(0.1)], name="augment")(x)
    for f in FILTERS:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(DENSE, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="kisanmitra_cropnet")


model = build()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cb = [
    tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_accuracy"),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss", min_lr=1e-5),
]
AUTO = tf.data.AUTOTUNE
def cast(x, y): return tf.cast(x, tf.float32), y
train_ds = (tf.data.Dataset.from_tensor_slices((Xt, yt)).shuffle(2048)
            .batch(64).map(cast, AUTO).prefetch(AUTO))
val_ds = tf.data.Dataset.from_tensor_slices((Xv, yv)).batch(64).map(cast, AUTO).prefetch(AUTO)

hist = model.fit(train_ds, validation_data=val_ds, epochs=50,
                 class_weight=cw, callbacks=cb, verbose=2)

val_acc = float(max(hist.history["val_accuracy"]))
pred = model.predict(val_ds, verbose=0).argmax(1)
per_class = {labels[c]: round(float((pred[yv == c] == c).mean()) if (yv == c).sum() else 0.0, 3)
             for c in range(n_classes)}
print(f"BEST VAL ACCURACY: {val_acc:.4f}", flush=True)

model.save(f"{OUT}/model.keras")

print("converting to quantized TFLite ...", flush=True)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
tfl = conv.convert()
open(f"{OUT}/model.tflite", "wb").write(tfl)
size_mb = round(len(tfl) / 1e6, 2)

json.dump(labels, open(f"{OUT}/labels.json", "w"), ensure_ascii=False, indent=2)
json.dump({
    "model_version": "cropnet-cnn-real-v2",
    "architecture": f"5-block CNN {FILTERS} + dense{DENSE} (from scratch)",
    "dataset": "PlantVillage — REAL leaf photos (tomato/potato/pepper subset, 15 classes)",
    "note": "Trained on real PlantVillage photos from the GitHub mirror (spMohanty/PlantVillage-Dataset).",
    "input_size": IMG, "train_resolution": TRAIN_RES, "num_classes": n_classes,
    "train_images": int(len(Xt)), "val_images": int(len(Xv)),
    "val_accuracy": round(val_acc, 4), "tflite_size_mb": size_mb,
    "per_class_accuracy": per_class,
}, open(f"{OUT}/metrics.json", "w"), ensure_ascii=False, indent=2)
print(f"DONE  val_acc={val_acc:.3f}  tflite={size_mb}MB", flush=True)
