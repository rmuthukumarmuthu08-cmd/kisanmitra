"""On-device-style TFLite inference wrapper for the crop-disease model.

The exported TFLite model bakes in MobileNetV2 preprocessing, so this wrapper
only needs to resize the image and feed float32 RGB in [0,255].
"""
import os, json
import numpy as np
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class TFLiteModel:
    def __init__(self):
        self.ready = False
        self.labels = []
        self.version = "untrained"
        self._load()

    def _load(self):
        model_path = os.path.join(MODELS_DIR, "model.tflite")
        labels_path = os.path.join(MODELS_DIR, "labels.json")
        meta_path = os.path.join(MODELS_DIR, "metrics.json")
        if not (os.path.exists(model_path) and os.path.exists(labels_path)):
            return
        # tflite runtime ships inside tensorflow
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        except Exception:
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        _, self.h, self.w, _ = self.inp["shape"]
        with open(labels_path) as f:
            self.labels = json.load(f)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.version = json.load(f).get("model_version", "v1")
        self.ready = True

    def preprocess(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize((self.w, self.h))
        arr = np.asarray(img, dtype=np.float32)[None, ...]  # (1,H,W,3) in [0,255]
        if self.inp["dtype"] == np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def predict(self, pil_img: Image.Image, topk: int = 3):
        if not self.ready:
            raise RuntimeError("Model not loaded")
        x = self.preprocess(pil_img)
        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.out["index"])[0].astype(np.float32)
        if probs.sum() > 1.5 or probs.min() < 0:  # logits -> softmax fallback
            e = np.exp(probs - probs.max()); probs = e / e.sum()
        order = probs.argsort()[::-1][:topk]
        return [{"label": self.labels[i], "confidence": float(probs[i])} for i in order]


MODEL = TFLiteModel()
