"""
MalariaAI Diagnostic Portal — Flask Backend
=============================================
B.Tech CSE Minor Project
Model : MobileNetV2 (Transfer Learning, 94.45 % Test Accuracy)
Dataset: NIH NLM Malaria — 27 558 cell images (80/10/10 split)
"""

import os
import time
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# ── App & Model ──────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "malaria_best_model.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page diagnostic dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an uploaded cell image, run MobileNetV2 inference, and return:
      - prediction   : human-readable label
      - is_parasitized: boolean flag
      - confidence    : percentage (0-100)
      - raw_sigmoid   : raw model output
      - inference_ms  : wall-clock inference time in milliseconds
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # ── Pre-processing (matches training pipeline) ───────────────────
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((64, 64))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)          # (1, 64, 64, 3)

        # ── Inference ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        prob = float(model.predict(img_array, verbose=0)[0][0])  # sigmoid output
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        # ── Label logic (TFDS Malaria): 0 → Parasitized, 1 → Uninfected ─
        if prob < 0.5:
            is_parasitized = True
            prediction = "Parasitized (Malaria Detected)"
            confidence = round((1.0 - prob) * 100, 2)
        else:
            is_parasitized = False
            prediction = "Uninfected (Healthy RBC)"
            confidence = round(prob * 100, 2)

        return jsonify({
            "prediction": prediction,
            "is_parasitized": is_parasitized,
            "confidence": confidence,
            "raw_sigmoid": round(prob, 6),
            "inference_ms": inference_ms,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Dev Server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[OK] Model loaded from {MODEL_PATH}")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
