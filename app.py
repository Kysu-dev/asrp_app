"""
FIXED ASR APP
"""

import os
import pickle
import tempfile
import numpy as np
import librosa
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# HARUS SAMA DENGAN TRAINING
SAMPLE_RATE = 16000
DURATION    = 2.0
N_MFCC      = 20
N_FFT       = 512
HOP_LENGTH  = 160

MODEL_PATH  = "models/asr_model.pkl"
SCALER_PATH = "models/scaler.pkl"

CLASSES = [str(i) for i in range(10)]

model = None
scaler = None


# ==========================
# LOAD MODEL
# ==========================
def load_model():
    global model, scaler

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print("[OK] Model Loaded")
    print("[OK] Scaler Loaded")
    print("Scaler feature count:", scaler.n_features_in_)


load_model()


# ==========================
# PREPROCESS
# ==========================
def preprocess_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as tmp:

        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(
            tmp_path,
            sr=SAMPLE_RATE,
            mono=True
        )

    finally:
        os.remove(tmp_path)

    audio, _ = librosa.effects.trim(audio)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    target_length = int(SAMPLE_RATE * DURATION)

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    return audio


# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    combined = np.concatenate(
        [mfcc, delta, delta2],
        axis=0
    )

    feature = np.concatenate([
        combined.mean(axis=1),
        combined.std(axis=1)
    ])

    feature = feature.reshape(1, -1)

    print("\n=== LIVE FEATURE DEBUG ===")
    print("Feature shape:", feature.shape)
    print("Expected:", scaler.n_features_in_)
    print("==========================")

    return feature


# ==========================
# ROUTES
# ==========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        model_ready=True
    )


@app.route("/api/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "message": "Audio tidak ditemukan"
        })

    try:

        file = request.files["audio"]

        audio = preprocess_audio(
            file.read()
        )

        feat = extract_features(audio)

        feat = scaler.transform(feat)

        proba = model.predict_proba(feat)[0]

        pred_idx = int(np.argmax(proba))

        top5_idx = np.argsort(proba)[::-1][:5]

        top5 = []

        for i in top5_idx:
            top5.append({
                "label": CLASSES[i],
                "probability": round(
                    float(proba[i]) * 100,
                    2
                )
            })

        return jsonify({
            "success": True,
            "prediction": CLASSES[pred_idx],
            "confidence": round(
                float(proba[pred_idx]) * 100,
                2
            ),
            "top5": top5
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "success": False,
            "message": str(e)
        })


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": True
    })


if __name__ == "__main__":
    print("=== ASR WEB APP ===")
    print("http://localhost:5000")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )