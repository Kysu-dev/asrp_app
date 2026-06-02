"""
ASR Flask app
"""

import os
import pickle
import tempfile
import numpy as np
import librosa
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# HARUS SAMA DENGAN TRAINING
SAMPLE_RATE = 16000
DURATION    = 2.0
N_MFCC      = 20
N_FFT       = 512
HOP_LENGTH  = 160

MODEL_PATH  = os.path.join(BASE_DIR, "models", "asr_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

CLASSES = [str(i) for i in range(10)]

model = None
scaler = None
model_error = None


# ==========================
# LOAD MODEL
# ==========================
def load_model():
    global model, scaler, model_error

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        model_error = None
        print("[OK] Model Loaded")
        print("[OK] Scaler Loaded")
        print("Scaler feature count:", scaler.n_features_in_)

    except Exception as e:
        model = None
        scaler = None
        model_error = str(e)
        print("[WARN] Model/scaler gagal dimuat:", model_error)


load_model()


def is_model_ready():
    return model is not None and scaler is not None


# ==========================
# PREPROCESS
# ==========================
def preprocess_audio(audio_bytes, suffix=".wav"):

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
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

    feature = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
    feature = feature.reshape(1, -1)

    print("\n=== LIVE FEATURE DEBUG ===")
    print("Feature shape:", feature.shape)
    print("Expected:", scaler.n_features_in_ if scaler is not None else "model not loaded")
    print("==========================")

    return feature


# ==========================
# ROUTES
# ==========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        model_ready=is_model_ready()
    )


@app.route("/api/predict", methods=["POST"])
def predict():

    if not is_model_ready():
        return jsonify({
            "success": False,
            "message": f"Model belum siap: {model_error or 'file model/scaler tidak tersedia'}"
        }), 503

    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "message": "Audio tidak ditemukan"
        }), 400

    try:

        file = request.files["audio"]
        filename = file.filename or ""
        suffix = os.path.splitext(filename)[1].lower() or ".wav"

        audio = preprocess_audio(
            file.read(),
            suffix=suffix
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
        "model_loaded": is_model_ready(),
        "error": model_error
    })


if __name__ == "__main__":
    print("=== ASR WEB APP ===")
    print("http://localhost:5000")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
