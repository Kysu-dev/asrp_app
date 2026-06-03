import os
import pickle
import tempfile
import numpy as np
import librosa
import librosa.display
import warnings
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16000
DURATION    = 2.0
N_MFCC      = 20
N_FFT       = 512
HOP_LENGTH  = 160

MODEL_PATH  = os.path.join(BASE_DIR, "models", "asr_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
CLASSES     = [str(i) for i in range(10)]

model       = None
scaler      = None
model_error = None


def load_model():
    global model, scaler, model_error
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        model_error = None
        print(f"[OK] Model loaded — expected features: {scaler.n_features_in_}")
    except Exception as e:
        model = scaler = None
        model_error = str(e)
        print("[WARN] Model/scaler failed to load:", model_error)


load_model()


def is_model_ready():
    return model is not None and scaler is not None


def _highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    sos = butter(4, cutoff / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def preprocess_audio(audio_bytes: bytes, suffix: str = ".wav") -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.remove(tmp_path)

    audio = _highpass_filter(audio, SAMPLE_RATE, cutoff=80.0)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    audio = librosa.effects.preemphasis(audio)
    audio = librosa.util.normalize(audio)
    
    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) > target_len:
        audio = audio[:target_len]
        
    return audio


def extract_features(audio: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    
    combined = np.vstack([mfcc, delta])
    
    n_frames = combined.shape[1]
    if n_frames < 3:
        combined = np.pad(combined, ((0,0), (0, 3 - n_frames)), mode='edge')
        
    chunks = np.array_split(combined, 3, axis=1)
    
    features = []
    for chunk in chunks:
        mean_feat = np.mean(chunk, axis=1)
        std_feat  = np.std(chunk, axis=1)
        features.extend([mean_feat, std_feat])
        
    features = np.concatenate(features) # 240 Fitur
    return features.reshape(1, -1).astype(np.float32), mfcc


@app.route("/")
def home():
    return render_template("index.html", model_ready=is_model_ready())


@app.route("/api/predict", methods=["POST"])
def predict():
    if not is_model_ready():
        return jsonify({"success": False, "message": f"Model error: {model_error}"}), 503

    if "audio" not in request.files:
        return jsonify({"success": False, "message": "No audio"}), 400

    try:
        file   = request.files["audio"]
        suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
        audio  = preprocess_audio(file.read(), suffix=suffix)
        feat, mfcc_raw = extract_features(audio)

        expected = scaler.n_features_in_
        got      = feat.shape[1]
        if got != expected:
            return jsonify({
                "success": False,
                "message": f"Dim error: model needs {expected}, got {got}."
            }), 500

        feat_scaled = scaler.transform(feat)
        proba       = model.predict_proba(feat_scaled)[0]
        pred_idx    = int(np.argmax(proba))
        classes     = getattr(model, "classes_", CLASSES)
        pred_label  = str(classes[pred_idx])
        top5_idx    = np.argsort(proba)[::-1][:5]

        top5 = [{"label": str(classes[i]), "probability": round(float(proba[i]) * 100, 2)} for i in top5_idx]

        # ── Generate MFCC Visualisation 3D ──
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        # Exclude 0th coefficient to prevent color washout
        mfcc_vis = mfcc_raw[1:13, :]  # Ambil 12 koefisien agar plot tidak terlalu padat
        
        X = np.arange(mfcc_vis.shape[1])
        Y = np.arange(mfcc_vis.shape[0]) + 1  # Index mulai dari 1
        X, Y = np.meshgrid(X, Y)
        
        # Gunakan colormap jet (seperti referensi gambar)
        surf = ax.plot_surface(X, Y, mfcc_vis, cmap='jet', linewidth=0, antialiased=True, alpha=0.9)
        
        ax.set_title('Visualisasi MFCC (3D)')
        ax.set_xlabel('Time (Frame number)', labelpad=10)
        ax.set_ylabel('MFCC index', labelpad=10)
        ax.set_zlabel('Magnitude', labelpad=10)
        
        # Atur sudut pandang kamera agar waktu (Time) berjalan dari kiri ke kanan
        ax.view_init(elev=30, azim=130)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        mfcc_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            "success":    True,
            "prediction": pred_label,
            "confidence": round(float(proba[pred_idx]) * 100, 2),
            "top5":       top5,
            "mfcc_image": "data:image/png;base64," + mfcc_b64
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/status")
def status():
    info = {}
    if is_model_ready():
        info["feature_dim"] = scaler.n_features_in_
        info["classes"]     = CLASSES
    return jsonify({
        "model_loaded": is_model_ready(),
        "error":        model_error,
        **info
    })


@app.route("/api/reload", methods=["POST"])
def reload_model():
    load_model()
    return jsonify({"model_loaded": is_model_ready(), "error": model_error})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
