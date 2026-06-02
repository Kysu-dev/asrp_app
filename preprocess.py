"""
preprocess.py
Ekstraksi fitur MFCC dari dataset audio (OPTIMIZED VERSION)

Struktur folder dataset:
dataset/
  0/
  1/
  ...
  9/

Output: models/features.pkl
"""

import os
import pickle
import numpy as np
import librosa
import noisereduce as nr
from tqdm import tqdm
import random

# ── Konfigurasi ──────────────────────────────────────────────
DATASET_DIR  = "dataset"
OUTPUT_PATH  = "models/features.pkl"

SAMPLE_RATE  = 16000
DURATION     = 2.0

# OPTIMIZED untuk MLP + noise dataset
N_MFCC       = 20
N_FFT        = 512
HOP_LENGTH   = 160

CLASSES      = [str(i) for i in range(10)]
# ─────────────────────────────────────────────────────────────


def load_and_clean(path: str) -> np.ndarray:
    """Load audio + preprocessing (robust version)."""

    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # ── Noise reduction (lebih aman) ──
    try:
        audio = nr.reduce_noise(y=audio, sr=sr, stationary=True)
    except:
        pass  # fallback kalau gagal

    # ── Trim silence (lebih longgar biar tidak kepotong kata) ──
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # ── Normalisasi stabil ──
    audio = librosa.util.normalize(audio)

    # ── Padding / truncation ──
    target_len = int(SAMPLE_RATE * DURATION)

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    return audio


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """MFCC + delta + delta-delta → mean + std features"""

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.concatenate([mfcc, delta, delta2], axis=0)

    # Statistik agar fixed-length (MLP friendly)
    features = np.concatenate([
        combined.mean(axis=1),
        combined.std(axis=1)
    ])

    # Normalisasi feature vector (IMPORTANT untuk MLP)
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)

    return features


def build_dataset():
    X, y = [], []
    missing = []

    for label in CLASSES:
        folder = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(folder):
            missing.append(label)
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))]

        files = sorted(files)
        random.shuffle(files)

        print(f"Kelas '{label}': {len(files)} file")

        for fname in tqdm(files, desc=f"Proses kelas {label}", leave=False):
            fpath = os.path.join(folder, fname)

            try:
                audio = load_and_clean(fpath)
                features = extract_mfcc(audio)

                X.append(features)
                y.append(int(label))

            except Exception as e:
                print(f"[SKIP] {fname}: {e}")

    if missing:
        print(f"\n[WARN] Folder tidak ditemukan: {missing}")

    X = np.array(X)
    y = np.array(y)

    print("\n=== Dataset Summary ===")
    print(f"Total sampel : {len(X)}")
    print(f"Shape fitur  : {X.shape}")

    # ── CHECK imbalance ──
    unique, counts = np.unique(y, return_counts=True)
    print("\nDistribusi kelas:")
    for u, c in zip(unique, counts):
        print(f"  Kelas {u}: {c}")

    return X, y


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("=== PREPROCESSING START ===")

    X, y = build_dataset()

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "X": X,
            "y": y,
            "classes": CLASSES
        }, f)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("Next step: train.py")