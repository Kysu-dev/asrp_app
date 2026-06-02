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

# Fitur: MFCC 20 + delta + delta² + Chroma + Spectral Contrast + ZCR
# Total dimensi: (20*3)*2 + 12*2 + 7*2 + 1*2 = 120 + 24 + 14 + 2 = 160 dim
N_MFCC       = 20
N_FFT        = 512
HOP_LENGTH   = 160

CLASSES      = [str(i) for i in range(10)]
# ─────────────────────────────────────────────────────────────


# ── 5 Strategi Augmentasi (dipilih acak, selalu terjadi) ─────
# Masing-masing strategi membuat audio yang benar-benar berbeda
# sehingga model belajar variasi antar speaker lebih baik.
AUG_STRATEGIES = [
    # 0: Noise ringan
    lambda a, sr: a + np.random.randn(len(a)) * random.uniform(0.005, 0.015),
    # 1: Noise sedang
    lambda a, sr: a + np.random.randn(len(a)) * random.uniform(0.015, 0.030),
    # 2: Time stretch (melambat/mempercepat ucapan)
    lambda a, sr: librosa.effects.time_stretch(a, rate=random.uniform(0.80, 1.20)),
    # 3: Pitch shift (nada naik/turun)
    lambda a, sr: librosa.effects.pitch_shift(a, sr=sr, n_steps=random.uniform(-2.5, 2.5)),
    # 4: Kombinasi noise + pitch (variasi terkuat)
    lambda a, sr: librosa.effects.pitch_shift(
        a + np.random.randn(len(a)) * random.uniform(0.003, 0.012),
        sr=sr, n_steps=random.uniform(-1.5, 1.5)
    ),
]


def apply_augmentation(audio: np.ndarray, sr: int) -> np.ndarray:
    """Pilih satu strategi augmentasi secara acak — selalu terjadi."""
    strategy = random.choice(AUG_STRATEGIES)
    try:
        return strategy(audio, sr)
    except Exception:
        # Fallback ke noise ringan jika strategi gagal
        return audio + np.random.randn(len(audio)) * 0.005


def load_and_clean(path: str, augment: bool = False) -> np.ndarray:
    """Load audio + preprocessing.

    Urutan penting:
    1. Load
    2. Augmentasi (sebelum noise reduction agar aug tetap terasa natural)
    3. Noise reduction
    4. Trim silence (longgar: top_db=25 agar kata tidak terpotong)
    5. Normalisasi
    6. Pad/truncate ke durasi tetap
    """
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # ── Augmentasi (SELALU terjadi jika augment=True) ──────────
    if augment:
        audio = apply_augmentation(audio, sr)

    # ── Noise reduction ─────────────────────────────────────────
    try:
        audio = nr.reduce_noise(y=audio, sr=sr, stationary=True)
    except Exception:
        pass  # fallback jika gagal

    # ── Trim silence (top_db=25 → lebih longgar, kata utuh) ────
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # ── Normalisasi amplitude ───────────────────────────────────
    audio = librosa.util.normalize(audio)

    # ── Pad / truncate ke panjang tetap ────────────────────────
    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    return audio


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Ekstrak fitur speaker-independent:
      - MFCC 20 + CMVN + delta + delta²  → mean + std  (120 dim)
      - Chroma STFT                       → mean + std  ( 24 dim)
      - Spectral Contrast                 → mean + std  ( 14 dim)
      - Zero Crossing Rate                → mean + std  (  2 dim)
    Total: 160 dimensi

    CMVN (Cepstral Mean Variance Normalization):
      Mengurangi pengaruh karakteristik speaker spesifik (kanal suara,
      frekuensi dasar) dari MFCC. Teknik standar dalam speaker-independent ASR.
    """

    # ── MFCC ────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # ── CMVN: kurangi mean per koefisien per utterance ──
    # Ini menghilangkan offset speaker (perbedaan nada dasar, mikrofon, dll)
    # sehingga model lebih fokus pada pola fonetik, bukan identitas speaker.
    mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)

    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.concatenate([mfcc, delta, delta2], axis=0)  # (60, T)

    # ── Chroma STFT (membedakan nada vokal) ─────────────
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )  # (12, T)

    # ── Spectral Contrast (tekstur spektral, bantu bedakan konsonan) ──
    spec_contrast = librosa.feature.spectral_contrast(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )  # (7, T)

    # ── Zero Crossing Rate (deteksi konsonan letup /k/, /t/) ──
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, hop_length=HOP_LENGTH
    )  # (1, T)

    # ── Statistik: mean + std per koefisien → fixed-length vector ──
    def stat(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x.mean(axis=1), x.std(axis=1)])

    features = np.concatenate([
        stat(mfcc_combined),   # 120 dim
        stat(chroma),          #  24 dim
        stat(spec_contrast),   #  14 dim
        stat(zcr),             #   2 dim
    ])  # total 160 dim

    # Normalisasi global (per-fitur) dilakukan oleh StandardScaler di train.py
    return features


def build_dataset():
    X, y, groups, paths = [], [], [], []
    missing = []

    for label in CLASSES:
        folder = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(folder):
            missing.append(label)
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))]
                 # File _dup_ diikutsertakan untuk menjaga keseimbangan kelas.
                 # Leakage sudah dicegah oleh GroupShuffleSplit di train.py:
                 # file asli + _dup_ dari speaker yang sama → masuk split yang sama.

        files = sorted(files)
        random.shuffle(files)

        print(f"Kelas '{label}': {len(files)} file")

        for fname in tqdm(files, desc=f"Proses kelas {label}", leave=False):
            fpath = os.path.join(folder, fname)

            try:
                audio = load_and_clean(fpath, augment=False)
                features = extract_mfcc(audio)

                X.append(features)
                y.append(int(label))
                paths.append(fpath)
                # Use filename prefix as speaker/group id when available.
                if "_" in fname:
                    groups.append(fname.split("_")[0])
                else:
                    groups.append("unknown")

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

    return X, y, groups, paths


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("=== PREPROCESSING START ===")

    X, y, groups, paths = build_dataset()

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "X": X,
            "y": y,
            "classes": CLASSES,
            "groups": groups,
            "paths": paths
        }, f)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("Next step: train.py")