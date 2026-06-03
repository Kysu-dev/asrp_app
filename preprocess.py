import os
import pickle
import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from joblib import Parallel, delayed
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

DATASET_DIR = "dataset"
OUTPUT_PATH = "models/features.pkl"
SAMPLE_RATE = 16000
DURATION    = 2.0
N_MFCC      = 20
N_FFT       = 512
HOP_LENGTH  = 160
CLASSES     = [str(i) for i in range(10)]

TARGET_PER_CLASS = 125
N_JOBS           = -1


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    sos = butter(4, cutoff / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def load_and_clean(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = highpass_filter(audio, sr, cutoff=80.0)
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
    
    # Gabungkan MFCC dan Delta-MFCC
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
        
    features = np.concatenate(features) # 240 Fitur (3 chunks * 2 (mean, std) * 40 (mfcc+delta))
    return features.astype(np.float32)


def augment_audio(audio: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    aug = audio.copy()
    variant = seed % 5

    if variant == 0:
        aug = librosa.effects.pitch_shift(aug, sr=SAMPLE_RATE, n_steps=rng.uniform(0.5, 2.0))
    elif variant == 1:
        aug = librosa.effects.pitch_shift(aug, sr=SAMPLE_RATE, n_steps=rng.uniform(-2.0, -0.5))
    elif variant == 2:
        rate = rng.uniform(0.7, 1.4)
        aug  = librosa.effects.time_stretch(aug, rate=rate)
    elif variant == 3:
        # Add strong room noise
        noise = rng.normal(0, 0.015, size=aug.shape)
        aug   = aug + noise
    else:
        aug = librosa.effects.pitch_shift(aug, sr=SAMPLE_RATE, n_steps=rng.uniform(-1.0, 1.0))
        aug = aug + rng.normal(0, 0.01, size=aug.shape)

    aug = librosa.util.normalize(aug)
    
    target_len = int(SAMPLE_RATE * DURATION)
    if len(aug) > target_len:
        aug = aug[:target_len]
        
    return aug.astype(np.float32)


def process_file_aug(fpath: str, label: int, aug_seed: int = -1, group_id: int = -1):
    try:
        audio = load_and_clean(fpath)
        if aug_seed >= 0:
            audio = augment_audio(audio, seed=aug_seed)
        features = extract_features(audio)
        return features, label, group_id
    except Exception:
        return None, None, None


def build_dataset():
    all_tasks = []
    print(f"Target per kelas: {TARGET_PER_CLASS} sampel\n")

    group_counter = 0
    for label in CLASSES:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))])
        random.shuffle(files)
        n = len(files)

        if n >= TARGET_PER_CLASS:
            for fname in files[:TARGET_PER_CLASS]:
                all_tasks.append((os.path.join(folder, fname), int(label), -1, group_counter))
                group_counter += 1
        else:
            for i, fname in enumerate(files):
                all_tasks.append((os.path.join(folder, fname), int(label), -1, group_counter))
                group_counter += 1

            needed = TARGET_PER_CLASS - n
            aug_pool = files * (needed // n + 2)
            
            # Cocokkan grup ID untuk file yang diaugmentasi
            for i, fname in enumerate(aug_pool[:needed]):
                # Cari group id asli dari file ini
                orig_group = -1
                for task in all_tasks:
                    if task[0] == os.path.join(folder, fname) and task[2] == -1:
                        orig_group = task[3]
                        break
                all_tasks.append((os.path.join(folder, fname), int(label), i, orig_group))

    results = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=0)(
        delayed(process_file_aug)(fpath, label, aug_seed, group_id)
        for fpath, label, aug_seed, group_id in tqdm(all_tasks, desc="Extracting features")
    )

    X, y, groups = [], [], []
    for feat, label, group_id in results:
        if feat is not None:
            X.append(feat)
            y.append(label)
            groups.append(group_id)

    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    X, y, groups = build_dataset()
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"X": X, "y": y, "groups": groups, "classes": CLASSES}, f)
    print(f"\nSaved to: {OUTPUT_PATH}")