"""
merge_dataset.py

Gabungkan dataset dari beberapa folder speaker ke dalam satu folder dataset/.
Rename semua file secara konsisten: {speaker}_{kelas}_{index:03d}.wav

Struktur INPUT (taruh script ini sejajar dengan folder dataset_*):
    dataset_farras/
        0/  atau  00/  (nama folder angka bebas)
            voice001.m4a
            rekam_2.wav
            ...
        01/
        ...
    dataset_felix/
        0/
        ...
    Dataset_rael/
        0/
        ...

Struktur OUTPUT:
    dataset/
        0/
            farras_0_001.wav
            farras_0_002.wav
            felix_0_001.wav
            ...
        1/
        ...
        9/

Jalankan:
    pip install pydub tqdm
    python merge_dataset.py
"""

import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False
    print("[WARN] pydub tidak terinstall. File non-WAV tidak akan dikonversi.")
    print("       Install: pip install pydub")
    print("       Juga butuh ffmpeg: https://ffmpeg.org/download.html\n")

# ── Konfigurasi ───────────────────────────────────────────────
OUTPUT_DIR = "dataset"  # folder output yang sudah kamu buat

# Daftarkan folder speaker + nama pendek untuk penamaan file
# Format: { "nama_folder_asli": "nama_pendek" }
SPEAKERS = {
    "dataset_farras": "farras",
    "dataset_felix" : "felix",
    "dataset_rael"  : "rael",
    "dataset_rifki" : "rifki",
}

# Format audio yang didukung (termasuk mp4a)
SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".mp4a", ".ogg", ".flac", ".aac", ".opus", ".mp4"}

# Mapping nama folder angka → label kelas (0-9)
# Handles: "0", "00", "01", "1", "02", "2", dst
def folder_to_label(folder_name: str) -> str | None:
    """Konversi nama folder (0, 00, 01, 1, ...) ke label kelas (0-9)."""
    stripped = folder_name.strip().lstrip("0") or "0"
    if stripped.isdigit() and 0 <= int(stripped) <= 9:
        return stripped
    return None
# ─────────────────────────────────────────────────────────────


def convert_to_wav(src_path: Path, dst_path: Path) -> bool:
    """Konversi file audio apapun ke WAV 16kHz mono."""
    ext = src_path.suffix.lower()

    if ext == ".wav" and not PYDUB_OK:
        # Langsung copy kalau sudah WAV dan pydub tidak ada
        shutil.copy2(src_path, dst_path)
        return True

    if not PYDUB_OK:
        # Non-WAV tapi pydub tidak ada → skip
        return False

    try:
        fmt = ext.lstrip(".")
        # Semua varian m4a/mp4a → pakai format mp4 di pydub
        if fmt in ("m4a", "mp4a"):
            fmt = "mp4"

        audio = AudioSegment.from_file(str(src_path), format=fmt)
        audio = audio.set_frame_rate(16000).set_channels(1)  # 16kHz mono
        audio.export(str(dst_path), format="wav")
        return True
    except Exception as e:
        print(f"  [ERROR] Gagal konversi {src_path.name}: {e}")
        return False


def merge():
    print("=== Merge & Normalize Dataset ===\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Buat subfolder 0-9 di output
    for i in range(10):
        os.makedirs(os.path.join(OUTPUT_DIR, str(i)), exist_ok=True)

    total_ok    = 0
    total_skip  = 0
    total_err   = 0

    # Counter per kelas untuk index file
    class_counter = {str(i): {} for i in range(10)}  # {kelas: {speaker: count}}

    for folder_name, speaker_id in SPEAKERS.items():
        speaker_path = Path(folder_name)

        if not speaker_path.exists():
            print(f"[WARN] Folder '{folder_name}' tidak ditemukan, dilewati.\n")
            continue

        print(f"Speaker: {speaker_id} ({folder_name})")

        # Iterasi subfolder angka
        subfolders = sorted(speaker_path.iterdir())
        for subfolder in subfolders:
            if not subfolder.is_dir():
                continue

            label = folder_to_label(subfolder.name)
            if label is None:
                print(f"  [SKIP] Folder '{subfolder.name}' bukan angka 0-9")
                continue

            # Inisialisasi counter
            if speaker_id not in class_counter[label]:
                class_counter[label][speaker_id] = 0

            # Ambil semua file audio
            audio_files = sorted([
                f for f in subfolder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
            ])

            print(f"  Kelas '{label}': {len(audio_files)} file", end="")

            ok_count = 0
            for audio_file in tqdm(audio_files, desc=f"    Proses", leave=False):
                class_counter[label][speaker_id] += 1
                idx = class_counter[label][speaker_id]

                # Format nama: {speaker}_{kelas}_{index:03d}.wav
                out_name = f"{speaker_id}_{label}_{idx:03d}.wav"
                out_path = Path(OUTPUT_DIR) / label / out_name

                success = convert_to_wav(audio_file, out_path)
                if success:
                    ok_count += 1
                    total_ok += 1
                else:
                    total_skip += 1

            print(f" → {ok_count} berhasil")

        print()

    # Ringkasan
    print("=" * 45)
    print(f"Selesai!")
    print(f"  Berhasil   : {total_ok} file")
    print(f"  Dilewati   : {total_skip} file")
    print(f"  Error      : {total_err} file")
    print()
    print("Distribusi per kelas:")
    for i in range(10):
        label = str(i)
        out_folder = Path(OUTPUT_DIR) / label
        count = len(list(out_folder.glob("*.wav")))
        speakers_info = ", ".join([
            f"{spk}:{cnt}" for spk, cnt in class_counter[label].items()
        ])
        print(f"  Kelas {label}: {count:3d} file  [{speakers_info}]")

    print(f"\nOutput: ./{OUTPUT_DIR}/")
    print("Selanjutnya jalankan: python preprocess.py")


if __name__ == "__main__":
    merge()