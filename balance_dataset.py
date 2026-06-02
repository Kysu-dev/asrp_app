"""
balance_dataset.py
Buat dataset seimbang per kelas. Bisa output ke folder baru
atau langsung menambah file di dataset asli.

Contoh:
    python balance_dataset.py --target 130 --src dataset --dst dataset_balanced
    python balance_dataset.py --target 130 --src dataset --dst dataset
"""

import argparse
import random
import shutil
from pathlib import Path

SUPPORTED_EXT = {".wav", ".mp3", ".ogg", ".flac"}


def collect_files(class_dir: Path) -> list[Path]:
    return sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def balance_class(files: list[Path], target: int, dst_dir: Path, seed: int, in_place: bool) -> int:
    """Copy/duplikasi sampai target. Jika in_place, hanya tambah file duplikat."""
    rng = random.Random(seed)
    count = len(files) if in_place else 0

    if not in_place:
        # Copy semua file asli dulu
        for idx, src in enumerate(files, start=1):
            dst_name = f"{src.stem}_orig_{idx:03d}{src.suffix.lower()}"
            copy_file(src, dst_dir / dst_name)
            count += 1

        # Jika sudah lebih dari target, potong dengan mengambil acak
        if count > target:
            all_files = sorted(dst_dir.glob("*"))
            rng.shuffle(all_files)
            keep = set(all_files[:target])
            for p in all_files:
                if p not in keep:
                    p.unlink(missing_ok=True)
            return target

    if count >= target:
        return count

    # Tambah duplikasi sampai target
    while count < target:
        src = rng.choice(files)
        count += 1
        dst_name = f"{src.stem}_dup_{count:03d}{src.suffix.lower()}"
        copy_file(src, dst_dir / dst_name)

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dataset", help="Folder dataset sumber")
    parser.add_argument("--dst", default="dataset", help="Folder output")
    parser.add_argument("--target", type=int, default=130, help="Jumlah target per kelas")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"Folder sumber tidak ditemukan: {src_root}")

    print(f"Sumber : {src_root}")
    print(f"Output : {dst_root}")
    print(f"Target : {args.target}\n")

    total = 0
    in_place = src_root.resolve() == dst_root.resolve()

    for i in range(10):
        label = str(i)
        src_dir = src_root / label
        dst_dir = dst_root / label

        if not src_dir.exists():
            print(f"[WARN] Folder tidak ada: {src_dir}")
            continue

        files = collect_files(src_dir)
        if not files:
            print(f"[WARN] Tidak ada file di: {src_dir}")
            continue

        final_count = balance_class(files, args.target, dst_dir, args.seed + i, in_place)
        total += final_count
        print(f"Kelas {label}: {len(files)} -> {final_count}")

    print(f"\nTotal file output: {total}")
    print("Selesai.")


if __name__ == "__main__":
    main()
