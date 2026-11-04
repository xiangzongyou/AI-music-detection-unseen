"""
Step 0: check what is inside FakeMusicCaps.zip (without extracting it) and check the computer setup.

Usage (from the command line):
    python check_dataset.py  "D:\\data\\FakeMusicCaps.zip"

It prints:
  1. how many audio files each folder in the archive holds and how much space they take
  2. whether any folder looks like real audio / MusicCaps
  3. your Python version, whether PyTorch is installed, and whether an NVIDIA GPU is available
Copy the whole output back into the chat.
"""
import sys
import zipfile
import platform
import shutil
from collections import defaultdict
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def inspect_zip(zip_path: Path):
    print(f"== Archive: {zip_path}  ({zip_path.stat().st_size / 1e9:.1f} GB)\n")
    counts = defaultdict(int)
    sizes = defaultdict(int)
    others = defaultdict(int)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            folder = str(p.parent) if str(p.parent) != "." else "(root)"
            if p.suffix.lower() in AUDIO_EXT:
                counts[folder] += 1
                sizes[folder] += info.file_size
            else:
                others[info.filename] += 1

    print(f"{'Folder':<50}{'Files':>10}{'Unzipped':>14}")
    print("-" * 74)
    for folder in sorted(counts):
        print(f"{folder:<50}{counts[folder]:>10}{sizes[folder] / 1e9:>11.2f} GB")
    print("-" * 74)
    print(f"{'Total':<50}{sum(counts.values()):>10}{sum(sizes.values()) / 1e9:>11.2f} GB\n")

    if others:
        print("== Non-audio files (possibly labels, split lists or documentation):")
        for name in sorted(others)[:30]:
            print("   ", name)
        if len(others) > 30:
            print(f"    ... and {len(others) - 30} more")
        print()

    keywords = ("real", "musiccaps", "original", "gt", "bonafide")
    real_like = [f for f in counts if any(k in f.lower() for k in keywords)]
    print("== Folders that look like real audio:", real_like if real_like else "none found (MusicCaps probably has to be downloaded from YouTube separately)")
    print()


def inspect_env(zip_path: Path):
    print("== Computer setup")
    print("   Python:", platform.python_version(), "|", platform.system(), platform.release())
    free = shutil.disk_usage(zip_path.parent).free / 1e9
    print(f"   Free space on the archive's disk: {free:.0f} GB (extraction needs about 15 GB more)")
    try:
        import torch
        print("   PyTorch:", torch.__version__)
        if torch.cuda.is_available():
            print("   GPU:", torch.cuda.get_device_name(0))
        else:
            print("   GPU: no usable CUDA GPU found (CPU will be used; enough for small models)")
    except ImportError:
        print("   PyTorch: not installed (install it later)")
    for lib in ("numpy", "librosa", "sklearn", "soundfile"):
        try:
            __import__(lib)
            print(f"   {lib}: installed")
        except ImportError:
            print(f"   {lib}: not installed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    zp = Path(sys.argv[1])
    if not zp.exists():
        sys.exit(f"File not found: {zp}")
    inspect_zip(zp)
    inspect_env(zp)
