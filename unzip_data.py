"""
Step 1: extract FakeMusicCaps.zip, skipping macOS __MACOSX files.

Usage:
    python unzip_data.py
By default extracts <FMC_ROOT>\\FakeMusicCaps.zip to <FMC_ROOT>\\data\\fake\\<generator>\\
(for other paths: python unzip_data.py "zip path" "output folder")
Safe to rerun after an interruption: files already extracted are skipped.
"""
import sys
import zipfile
from pathlib import Path

from paths import ROOT

ZIP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "FakeMusicCaps.zip"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "fake"

with zipfile.ZipFile(ZIP) as zf:
    members = [m for m in zf.infolist()
               if not m.is_dir()
               and not m.filename.startswith("__MACOSX")
               and m.filename.lower().endswith(".wav")]
    print(f"Extracting {len(members)} audio files to {OUT}")
    done = skipped = 0
    for i, m in enumerate(members, 1):
        target = OUT / m.filename
        if target.exists() and target.stat().st_size == m.file_size:
            skipped += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            done += 1
        if i % 1000 == 0 or i == len(members):
            print(f"  progress {i}/{len(members)}")

print(f"\nDone: {done} newly extracted, {skipped} already present.")
for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
    print(f"  {d.name:<22}{len(list(d.glob('*.wav'))):>6} wav files   e.g. {next(d.glob('*.wav')).name}")
