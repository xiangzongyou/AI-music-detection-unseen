"""
Step 3: turn every clip into a log-mel spectrogram and save it; every later experiment uses these.

What is a log-mel spectrogram?
  The 10 s clip is cut into short frames (1,024 samples each, about 64 ms, moving 512 samples at a time).
  A Fourier transform of each frame gives how loud each frequency is at that moment.
  Frequencies are merged into 128 bands on the mel scale (spaced as human hearing is), then converted to decibels.
  The result is a 128 (frequency) x 313 (time) image: time across, pitch up, brightness = loudness.
  AI generators often leave traces in this image that listeners cannot hear (e.g. missing high frequencies, regular textures).

Usage:
    python extract_features.py --limit 50    # only 50 clips per class, to test that it runs
    python extract_features.py               # everything (5 generators + the real audio downloaded so far)
After downloading more real audio, run it again to update the real class; classes already computed are skipped.

Output: <FMC_ROOT>\\features\\<class>.npy (shape N x 128 x 313, float16)
        and <class>_names.json (the file name for each row)
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from paths import ROOT

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

DATA = ROOT / "data"
FEAT = ROOT / "features"

SR = 16000          # sample rate: 16,000 samples per second (same as the dataset)
SECONDS = 10
LEN = SR * SECONDS  # every clip becomes 160,000 samples
N_FFT = 1024        # length of each frame
HOP = 512           # how far the frame moves each step
N_MELS = 128        # number of mel bands

device = "cuda" if torch.cuda.is_available() else "cpu"
MEL_FB = torch.tensor(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS), device=device)
WINDOW = torch.hann_window(N_FFT, device=device)


def load_wav(path):
    """Read a wav as mono, 16 kHz, exactly 10 s (zero-padded if shorter, cut if longer)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    if len(x) < LEN:
        x = np.pad(x, (0, LEN - len(x)))
    return x[:LEN]


def logmel(batch):
    """batch: (B, 160000) -> (B, 128, 313) log-mel spectrogram in dB. Computed on the GPU."""
    spec = torch.stft(batch, N_FFT, HOP, window=WINDOW, return_complex=True).abs() ** 2
    mel = MEL_FB @ spec
    return 10 * torch.log10(mel + 1e-10)


def class_folders():
    folders = {}
    fake_root = DATA / "fake"
    for d in sorted(fake_root.iterdir()) if fake_root.exists() else []:
        if d.is_dir():
            folders[d.name] = d
    if (DATA / "real").exists():
        folders["real"] = DATA / "real"
    return folders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips per class (for testing)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    folders = class_folders()
    if not folders:
        raise SystemExit(f"No data found under {DATA}; run unzip_data.py first")

    for name, folder in folders.items():
        files = sorted(folder.glob("*.wav"))
        if args.limit:
            files = files[: args.limit]
        out_npy, out_names = FEAT / f"{name}.npy", FEAT / f"{name}_names.json"
        if out_names.exists() and len(json.loads(out_names.read_text())) == len(files):
            print(f"[{name}] already has {len(files)} clips, skipping")
            continue

        n_frames = 1 + LEN // HOP
        feats = np.zeros((len(files), N_MELS, n_frames), dtype=np.float16)
        with ThreadPoolExecutor(8) as pool:
            for start in tqdm(range(0, len(files), args.batch), desc=name):
                chunk = files[start: start + args.batch]
                wavs = np.stack(list(pool.map(load_wav, chunk)))
                with torch.no_grad():
                    m = logmel(torch.from_numpy(wavs).to(device))
                feats[start: start + len(chunk)] = m.cpu().numpy().astype(np.float16)

        np.save(out_npy, feats)
        out_names.write_text(json.dumps([f.stem for f in files]))
        print(f"[{name}] done: {feats.shape}, mean level {feats.astype(np.float32).mean():.1f} dB")

    print("\nAll done. Features are in", FEAT)


if __name__ == "__main__":
    main()
