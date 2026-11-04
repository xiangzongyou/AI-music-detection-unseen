"""
Experiment 1b: what does the model actually use to recognize a generator? (control experiments)

Question 1: is it just loudness?
  Generators differ a lot in average level (-20 to -33 dB). On a log spectrogram, changing the volume adds a constant to the whole image.
  Subtracting each clip's own mean therefore removes loudness ("gain normalization").
  If accuracy stays high, the model has learned spectral shape, not volume.

Question 2: in which frequencies is the fingerprint?
  The logistic regression sees only low frequencies (< 2 kHz: mostly melody and bass) or only high frequencies (> 4 kHz: timbre detail and artifacts),
  to see which side gives higher accuracy.

Also plots the average spectrum of each generator (and of real music).

Usage:
    python exp1b_robustness.py
Output: <FMC_ROOT>\\results\\exp1b\\
"""
import json
from pathlib import Path

from paths import ROOT

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exp1_attribution as e1   # reuse Experiment 1's data loading, split and models

OUT = ROOT / "results" / "exp1b"
e1.OUT = OUT
HZ = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)   # center frequency of each mel band

# Experiment 1's report() names confusion-matrix figures by model; use each control's own name here so they do not overwrite each other
CUR = {"tag": ""}
_orig_report = e1.report
e1.report = lambda name, *a: _orig_report(CUR["tag"], *a)


def run(tag, fn, *args):
    CUR["tag"] = tag
    return fn(*args)


def gain_normalize(X):
    """Subtract each clip's own mean (removes loudness differences). Done in chunks to save memory."""
    out = np.empty_like(X)
    for s in range(0, len(X), 2000):
        c = X[s: s + 2000].astype(np.float32)
        out[s: s + 2000] = (c - c.mean(axis=(1, 2), keepdims=True)).astype(np.float16)
    return out


def plot_mean_spectra(classes, X, y, Xn):
    curves = {c: (X[y == k].astype(np.float32).mean(axis=(0, 2)),
                  Xn[y == k].astype(np.float32).mean(axis=(0, 2))) for k, c in enumerate(classes)}
    real = e1.FEAT / "real.npy"
    if real.exists():
        R = np.load(real)
        curves["real"] = (R.astype(np.float32).mean(axis=(0, 2)),
                          gain_normalize(R).astype(np.float32).mean(axis=(0, 2)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for c, (raw, norm) in curves.items():
        style = dict(color="black", lw=2.5) if c == "real" else dict(lw=1.5)
        axes[0].plot(HZ, raw, label=c, **style)
        axes[1].plot(HZ, norm, label=c, **style)
    axes[0].set_title("Mean log-mel spectrum (raw)")
    axes[1].set_title("Mean log-mel spectrum (loudness-normalized)")
    for ax in axes:
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("dB"); ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mean_spectra.png", dpi=150); plt.close(fig)
    print("  saved mean_spectra.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading features:")
    classes, X, y, ids = e1.load_classes()
    tr, va, te = e1.split_by_id(ids)
    Xn = gain_normalize(X)

    print("\nPlotting mean spectra ...")
    plot_mean_spectra(classes, X, y, Xn)

    low, high = HZ < 2000, HZ > 4000
    res = {}
    print("\n===== Logistic regression 1: raw (as in Experiment 1) =====")
    res["logreg_raw"] = run("logreg_raw", e1.run_logreg, X, y, tr, te, classes)
    print("\n===== Logistic regression 2: gain-normalized =====")
    res["logreg_gainnorm"] = run("logreg_gainnorm", e1.run_logreg, Xn, y, tr, te, classes)
    print(f"\n===== Logistic regression 3: gain-normalized, low frequencies only < 2 kHz ({low.sum()} bands) =====")
    res["logreg_gainnorm_low"] = run("logreg_gainnorm_low", e1.run_logreg, Xn[:, low], y, tr, te, classes)
    print(f"\n===== Logistic regression 4: gain-normalized, high frequencies only > 4 kHz ({high.sum()} bands) =====")
    res["logreg_gainnorm_high"] = run("logreg_gainnorm_high", e1.run_logreg, Xn[:, high], y, tr, te, classes)
    print("\n===== CNN: gain-normalized =====")
    res["cnn_gainnorm"] = run("cnn_gainnorm", e1.run_cnn, Xn, y, tr, va, te, classes)

    names = ["logreg_raw", "logreg_gainnorm", "logreg_gainnorm_low", "logreg_gainnorm_high", "cnn_gainnorm"]
    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n================ Summary (test balanced accuracy; chance 0.200) ================")
    labels = {"logreg_raw": "Logistic regression, raw",
              "logreg_gainnorm": "Logistic regression, gain-normalized",
              "logreg_gainnorm_low": "Logistic regression, normalized, low only <2kHz",
              "logreg_gainnorm_high": "Logistic regression, normalized, high only >4kHz",
              "cnn_gainnorm": "CNN, gain-normalized"}
    for k in names:
        print(f"  {labels[k]:<30}{res[k]['balanced_accuracy']:.3f}")
    print(f"\nResults and figures in {OUT}")


if __name__ == "__main__":
    main()
