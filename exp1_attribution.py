"""
Experiment 1: generator attribution. Given an AI clip, which of the 5 generators made it?

Two models, from simple to complex:
  A. Logistic regression: mean and standard deviation of each band over the 10 s -> 128 x 2 = 256 numbers -> linear classifier.
     Shows how far the simplest statistics go, and the result is easy to interpret (which bands matter most).
  B. Small CNN on the full 128 x 313 spectrogram, 4 convolution blocks, learning its own features.

Key design choice:
  All 5 generators in FakeMusicCaps used the same captions (each MusicCaps ID has one clip in every folder).
  A random split would put clips from the same caption in both training and test sets, so a model could succeed on content rather than generator traces.
  The data are therefore split by caption ID: no test caption is seen in training.

Usage:
    python exp1_attribution.py
Output: results.json and two confusion-matrix figures in <FMC_ROOT>\\results\\exp1\\
"""
import json
import time
from pathlib import Path

from paths import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

FEAT = ROOT / "features"
OUT = ROOT / "results" / "exp1"
SEED = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- 1. Load features ----------
def load_classes():
    names = sorted(p.stem for p in FEAT.glob("*.npy") if p.stem != "real")
    X, y, ids = [], [], []
    for k, name in enumerate(names):
        x = np.load(FEAT / f"{name}.npy")
        stems = json.loads((FEAT / f"{name}_names.json").read_text())
        X.append(x); y += [k] * len(x); ids += stems
        print(f"  {name:<22}{len(x):>6} clips")
    return names, np.concatenate(X), np.array(y), np.array(ids)


# ---------- 2. Split by caption ID: train 70% / validation 15% / test 15% ----------
def split_by_id(ids):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(ids)
    rng.shuffle(uniq)
    n = len(uniq)
    tr_ids = set(uniq[: int(0.7 * n)])
    va_ids = set(uniq[int(0.7 * n): int(0.85 * n)])
    tr = np.array([i in tr_ids for i in ids])
    va = np.array([i in va_ids for i in ids])
    te = ~(tr | va)
    return tr, va, te


def report(name, y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    recall = (cm.diagonal() / cm.sum(axis=1)).round(3).tolist()
    print(f"\n[{name}] test accuracy {acc:.3f}   balanced accuracy {bacc:.3f}")
    for c, r in zip(classes, recall):
        print(f"    {c:<22}recall {r:.3f}")
    plot_cm(cm, classes, name)
    return {"accuracy": acc, "balanced_accuracy": bacc,
            "per_class_recall": dict(zip(classes, recall)), "confusion_matrix": cm.tolist()}


def plot_cm(cm, classes, name):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name}.png", dpi=150); plt.close(fig)


# ---------- 3. Model A: logistic regression ----------
def run_logreg(X, y, tr, te, classes):
    t = time.time()
    stats = np.concatenate([  # (N, 256); computed in chunks to save memory
        np.concatenate([np.concatenate([c.mean(axis=2), c.std(axis=2)], axis=1)
                        for c in [X[s: s + 2000].astype(np.float32)]])
        for s in range(0, len(X), 2000)])
    scaler = StandardScaler().fit(stats[tr])
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(scaler.transform(stats[tr]), y[tr])
    pred = clf.predict(scaler.transform(stats[te]))
    print(f"  (logistic regression took {time.time() - t:.0f} s)")
    return report("logreg", y[te], pred, classes)


# ---------- 4. Model B: small CNN ----------
class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        layers, c_in = [], 1
        for c_out in (16, 32, 64, 128):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                       nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def run_cnn(X, y, tr, va, te, classes, epochs=15, bs=64):
    torch.manual_seed(SEED)
    t = time.time()
    mu, sd = float(X[tr].astype(np.float32).mean()), float(X[tr].astype(np.float32).std())
    Xg = torch.from_numpy(X).to(device)            # everything on the GPU (float16, about 2 GB)
    yg = torch.from_numpy(y).long().to(device)
    idx = {k: torch.from_numpy(np.where(m)[0]).to(device) for k, m in
           {"tr": tr, "va": va, "te": te}.items()}

    def batch_x(b):
        return ((Xg[b].float() - mu) / sd).unsqueeze(1)

    model = SmallCNN(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def predict(ix):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(ix), 256):
                out.append(model(batch_x(ix[s: s + 256])).argmax(1))
        return torch.cat(out).cpu().numpy()

    best, best_state = -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = idx["tr"][torch.randperm(len(idx["tr"]), device=device)]
        total = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s: s + bs]
            loss = loss_fn(model(batch_x(b)), yg[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        va_acc = balanced_accuracy_score(y[va], predict(idx["va"]))
        print(f"  epoch {ep:>2}  train loss {total / len(perm):.4f}  val balanced accuracy {va_acc:.3f}")
        if va_acc > best:
            best, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, OUT / "cnn_best.pt")
    print(f"  (CNN took {time.time() - t:.0f} s; keeping the epoch with the best validation score)")
    return report("cnn", y[te], predict(idx["te"]), classes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading features:")
    classes, X, y, ids = load_classes()
    tr, va, te = split_by_id(ids)
    shared = len(ids) - len(np.unique(ids))
    print(f"\n{len(y)} clips in total; {len(np.unique(ids))} distinct caption IDs"
          f" ({'captions are shared across generators, split by ID' if shared else 'file names do not repeat, equivalent to a random split'})")
    print(f"train {tr.sum()} / val {va.sum()} / test {te.sum()}    device: {device}")

    results = {"classes": classes, "n_train": int(tr.sum()), "n_val": int(va.sum()),
               "n_test": int(te.sum()), "split": "by caption id, seed 0"}
    print("\n===== Model A: logistic regression (band mean + std) =====")
    results["logreg"] = run_logreg(X, y, tr, te, classes)
    print("\n===== Model B: small CNN (full log-mel spectrogram) =====")
    results["cnn"] = run_cnn(X, y, tr, va, te, classes)

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {OUT}")
    print(f"Summary: logistic regression {results['logreg']['balanced_accuracy']:.3f}, "
          f"CNN {results['cnn']['balanced_accuracy']:.3f} (test balanced accuracy; chance is 0.200)")


if __name__ == "__main__":
    main()
