"""
Experiment 2: real-vs-AI detection, including generators never seen in training (research question D1)

A. Standard setting: train on real vs. all 5 generators, test on the test set (all generators seen).
B. Leave-one-generator-out: each time remove one generator and train on real vs. the other 4;
   at test time, how much of the removed generator's AI music is still detected as fake?
   This mirrors reality: new AI music models keep appearing, and a detector cannot have seen them all.

Details:
  - All inputs are gain-normalized (real music is generally louder; this stops the model judging by volume).
  - The split is still by MusicCaps ID: the real and AI clips of the same caption are always on the same side.
  - There are fewer real clips than AI clips, so training uses class weights; metrics are balanced accuracy and AUC (threshold-free).
  - "Detection rate" = share of fake clips predicted fake; "real kept" = share of real clips predicted real.

Usage:
    python exp2_detection.py                  # logistic regression + CNN, all settings (about 15-20 min)
    python exp2_detection.py --models logreg  # logistic regression only (under a minute, for a first look)
Before running: after downloading more real audio, rerun python extract_features.py to update the real features.
Output: <FMC_ROOT>\\results\\exp2\\
"""
import argparse
import json
from pathlib import Path

from paths import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import exp1_attribution as e1
from exp1b_robustness import gain_normalize

OUT = ROOT / "results" / "exp2"
device = e1.device


def load_all():
    gens = sorted(p.stem for p in e1.FEAT.glob("*.npy") if p.stem != "real")
    X, gen, ids = [], [], []
    for name in gens + ["real"]:
        x = np.load(e1.FEAT / f"{name}.npy")
        X.append(gain_normalize(x)); gen += [name] * len(x)
        ids += json.loads((e1.FEAT / f"{name}_names.json").read_text())
        print(f"  {name:<22}{len(x):>6} clips")
    return gens, np.concatenate(X), np.array(gen), np.array(ids)


def split(ids):
    """Split 70/15/15 by MusicCaps ID; real and AI clips share the same ID split."""
    tr, va, te = e1.split_by_id(ids)
    return {"tr": tr, "va": va, "te": te}


def metrics(y, p, gen, thr=0.5):
    """y: 1 = fake, 0 = real; p: predicted probability of fake."""
    pred = (p >= thr).astype(int)
    out = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
           "real_kept": float((pred[y == 0] == 0).mean())}
    for g in sorted(set(gen[y == 1])):
        out[f"detect_{g}"] = float(pred[gen == g].mean())
    return out


# ---------- Models ----------
def fit_logreg(X, y, tr, te):
    def stats(idx):
        c = X[idx].astype(np.float32)
        return np.concatenate([c.mean(axis=2), c.std(axis=2)], axis=1)
    s_tr = stats(tr)
    sc = StandardScaler().fit(s_tr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(s_tr), y[tr])
    return clf.predict_proba(sc.transform(stats(te)))[:, 1]


def fit_cnn(X, y, tr, va, te, epochs, bs=64):
    torch.manual_seed(e1.SEED)
    sample = X[tr[:: max(1, len(tr) // 3000)]].astype(np.float32)
    mu, sd = float(sample.mean()), float(sample.std())
    Xg = torch.from_numpy(X).to(device)
    yg = torch.from_numpy(y).float().to(device)
    T = {k: torch.from_numpy(v).to(device) for k, v in {"tr": tr, "va": va, "te": te}.items()}
    bx = lambda b: ((Xg[b].float() - mu) / sd).unsqueeze(1)

    model = e1.SmallCNN(1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_fake, n_real = float(y[tr].sum()), float(len(tr) - y[tr].sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_fake, device=device))  # class weighting

    def prob(ix):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(ix), 256):
                out.append(torch.sigmoid(model(bx(ix[s: s + 256])).squeeze(1)))
        return torch.cat(out).cpu().numpy()

    best, state = -1, None
    for ep in range(epochs):
        model.train()
        perm = T["tr"][torch.randperm(len(T["tr"]), device=device)]
        for s in range(0, len(perm), bs):
            b = perm[s: s + bs]
            loss = loss_fn(model(bx(b)).squeeze(1), yg[b])
            opt.zero_grad(); loss.backward(); opt.step()
        v = balanced_accuracy_score(y[va], prob(T["va"]) >= 0.5)
        if v > best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    print(f"    (CNN best validation balanced accuracy {best:.3f})")
    return prob(T["te"])


def run_setting(name, X, y, gen, S, train_gens, test_gens, models, epochs):
    use_tr = S["tr"] & ((gen == "real") | np.isin(gen, train_gens))
    use_va = S["va"] & ((gen == "real") | np.isin(gen, train_gens))
    use_te = S["te"] & ((gen == "real") | np.isin(gen, test_gens))
    tr, va, te = (np.where(m)[0] for m in (use_tr, use_va, use_te))
    print(f"\n=== {name} ===  train {len(tr)} (real {int((y[tr] == 0).sum())})  test {len(te)}")
    res = {}
    for m in models:
        p = fit_logreg(X, y, tr, te) if m == "logreg" else fit_cnn(X, y, tr, va, te, epochs)
        r = metrics(y[te], p, gen[te])
        res[m] = r
        det = "  ".join(f"{k[7:]} {v:.2f}" for k, v in r.items() if k.startswith("detect_"))
        print(f"  [{m:<6}] balanced accuracy {r['balanced_accuracy']:.3f}  AUC {r['auc']:.3f}  "
              f"real kept {r['real_kept']:.2f}  detection: {det}")
    return res


def plot(results, gens, models):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        unseen = [results[f"holdout_{g}"][m][f"detect_{g}"] for g in gens]
        seen = [results["all_seen"][m][f"detect_{g}"] for g in gens]
        x = np.arange(len(gens)) + i * w
        ax.bar(x, seen, w, alpha=0.3, label=f"{m}: seen in training")
        ax.bar(x, unseen, w * 0.6, label=f"{m}: held out (unseen)")
    ax.set_xticks(np.arange(len(gens)) + w * (len(models) - 1) / 2, gens, rotation=20)
    ax.set_ylabel("Detection rate (fake judged as fake)"); ax.set_ylim(0, 1.05)
    ax.set_title("Detecting AI music from seen vs. unseen generators")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "detection_seen_vs_unseen.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg,cnn")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    models = args.models.split(",")
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading features (gain-normalized):")
    gens, X, gen, ids = load_all()
    n_real = int((gen == "real").sum())
    if n_real < 1000:
        print(f"\nWarning: only {n_real} real clips; results will be unstable. Download at least 2,000 before the final run.")
    y = (gen != "real").astype(np.int64)   # 1 = AI-generated, 0 = real
    S = split(ids)

    results = {"n_real": n_real, "generators": gens}
    results["all_seen"] = run_setting("A. Standard: all 5 generators seen", X, y, gen, S, gens, gens, models, args.epochs)
    for g in gens:
        others = [o for o in gens if o != g]
        results[f"holdout_{g}"] = run_setting(f"B. Hold out {g} (unseen in training)", X, y, gen, S,
                                              others, [g], models, args.epochs)

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    plot(results, gens, models)

    print("\n================ Summary: detection rate of AI music (seen -> unseen) ================")
    for m in models:
        print(f"[{m}]  standard-setting balanced accuracy {results['all_seen'][m]['balanced_accuracy']:.3f}")
        for g in gens:
            s = results["all_seen"][m][f"detect_{g}"]
            u = results[f"holdout_{g}"][m][f"detect_{g}"]
            auc = results[f"holdout_{g}"][m]["auc"]
            print(f"   {g:<22}seen {s:.2f}  ->  unseen {u:.2f}   (AUC when held out {auc:.3f})")
    print(f"\nResults and figures in {OUT}")


if __name__ == "__main__":
    main()
