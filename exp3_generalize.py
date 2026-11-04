"""
Experiment 3: can the detector be made more useful on generators it has never seen?

Experiment 2 found that detectors fail badly on unseen generators (mean detection ~0.3; MusicLDM's AUC is even below 0.5).
Hypothesis: the detector mostly memorizes each generator's spectral envelope (overall EQ curve, high-frequency cutoff) and fails when the envelope changes.
Two interventions, both forcing the model to ignore the envelope:
  1. cmn (envelope removal): subtract each band's own mean over the 10 s, leaving only texture that changes over time.
  2. aug (envelope augmentation, CNN only): during training, add a random spectral tilt and randomly cut part of the high frequencies,
     so the envelope becomes an unreliable cue; nothing is changed at test time.
Runs the standard setting + 5 leave-one-out settings, compared with Experiment 2.

Usage:
    python exp3_generalize.py                 # both interventions (about 25-30 min)
    python exp3_generalize.py --variants cmn  # envelope removal only
Output: <FMC_ROOT>\\results\\exp3\\
"""
import argparse
import json
from pathlib import Path

from paths import ROOT

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

import exp1_attribution as e1
import exp2_detection as e2

OUT = ROOT / "results" / "exp3"
device = e1.device


def cmn(X):
    """Subtract each band's own time average in every clip (in chunks to save memory)."""
    out = np.empty_like(X)
    for s in range(0, len(X), 2000):
        c = X[s: s + 2000].astype(np.float32)
        out[s: s + 2000] = (c - c.mean(axis=2, keepdims=True)).astype(np.float16)
    return out


def augment(x):
    """x: (B,1,128,T) in dB. Random tilt of +/-10 dB; for half the clips, a random low-pass (bands above a cutoff attenuated by 10-30 dB)."""
    B, _, F, _ = x.shape
    ramp = torch.linspace(-0.5, 0.5, F, device=x.device).view(1, 1, F, 1)
    x = x + (torch.rand(B, 1, 1, 1, device=x.device) * 20 - 10) * ramp
    cut = torch.randint(80, 128, (B, 1, 1, 1), device=x.device)
    band = torch.arange(F, device=x.device).view(1, 1, F, 1)
    atten = (torch.rand(B, 1, 1, 1, device=x.device) * 20 + 10) * (band >= cut)
    use = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).float()
    return x - atten * use


def fit_cnn_aug(X, y, tr, va, te, epochs, bs=64):
    """Same CNN as Experiment 2, but training batches go through augment() first."""
    torch.manual_seed(e1.SEED)
    sample = X[tr[:: max(1, len(tr) // 3000)]].astype(np.float32)
    mu, sd = float(sample.mean()), float(sample.std())
    Xg = torch.from_numpy(X).to(device)
    yg = torch.from_numpy(y).float().to(device)
    T = {k: torch.from_numpy(v).to(device) for k, v in {"tr": tr, "va": va, "te": te}.items()}
    raw = lambda b: Xg[b].float().unsqueeze(1)
    norm = lambda x: (x - mu) / sd

    model = e1.SmallCNN(1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_fake = float(y[tr].sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr) - n_fake) / n_fake, device=device))

    def prob(ix):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(ix), 256):
                out.append(torch.sigmoid(model(norm(raw(ix[s: s + 256]))).squeeze(1)))
        return torch.cat(out).cpu().numpy()

    best, state = -1, None
    for _ in range(epochs):
        model.train()
        perm = T["tr"][torch.randperm(len(T["tr"]), device=device)]
        for s in range(0, len(perm), bs):
            b = perm[s: s + bs]
            loss = loss_fn(model(norm(augment(raw(b)))).squeeze(1), yg[b])
            opt.zero_grad(); loss.backward(); opt.step()
        v = balanced_accuracy_score(y[va], prob(T["va"]) >= 0.5)
        if v > best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    print(f"    (CNN+aug best validation balanced accuracy {best:.3f})")
    return prob(T["te"])


def run_all(tag, X, y, gen, S, gens, models, epochs):
    res = {"all_seen": e2.run_setting(f"[{tag}] A. standard", X, y, gen, S, gens, gens, models, epochs)}
    for g in gens:
        res[f"holdout_{g}"] = e2.run_setting(f"[{tag}] B. hold out {g}", X, y, gen, S,
                                             [o for o in gens if o != g], [g], models, epochs)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="cmn,aug")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    variants = args.variants.split(",")
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading features (gain-normalized):")
    gens, X, gen, ids = e2.load_all()
    y = (gen != "real").astype(np.int64)
    S = e2.split(ids)
    results = {"n_real": int((gen == "real").sum())}

    if "cmn" in variants:
        results["cmn"] = run_all("envelope removed", cmn(X), y, gen, S, gens, ["logreg", "cnn"], args.epochs)
    if "aug" in variants:
        orig = e2.fit_cnn
        e2.fit_cnn = fit_cnn_aug          # make "cnn" in run_setting use the augmented version
        results["aug"] = run_all("envelope augmented", X, y, gen, S, gens, ["cnn"], args.epochs)
        e2.fit_cnn = orig
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    base_path = e2.OUT / "results.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else None
    rows = []
    if base:
        rows += [("Exp2 LogReg", base, "logreg"), ("Exp2 CNN", base, "cnn")]
    if "cmn" in results:
        rows += [("Env-removed LogReg", results["cmn"], "logreg"), ("Env-removed CNN", results["cmn"], "cnn")]
    if "aug" in results:
        rows += [("Env-augmented CNN", results["aug"], "cnn")]

    print("\n================ Summary: detection rate / AUC on unseen generators ================")
    print(f"{'Method':<20}{'Seen bal.acc':>12}  " + "  ".join(f"{g[:10]:>10}" for g in gens) + "     mean det  mean AUC")
    for name, r, m in rows:
        det = [r[f"holdout_{g}"][m][f"detect_{g}"] for g in gens]
        auc = [r[f"holdout_{g}"][m]["auc"] for g in gens]
        print(f"{name:<20}{r['all_seen'][m]['balanced_accuracy']:>12.3f}      "
              + "  ".join(f"{d:.2f}/{a:.2f}" for d, a in zip(det, auc))
              + f"     {np.mean(det):.3f}    {np.mean(auc):.3f}")
    print("(each cell: detection rate/AUC. AUC 0.5 = chance, < 0.5 = the decision is inverted)")
    print(f"\nResults in {OUT}")


if __name__ == "__main__":
    main()
