"""
Final runs: Experiments 2 + 3, each with 3 different random seeds, reporting mean +/- standard deviation.
(Each seed = a different train/test split + a different model initialization, to check the conclusions are not luck.)

Methods:
  base_logreg / base_cnn   Experiment 2 as is (gain-normalized)
  cmn_logreg  / cmn_cnn    envelope removed (each band minus its own time mean)
  aug_cnn                  envelope augmentation (random tilt + random low-pass during training)

Usage:
    python final_runs.py                # 3 seeds, about 2-3 hours; can run unattended
    python final_runs.py --seeds 0      # a single seed (about 45 min)
Safe to stop: each seed is saved when it finishes, and rerunning skips finished seeds.
Output: <FMC_ROOT>\\results\\final\\  (summary.md, summary.csv, two figures)
"""
import argparse
import json
import time
from pathlib import Path

from paths import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exp1_attribution as e1
import exp2_detection as e2
import exp3_generalize as e3

OUT = ROOT / "results" / "final"
METHODS = [("base", "logreg", "Baseline logistic regression"), ("base", "cnn", "Baseline CNN"),
           ("cmn", "logreg", "Envelope-removed logistic regression"), ("cmn", "cnn", "Envelope-removed CNN"),
           ("aug", "cnn", "Envelope-augmented CNN")]


def run_seed(seed, X, Xc, y, gen, ids, gens, epochs):
    e1.SEED = seed                      # this seed sets both the split and the model initialization
    S = e2.split(ids)
    res = {}
    t = time.time()
    res["base"] = e3.run_all(f"seed {seed} baseline", X, y, gen, S, gens, ["logreg", "cnn"], epochs)
    res["cmn"] = e3.run_all(f"seed {seed} envelope removed", Xc, y, gen, S, gens, ["logreg", "cnn"], epochs)
    orig = e2.fit_cnn
    e2.fit_cnn = e3.fit_cnn_aug
    res["aug"] = e3.run_all(f"seed {seed} envelope augmented", X, y, gen, S, gens, ["cnn"], epochs)
    e2.fit_cnn = orig
    print(f"\n### Seed {seed} done in {(time.time() - t) / 60:.0f} min")
    return res


def collect(all_res, gens):
    """-> {method name: {metric: [value per seed]}}"""
    table = {}
    for v, m, name in METHODS:
        d = {}
        for r in all_res.values():
            rr = r[v]
            d.setdefault("seen_bacc", []).append(rr["all_seen"][m]["balanced_accuracy"])
            d.setdefault("seen_real_kept", []).append(rr["all_seen"][m]["real_kept"])
            dets, aucs = [], []
            for g in gens:
                h = rr[f"holdout_{g}"][m]
                d.setdefault(f"det_{g}", []).append(h[f"detect_{g}"])
                d.setdefault(f"auc_{g}", []).append(h["auc"])
                dets.append(h[f"detect_{g}"]); aucs.append(h["auc"])
            d.setdefault("mean_det", []).append(float(np.mean(dets)))
            d.setdefault("mean_auc", []).append(float(np.mean(aucs)))
        table[name] = d
    return table


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def write_summary(table, gens, n_seeds, n_real):
    lines = [f"# FakeMusicCaps final results ({n_seeds} random seeds, mean +/- SD; {n_real} real clips)", "",
             "## Overview", "",
             "| Method | Seen: balanced acc. | Seen: real kept | Unseen: mean detection | Unseen: mean AUC |",
             "|---|---|---|---|---|"]
    for name, d in table.items():
        lines.append(f"| {name} | {ms(d['seen_bacc'])} | {ms(d['seen_real_kept'])} | "
                     f"{ms(d['mean_det'])} | {ms(d['mean_auc'])} |")
    for key, title in (("det", "Unseen generators: detection rate"), ("auc", "Unseen generators: AUC")):
        lines += ["", f"## {title}", "", "| Method | " + " | ".join(gens) + " |",
                  "|---|" + "---|" * len(gens)]
        for name, d in table.items():
            lines.append(f"| {name} | " + " | ".join(ms(d[f"{key}_{g}"]) for g in gens) + " |")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    rows = ["method,metric,mean,std,values"]
    for name, d in table.items():
        for k, v in d.items():
            rows.append(f"{name},{k},{np.mean(v):.4f},{np.std(v):.4f},{' '.join(f'{x:.4f}' for x in v)}")
    (OUT / "summary.csv").write_text("\n".join(rows), encoding="utf-8-sig")
    print("\n".join(lines))


def plot(table, gens):
    en = {"Baseline logistic regression": "Baseline LogReg", "Baseline CNN": "Baseline CNN", "Envelope-removed logistic regression": "Envelope-removed LogReg",
          "Envelope-removed CNN": "Envelope-removed CNN", "Envelope-augmented CNN": "Envelope-augmented CNN"}
    names = list(table)
    # Figure 1: overview
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "mean_det", "Unseen generators: mean detection rate"),
                           (axes[1], "mean_auc", "Unseen generators: mean AUC")):
        mu = [np.mean(table[n][key]) for n in names]; sd = [np.std(table[n][key]) for n in names]
        ax.bar(range(len(names)), mu, yerr=sd, capsize=4, color="#4C72B0")
        ax.set_xticks(range(len(names)), [en[n] for n in names], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        if key == "mean_auc":
            ax.axhline(0.5, ls="--", c="gray", lw=1)
    fig.tight_layout(); fig.savefig(OUT / "overview.png", dpi=150); plt.close(fig)
    # Figure 2: AUC per generator
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        mu = [np.mean(table[n][f"auc_{g}"]) for g in gens]; sd = [np.std(table[n][f"auc_{g}"]) for g in gens]
        ax.bar(np.arange(len(gens)) + i * w, mu, w, yerr=sd, capsize=2, label=en[n])
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(np.arange(len(gens)) + w * (len(names) - 1) / 2, gens)
    ax.set_ylabel("AUC when this generator is unseen"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "per_generator_auc.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading features (gain-normalized):")
    gens, X, gen, ids = e2.load_all()
    y = (gen != "real").astype(np.int64)
    Xc = e3.cmn(X)
    n_real = int((gen == "real").sum())

    all_res = {}
    for s in seeds:
        f = OUT / f"seed{s}.json"
        if f.exists():
            saved = json.loads(f.read_text())
            if saved.get("n_real") == n_real:
                print(f"Seed {s} already has results ({n_real} real clips), skipping"); all_res[s] = saved["res"]; continue
        print(f"\n######## Starting seed {s} ({seeds.index(s) + 1}/{len(seeds)})  {time.strftime('%H:%M')} ########")
        all_res[s] = run_seed(s, X, Xc, y, gen, ids, gens, args.epochs)
        f.write_text(json.dumps({"n_real": n_real, "res": all_res[s]}, ensure_ascii=False))

    table = collect(all_res, gens)
    print("\n")
    write_summary(table, gens, len(seeds), n_real)
    plot(table, gens)
    print(f"\nResults in {OUT} (summary.md / summary.csv / overview.png / per_generator_auc.png)")


if __name__ == "__main__":
    main()
