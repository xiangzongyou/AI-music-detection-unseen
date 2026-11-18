"""
Compression control, experiment C: put BOTH classes through the same codec chain.

Experiment B (AI clips re-encoded, real clips untouched) may have created a new shortcut:
every AI clip carries the trace of our own ffmpeg AAC encoder, the real clips do not.
Here the real clips also go through the identical chain
(16 kHz -> 44.1 kHz stereo -> AAC 128 kbps -> decode -> 16 kHz mono), so encoder traces no
longer separate the classes. The real clips were already AAC from YouTube, so they are encoded
twice and the AI clips once, but both classes now carry the same encoder trace. Then the leave-one-generator-out runs are repeated.

Needs codec_control.py to have finished (uses features_codec/aac128/<generator>.npy).
Checkpointed like codec_control.py: a rerun skips finished parts.

Usage:
    python codec_control_c.py             # about 45 minutes
Output: <FMC_ROOT>\\results\\codec\\  (runs_c.jsonl, summary_c.md)
"""
import argparse
import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import exp1_attribution as e1
import exp2_detection as e2
import exp3_generalize as e3
import extract_features as ef
from exp1b_robustness import gain_normalize
import codec_control as cc

COND = "aac128"
RUNS_C = cc.OUT / "runs_c.jsonl"


def encode_real(workers):
    """Encode + decode every real clip with the aac128 chain; save features and ok mask."""
    d = cc.CFEAT / COND
    final, okf = d / "real.npy", d / "real_ok.npy"
    if final.exists() and okf.exists():
        cc.log("[encode real] already done, skipped"); return
    names = json.loads((cc.FEAT / "real_names.json").read_text())
    feats = np.zeros((len(names), ef.N_MELS, 1 + ef.LEN // ef.HOP), np.float16)
    ok = np.ones(len(names), bool)
    fails = open(cc.OUT / "encode_failures.txt", "a", encoding="utf-8")
    t = time.time()
    with tempfile.TemporaryDirectory() as tmpdir, ThreadPoolExecutor(workers) as pool:
        def one(n):
            try:
                return cc.codec_roundtrip(cc.DATA / "real" / f"{n}.wav", COND, tmpdir), True
            except Exception as ex:
                with cc.FAIL_LOCK:
                    fails.write(f"{COND}\treal\t{n}\t{str(ex)[:200]}\n"); fails.flush()
                return np.zeros(ef.LEN, np.float32), False
        for s in range(0, len(names), 500):
            res = list(pool.map(one, names[s: s + 500]))
            wavs = np.stack([r[0] for r in res])
            ok[s: s + len(res)] = [r[1] for r in res]
            with torch.no_grad():
                for b in range(0, len(wavs), 64):
                    m = ef.logmel(torch.from_numpy(wavs[b: b + 64]).to(ef.device))
                    feats[s + b: s + b + len(m)] = m.cpu().numpy().astype(np.float16)
            cc.log(f"[encode real] {min(s + 500, len(names))}/{len(names)}  failed {int((~ok).sum())}  "
                   f"({(time.time() - t) / 60:.1f} min)")
    fails.close()
    np.save(final, feats); np.save(okf, ok)
    orig = np.load(cc.FEAT / "real.npy", mmap_mode="r")
    diff = float(np.abs(orig[:200].astype(np.float32) - feats[:200].astype(np.float32)).mean())
    cc.log(f"[encode real] done: {feats.shape}, failed {int((~ok).sum())}, mean difference from the original {diff:.2f} dB (first 200 clips)")


def done_keys():
    if not RUNS_C.exists():
        return set()
    return {json.loads(l)["key"] for l in RUNS_C.read_text(encoding="utf-8").splitlines() if l.strip()}


def per_seed(path, keep):
    """Read a runs file -> {(variant, test_cond): {seed: {setting: metrics}}} for records passing keep()."""
    out = {}
    if not path.exists():
        return out
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if not keep(r):
            continue
        for tc, m in r["metrics"].items():
            out.setdefault((r["variant"], tc), {}).setdefault(r["seed"], {})[r["setting"]] = m
    return out


def row(name, d, gens):
    seeds = [s for s, v in d.items() if "all_seen" in v and all(f"holdout_{g}" in v for g in gens)]
    if not seeds:
        return None
    sb = [d[s]["all_seen"]["balanced_accuracy"] for s in seeds]
    rk = [d[s]["all_seen"]["real_kept"] for s in seeds]
    md = [np.mean([d[s][f"holdout_{g}"][f"detect_{g}"] for g in gens]) for s in seeds]
    ma = [np.mean([d[s][f"holdout_{g}"]["auc"] for g in gens]) for s in seeds]
    per = " | ".join(f"{np.mean([d[s][f'holdout_{g}'][f'detect_{g}'] for s in seeds]):.2f} / "
                     f"{np.mean([d[s][f'holdout_{g}']['auc'] for s in seeds]):.3f}" for g in gens)
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    return f"| {name} | {len(seeds)} | {ms(sb)} | {ms(rk)} | {ms(md)} | {ms(ma)} | {per} |"


def summarize(gens):
    lines = ["# Compression control, Exp C: real and AI clips through the same codec chain", "",
             "| Detector / condition | Seeds | Seen: balanced acc. | Seen: real kept | Unseen: mean detection | Unseen: mean AUC | "
             + " | ".join(f"{g} det./AUC" for g in gens) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(gens)]
    old = per_seed(cc.RUNS, lambda r: True)
    a = per_seed(cc.RUNS, lambda r: r["exp"] == "A")
    b = per_seed(cc.RUNS, lambda r: r["exp"] == "B")
    c = per_seed(RUNS_C, lambda r: True)
    for v, vn in (("base", "Baseline CNN"), ("cmn", "Envelope-removed CNN")):
        for name, d in ((f"{vn}, no compression (Exp A baseline)", a.get((v, "orig"))),
                        (f"{vn}, only AI compressed (Exp B, confounded)", b.get((v, COND))),
                        (f"{vn}, both compressed (Exp C)", c.get((v, "both"))),
                        (f"{vn}, Exp C model tested on original AI + compressed real", c.get((v, "ai_orig"))),
                        (f"{vn}, Exp C model tested on compressed AI + original real", c.get((v, "real_orig")))):
            r = row(name, d, gens) if d else None
            if r:
                lines.append(r)
    (cc.OUT / "summary_c.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="base,cmn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    t0 = time.time()
    gens = sorted(p.stem for p in cc.FEAT.glob("*.npy") if p.stem != "real")
    for g in gens:
        if not (cc.CFEAT / COND / f"{g}.npy").exists():
            raise SystemExit(f"missing {COND}/{g}.npy, run codec_control.py first")
    cc.log(f"device {ef.device}; workers {args.workers}")

    cc.log("==== Step 1: encode the real clips and extract features ====")
    encode_real(args.workers)

    cc.log("==== Step 2: Exp C (train and test with both classes compressed) ====")
    _, X0, gen, ids = e2.load_all()
    y = (gen != "real").astype(np.int64)
    n_ai = int((gen != "real").sum())
    settings = [("all_seen", gens, gens)] + [(f"holdout_{g}", [o for o in gens if o != g], [g]) for g in gens]
    done = done_keys()
    total = len(variants) * len(seeds) * len(settings)
    k = 0
    for v in variants:
        if all(f"C|{v}|{s}|{st[0]}" in done for s in seeds for st in settings):
            k += len(seeds) * len(settings); cc.log(f"[{v}] all done, skipped"); continue
        ai = cc.CodecRows(COND, gens, v)
        real_c = gain_normalize(np.load(cc.CFEAT / COND / "real.npy"))
        if v == "cmn":
            real_c = e3.cmn(real_c)
        ok_real = np.load(cc.CFEAT / COND / "real_ok.npy")
        Xorig = X0 if v == "base" else e3.cmn(X0)
        XC = np.empty_like(Xorig)
        XC[:n_ai] = ai.full()
        XC[n_ai:] = real_c
        del real_c
        okall = np.concatenate([ai.ok, ok_real])
        cc.log(f"[{v}] features ready; clips encoded on both sides: {int(okall.sum())}/{len(okall)}")
        for s in seeds:
            e1.SEED = s
            S = e2.split(ids)
            for name, train_gens, test_gens in settings:
                k += 1
                key = f"C|{v}|{s}|{name}"
                if key in done:
                    continue
                m_tr = S["tr"] & ((gen == "real") | np.isin(gen, train_gens)) & okall
                m_va = S["va"] & ((gen == "real") | np.isin(gen, train_gens)) & okall
                m_te = S["te"] & ((gen == "real") | np.isin(gen, test_gens)) & okall
                tr, va, te = (np.where(m)[0] for m in (m_tr, m_va, m_te))
                ai_te, real_te = te[te < n_ai], te[te >= n_ai]
                t = time.time()
                try:
                    predict, best = cc.fit_model(XC, y, tr, va, args.epochs)
                    pa_c, pr_c = predict(XC[ai_te]), predict(XC[real_te])
                    pa_o, pr_o = predict(Xorig[ai_te]), predict(Xorig[real_te])
                    idx = np.concatenate([ai_te, real_te])
                    metrics = {"both": e2.metrics(y[idx], np.concatenate([pa_c, pr_c]), gen[idx]),
                               "ai_orig": e2.metrics(y[idx], np.concatenate([pa_o, pr_c]), gen[idx]),
                               "real_orig": e2.metrics(y[idx], np.concatenate([pa_c, pr_o]), gen[idx])}
                    with open(RUNS_C, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": key, "exp": "C", "variant": v, "seed": s, "setting": name,
                                             "val_bacc": best, "metrics": metrics}) + "\n")
                    m = metrics["both"]
                    det = f"  detection {m[f'detect_{test_gens[0]}']:.2f}" if len(test_gens) == 1 else ""
                    cc.log(f"[{k}/{total}] Exp C {v} seed {s} {name}  bacc {m['balanced_accuracy']:.3f}  "
                           f"AUC {m['auc']:.3f}  real kept {m['real_kept']:.3f}{det}  ({(time.time() - t) / 60:.1f} min)")
                except Exception:
                    cc.log(f"[{k}/{total}] Exp C {v} seed {s} {name} failed, skipped:\n{traceback.format_exc()}")
        del XC, Xorig
    cc.log(f"==== All done in {(time.time() - t0) / 60:.0f} min ====")
    summarize(gens)
    print(f"\nResults in {cc.OUT} (summary_c.md / runs_c.jsonl)", flush=True)


if __name__ == "__main__":
    main()
