"""
Compression control: does the detector partly rely on lossy-codec artifacts?

Real clips come from YouTube. download_musiccaps.py prefers the m4a stream, and check_real_codec.py
found that YouTube serves AAC-LC at about 129 kbps, 44.1 kHz stereo (40 of 40 clips checked); ffmpeg then
converts it to 16 kHz mono WAV. AI clips are lossless WAV. Here the AI clips get
the same treatment: 16 kHz -> 44.1 kHz stereo -> AAC 128 kbps -> decode -> 16 kHz mono -> log-mel.

One run does everything, in order; every step is checkpointed and a rerun skips finished parts:
  1. Encode + decode every AI clip and extract log-mel features (no audio copies are kept).
     Conditions: aac128 (matches YouTube), opus128 (YouTube's other audio stream),
                 aac64 (harsher stress test, used at test time only).
  2. Exp A: train baseline / envelope-removed (cmn) CNN on the ORIGINAL lossless AI clips,
     test on original AI clips and on each compressed version (real test clips unchanged).
  3. Exp B: train AND test with aac128 AI clips.
     Exp B turned out to be confounded: every AI clip then carries the same encoder trace and the
     real clips do not. codec_control_c.py fixes this by passing the real clips through the same chain.
  Same protocol as final_runs.py: 3 seeds x (all_seen + 5 held-out generators), 10 epochs,
  threshold 0.5, split grouped by MusicCaps ID.

Usage:
    python codec_control.py               # all steps, 3 seeds (about 1.5-3 hours)
    python codec_control.py --seeds 0     # one seed only
Output:   <FMC_ROOT>\\results\\codec\\  (summary.md, summary.csv, runs.jsonl, encode_failures.txt)
Features: <FMC_ROOT>\\features_codec\\<condition>\\<generator>.npy
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from paths import ROOT
import exp1_attribution as e1
import exp2_detection as e2
import exp3_generalize as e3
import extract_features as ef
from exp1b_robustness import gain_normalize

DATA = ROOT / "data"
FEAT = ROOT / "features"
CFEAT = ROOT / "features_codec"
OUT = ROOT / "results" / "codec"
RUNS = OUT / "runs.jsonl"
CHUNK = 500  # clips per checkpoint file during encoding
FAIL_LOCK = threading.Lock()

# ffmpeg encode settings per condition: (extension, encoder args)
CONDITIONS = {
    "aac128": (".m4a", ["-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "128k"]),
    "opus128": (".ogg", ["-ar", "48000", "-ac", "2", "-c:a", "libopus", "-b:a", "128k"]),
    "aac64": (".m4a", ["-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "64k"]),
}
TRAIN_COND = "aac128"  # condition used for Exp B


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- step 1: encode -> decode -> features ----------------
def codec_roundtrip(wav, cond, tmpdir):
    """Encode one wav with the given codec, decode to 16 kHz mono float32. Returns array or raises."""
    ext, enc = CONDITIONS[cond]
    tmp = Path(tmpdir) / f"{os.getpid()}_{wav.stem}{ext}"
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), *enc, str(tmp)],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError("encode: " + r.stderr.decode(errors="replace")[:200])
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(tmp), "-ac", "1", "-ar", str(ef.SR),
                            "-f", "f32le", "-"], capture_output=True)
        if r.returncode != 0 or len(r.stdout) < 4000:
            raise RuntimeError("decode: " + r.stderr.decode(errors="replace")[:200])
        x = np.frombuffer(r.stdout, dtype=np.float32).copy()
    finally:
        tmp.unlink(missing_ok=True)
    if len(x) < ef.LEN:
        x = np.pad(x, (0, ef.LEN - len(x)))
    return x[: ef.LEN]


def encode_condition(cond, gens, workers, failures):
    d = CFEAT / cond
    d.mkdir(parents=True, exist_ok=True)
    for g in gens:
        final, okf = d / f"{g}.npy", d / f"{g}_ok.npy"
        if final.exists() and okf.exists():
            log(f"[encode {cond}] {g} already done, skipped"); continue
        names = json.loads((FEAT / f"{g}_names.json").read_text())
        parts = d / f"_parts_{g}"
        parts.mkdir(exist_ok=True)
        n_chunks = (len(names) + CHUNK - 1) // CHUNK
        t = time.time()
        with tempfile.TemporaryDirectory() as tmpdir, ThreadPoolExecutor(workers) as pool:
            for k in range(n_chunks):
                pf, po = parts / f"{k:03d}.npy", parts / f"{k:03d}_ok.npy"
                if pf.exists() and po.exists():
                    continue
                chunk = names[k * CHUNK: (k + 1) * CHUNK]

                def one(n):
                    try:
                        return codec_roundtrip(DATA / "fake" / g / f"{n}.wav", cond, tmpdir), True
                    except Exception as ex:
                        with FAIL_LOCK:
                            failures.write(f"{cond}\t{g}\t{n}\t{str(ex)[:200]}\n"); failures.flush()
                        return np.zeros(ef.LEN, np.float32), False
                res = list(pool.map(one, chunk))
                wavs = np.stack([r[0] for r in res]); ok = np.array([r[1] for r in res])
                feats = []
                with torch.no_grad():
                    for s in range(0, len(wavs), 64):
                        m = ef.logmel(torch.from_numpy(wavs[s: s + 64]).to(ef.device))
                        feats.append(m.cpu().numpy().astype(np.float16))
                np.save(pf, np.concatenate(feats)); np.save(po, ok)
                done = min((k + 1) * CHUNK, len(names))
                rate = done / max(time.time() - t, 1e-6)
                log(f"[encode {cond}] {g}  {done}/{len(names)}  failed {int((~ok).sum())}  "
                    f"(about {rate:.0f} clips/s)")
        feats = np.concatenate([np.load(parts / f"{k:03d}.npy") for k in range(n_chunks)])
        ok = np.concatenate([np.load(parts / f"{k:03d}_ok.npy") for k in range(n_chunks)])
        np.save(final, feats); np.save(okf, ok)
        shutil.rmtree(parts, ignore_errors=True)
        orig = np.load(FEAT / f"{g}.npy", mmap_mode="r")
        diff = float(np.abs(orig[:200].astype(np.float32) - feats[:200].astype(np.float32)).mean())
        log(f"[encode {cond}] {g} done: {feats.shape}, failed {int((~ok).sum())}, "
            f"mean difference from the original {diff:.2f} dB (first 200 clips)")


# ---------------- CNN that returns the trained model ----------------
def fit_model(X, y, tr, va, epochs, bs=64):
    """Same training as exp2_detection.fit_cnn, but returns (predict_fn) so we can test on other inputs."""
    torch.manual_seed(e1.SEED)
    sample = X[tr[:: max(1, len(tr) // 3000)]].astype(np.float32)
    mu, sd = float(sample.mean()), float(sample.std())
    Xg = torch.from_numpy(X).to(e1.device)
    yg = torch.from_numpy(y).float().to(e1.device)
    T = {k: torch.from_numpy(v).to(e1.device) for k, v in {"tr": tr, "va": va}.items()}
    model = e1.SmallCNN(1).to(e1.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_fake = float(y[tr].sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr) - n_fake) / n_fake, device=e1.device))

    def prob_gpu(ix):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(ix), 256):
                b = ix[s: s + 256]
                out.append(torch.sigmoid(model(((Xg[b].float() - mu) / sd).unsqueeze(1)).squeeze(1)))
        return torch.cat(out).cpu().numpy()

    best, state = -1, None
    for _ in range(epochs):
        model.train()
        perm = T["tr"][torch.randperm(len(T["tr"]), device=e1.device)]
        for s in range(0, len(perm), bs):
            b = perm[s: s + bs]
            loss = loss_fn(model(((Xg[b].float() - mu) / sd).unsqueeze(1)).squeeze(1), yg[b])
            opt.zero_grad(); loss.backward(); opt.step()
        v = balanced_accuracy_score(y[va], prob_gpu(T["va"]) >= 0.5)
        if v > best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    del Xg
    torch.cuda.empty_cache()

    def predict(Xnp):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(Xnp), 256):
                xb = torch.from_numpy(Xnp[s: s + 256]).to(e1.device).float()
                out.append(torch.sigmoid(model(((xb - mu) / sd).unsqueeze(1)).squeeze(1)))
        return torch.cat(out).cpu().numpy()
    return predict, best


# ---------------- helpers ----------------
class CodecRows:
    """Codec features of one condition, read lazily from disk (memory-mapped), rows in e2.load_all() order."""
    def __init__(self, cond, gens, variant):
        self.mm = [np.load(CFEAT / cond / f"{g}.npy", mmap_mode="r") for g in gens]
        self.ok = np.concatenate([np.load(CFEAT / cond / f"{g}_ok.npy") for g in gens])
        self.off = np.cumsum([0] + [len(m) for m in self.mm])
        self.variant = variant

    def rows(self, idx):
        out = np.empty((len(idx), *self.mm[0].shape[1:]), np.float16)
        for j, (a, b) in enumerate(zip(self.off[:-1], self.off[1:])):
            sel = (idx >= a) & (idx < b)
            if sel.any():
                out[sel] = self.mm[j][np.sort(idx[sel]) - a][np.argsort(np.argsort(idx[sel]))]
        out = gain_normalize(out)
        return e3.cmn(out) if self.variant == "cmn" else out

    def full(self):
        return self.rows(np.arange(self.off[-1]))


def done_keys():
    if not RUNS.exists():
        return set()
    return {json.loads(l)["key"] for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()}


def save_run(rec):
    with open(RUNS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def summarize(gens):
    recs = [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = {}  # (variant, train, test) -> metric -> [per-seed values]
    for r in recs:
        for test_cond, m in r["metrics"].items():
            key = (r["variant"], r["train"], test_cond)
            d = rows.setdefault(key, {})
            d.setdefault((r["seed"], r["setting"]), m)
    lines = ["# Compression control", "",
             "train / test = which version of the AI clips was used for training / testing (real clips are never changed here)", "",
             "| Detector | train → test | Seeds | Seen: balanced acc. | Seen: real kept | Unseen: mean detection | Unseen: mean AUC |",
             "|---|---|---|---|---|---|---|"]
    csv = ["variant,train,test,seed,seen_bacc,seen_real_kept,mean_det,mean_auc," +
           ",".join(f"det_{g},auc_{g}" for g in gens)]
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    for (v, trc, tec), d in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] != "orig", kv[0][2] != "orig", kv[0][2])):
        seeds = sorted({s for s, _ in d})
        full = [s for s in seeds if ("all_seen" in {st for ss, st in d if ss == s})
                and all((s, f"holdout_{g}") in d for g in gens)]
        if not full:
            continue
        sb, sk, md, ma = [], [], [], []
        for s in full:
            a = d[(s, "all_seen")]
            dets = [d[(s, f"holdout_{g}")][f"detect_{g}"] for g in gens]
            aucs = [d[(s, f"holdout_{g}")]["auc"] for g in gens]
            sb.append(a["balanced_accuracy"]); sk.append(a["real_kept"]); md.append(np.mean(dets)); ma.append(np.mean(aucs))
            csv.append(f"{v},{trc},{tec},{s},{sb[-1]:.4f},{sk[-1]:.4f},{md[-1]:.4f},{ma[-1]:.4f}," +
                       ",".join(f"{x:.4f},{y_:.4f}" for x, y_ in zip(dets, aucs)))
        name = "Baseline CNN" if v == "base" else "Envelope-removed CNN"
        lines.append(f"| {name} | {trc} → {tec} | {len(full)} | {ms(sb)} | {ms(sk)} | {ms(md)} | {ms(ma)} |")
    # per-generator unseen detection / AUC, averaged over seeds
    lines += ["", "## Unseen generator: detection / AUC (mean over seeds)", "",
              "| Detector | train → test | " + " | ".join(gens) + " |", "|---|---|" + "---|" * len(gens)]
    for (v, trc, tec), d in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] != "orig", kv[0][2] != "orig", kv[0][2])):
        cells = []
        for g in gens:
            vals = [m for (s, st), m in d.items() if st == f"holdout_{g}"]
            cells.append(f"{np.mean([m[f'detect_{g}'] for m in vals]):.2f} / {np.mean([m['auc'] for m in vals]):.3f}"
                         if vals else "-")
        name = "Baseline CNN" if v == "base" else "Envelope-removed CNN"
        lines.append(f"| {name} | {trc} → {tec} | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.csv").write_text("\n".join(csv), encoding="utf-8-sig")
    print("\n" + "\n".join(lines), flush=True)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="base,cmn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found")
    encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    conds = [c for c in CONDITIONS if not (c == "opus128" and "libopus" not in encs)]
    if "opus128" not in conds:
        log("ffmpeg has no libopus, skipping opus128")
    gens = sorted(p.stem for p in FEAT.glob("*.npy") if p.stem != "real")
    log(f"device {ef.device}; generators {gens}; codecs {conds}; workers {args.workers}")

    # ---- step 1 ----
    log("==== Step 1: encode the AI clips and extract features ====")
    with open(OUT / "encode_failures.txt", "a", encoding="utf-8") as failures:
        for c in conds:
            encode_condition(c, gens, args.workers, failures)

    # ---- steps 2 + 3 ----
    log("==== Steps 2 and 3: Exp A (train original, test compressed) + Exp B (train and test compressed) ====")
    log("Loading the original features (loudness-normalized):")
    _, X0, gen, ids = e2.load_all()
    y = (gen != "real").astype(np.int64)
    n_ai = int((gen != "real").sum())
    assert (gen[:n_ai] != "real").all(), "AI rows are expected first"
    settings = [("all_seen", gens, gens)] + [(f"holdout_{g}", [o for o in gens if o != g], [g]) for g in gens]
    done = done_keys()
    total = len(variants) * len(seeds) * len(settings) * 2
    k = 0

    for v in variants:
        need = [(exp, s, st) for s in seeds for st in settings for exp in ("A", "B")
                if f"{exp}|{v}|{s}|{st[0]}" not in done]
        if not need:
            k += len(seeds) * len(settings) * 2
            log(f"[{v}] all done, skipped"); continue
        if v == "base":
            Xv = X0
        else:
            Xv = e3.cmn(X0)
            X0 = None  # free memory; base variant is done by now
        codec = {c: CodecRows(c, gens, v) for c in conds}
        okall = np.ones(len(gen), bool)
        for c in conds:
            okall[:n_ai] &= codec[c].ok
        XB = Xv.copy()
        XB[:n_ai] = codec[TRAIN_COND].full()
        log(f"[{v}] features ready; AI clips encoded in every condition: {int(okall[:n_ai].sum())}/{n_ai}")

        for s in seeds:
            e1.SEED = s
            S = e2.split(ids)
            for name, train_gens, test_gens in settings:
                m_tr = S["tr"] & ((gen == "real") | np.isin(gen, train_gens)) & okall
                m_va = S["va"] & ((gen == "real") | np.isin(gen, train_gens)) & okall
                m_te = S["te"] & ((gen == "real") | np.isin(gen, test_gens)) & okall
                tr, va, te = (np.where(m)[0] for m in (m_tr, m_va, m_te))
                ai_te = te[te < n_ai]; real_te = te[te >= n_ai]
                for exp in ("A", "B"):
                    k += 1
                    key = f"{exp}|{v}|{s}|{name}"
                    if key in done:
                        continue
                    t = time.time()
                    try:
                        Xtrain = Xv if exp == "A" else XB
                        predict, best = fit_model(Xtrain, y, tr, va, args.epochs)
                        p_real = predict(Xv[real_te])
                        tests = ["orig", *conds] if exp == "A" else [TRAIN_COND, "orig"]
                        metrics = {}
                        for tc in tests:
                            Xai = Xv[ai_te] if tc == "orig" else codec[tc].rows(ai_te)
                            p = np.concatenate([predict(Xai), p_real])
                            idx = np.concatenate([ai_te, real_te])
                            metrics[tc] = e2.metrics(y[idx], p, gen[idx])
                        rec = {"key": key, "exp": exp, "variant": v, "seed": s, "setting": name,
                               "train": "orig" if exp == "A" else TRAIN_COND, "val_bacc": best,
                               "metrics": metrics}
                        save_run(rec)
                        msg = "  ".join(f"{tc}: bacc {m['balanced_accuracy']:.3f} AUC {m['auc']:.3f}"
                                        + (f" detection {m[f'detect_{test_gens[0]}']:.2f}" if len(test_gens) == 1 else "")
                                        for tc, m in metrics.items())
                        log(f"[{k}/{total}] Exp {exp} {v} seed {s} {name}  ({(time.time() - t) / 60:.1f} min)\n      {msg}")
                    except Exception:
                        log(f"[{k}/{total}] Exp {exp} {v} seed {s} {name} failed, skipped:\n{traceback.format_exc()}")
        del Xv, XB, codec
        gc.collect()
    log(f"==== All done in {(time.time() - t0) / 60:.0f} min ====")
    summarize(gens)
    print(f"\nResults in {OUT} (summary.md / summary.csv / runs.jsonl)", flush=True)


if __name__ == "__main__":
    main()
