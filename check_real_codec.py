"""
Check which audio stream YouTube serves for the real MusicCaps clips, using the same
format selector as download_musiccaps.py. Nothing is downloaded (metadata only).

Usage:
    python check_real_codec.py --browser firefox            # 40 random real clips
    python check_real_codec.py --browser firefox --n 80
Output: printed table + <FMC_ROOT>\\results\\codec\\real_codec_check.csv
"""
import argparse
import collections
import csv
import json
import random
import time

import yt_dlp

from paths import ROOT

FEAT = ROOT / "features"
OUT = ROOT / "results" / "codec"
FMT = "ba[ext=m4a]/ba[acodec^=opus]/ba[acodec!*=iamf]/b[acodec!*=iamf]"  # same as download_musiccaps.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=3)
    args = ap.parse_args()
    ids = json.loads((FEAT / "real_names.json").read_text())
    random.Random(0).shuffle(ids)
    opts = {"quiet": True, "no_warnings": True, "format": FMT, "skip_download": True}
    if args.browser:
        opts["cookiesfrombrowser"] = (args.browser,)
    rows, counts = [], collections.Counter()
    with yt_dlp.YoutubeDL(opts) as ydl:
        for k, yt in enumerate(ids[: args.n], 1):
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={yt}", download=False)
                f = (info.get("requested_formats") or [info])[0]
                r = {"ytid": yt, "format_id": f.get("format_id"), "ext": f.get("ext"), "acodec": f.get("acodec"),
                     "abr_kbps": f.get("abr"), "sample_rate": f.get("asr"), "channels": f.get("audio_channels")}
                counts[(r["ext"], str(r["acodec"]).split(".")[0], r["sample_rate"], round(r["abr_kbps"] or 0))] += 1
            except Exception as e:
                r = {"ytid": yt, "format_id": "ERROR", "ext": "", "acodec": str(e).splitlines()[0][:80],
                     "abr_kbps": "", "sample_rate": "", "channels": ""}
                counts[("error",)] += 1
            rows.append(r)
            print(f"[{k}/{args.n}] {yt}  {r['format_id']}  {r['ext']}  {r['acodec']}  "
                  f"{r['abr_kbps']} kbps  {r['sample_rate']} Hz  {r['channels']} ch", flush=True)
            time.sleep(args.sleep)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "real_codec_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\nSummary (ext, codec, sample rate, kbps): count")
    for key, c in counts.most_common():
        print(f"  {key}: {c}")
    print(f"\nSaved to {OUT / 'real_codec_check.csv'}")


if __name__ == "__main__":
    main()
