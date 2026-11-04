"""
Step 2: download the real MusicCaps music clips from YouTube (the same captions FakeMusicCaps used).

Only the 10-second excerpt is downloaded and converted to the same format as the AI clips: 16 kHz mono WAV.
Saved to <FMC_ROOT>\\data\\real\\<ytid>.wav

Usage:
    python download_musiccaps.py --limit 20     # try 20 clips first to check downloads work
    python download_musiccaps.py                # all 5,521 clips (several hours; can run unattended)
    python download_musiccaps.py --browser firefox   # when YouTube asks you to sign in: use the browser's session
Safe to stop at any time: rerunning skips clips already downloaded and videos confirmed as gone.
Clips blocked by YouTube (sign-in or rate limit) are not marked as missing and are retried next time; 10 blocks in a row stop the run.
Failed downloads are logged in data\\real_failed.csv.

Note: MusicCaps audio comes from YouTube. Use it for local research only and do not redistribute it.
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from paths import ROOT as FMC_ROOT

import yt_dlp
from yt_dlp.utils import download_range_func

ROOT = FMC_ROOT / "data"
REAL = ROOT / "real"
CSV_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"
CSV_PATH = ROOT / "musiccaps-public.csv"
FAILED = ROOT / "real_failed.csv"


def get_list():
    if not CSV_PATH.exists():
        print("Downloading the MusicCaps list ...")
        ROOT.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


PERMANENT_WORDS = ("not available", "unavailable", "private video", "removed", "terminated",
                   "copyright", "does not exist", "no longer available", "account associated")


def is_permanent(msg):
    """Only errors saying the video itself is gone count as permanent; other errors (blocked, cookies unreadable, network) are retried next time."""
    m = msg.lower()
    if any(w in m for w in ("sign in", "bot", "cookie", "requested format", "format is not available",
                            "try again later", "rate-limited", "rate limit")):
        return False
    return any(w in m for w in PERMANENT_WORDS)


def load_failed():
    """Read the failure log; drop entries that are not permanent so they are retried."""
    if not FAILED.exists():
        return {}
    with open(FAILED, encoding="utf-8") as f:
        rows = {r["ytid"]: r["reason"] for r in csv.DictReader(f)}
    real_fail = {k: v for k, v in rows.items() if is_permanent(v)}
    if len(real_fail) != len(rows):
        with open(FAILED, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ytid", "reason"])
            w.writerows(real_fail.items())
        print(f"Cleared {len(rows) - len(real_fail)} failure entries that were not missing videos; they will be retried")
    return real_fail


EXTRA_OPTS = {}


def download_one(row):
    ytid, start, end = row["ytid"], float(row["start_s"]), float(row["end_s"])
    out = REAL / f"{ytid}.wav"
    if out.exists():
        return ytid, "exists"
    # First try downloading only the 10 s excerpt (fast); if that fails, download the whole audio and cut 10 s locally (slower but reliable)
    first_err = None
    for whole in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            opts = {
                # Prefer plain AAC (m4a) / Opus audio; avoid YouTube's newer IAMF surround format (ffmpeg cannot read it)
                "format": "ba[ext=m4a]/ba[acodec^=opus]/ba[acodec!*=iamf]/b[acodec!*=iamf]",
                "outtmpl": str(Path(tmp) / "clip.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "noplaylist": True,
                **EXTRA_OPTS,
            }
            if not whole:
                opts["download_ranges"] = download_range_func(None, [(start, end)])
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
            except Exception as e:
                msg = "FAIL: " + str(e).splitlines()[0][:150]
                if is_permanent(msg) or not any(w in msg.lower() for w in ("ffmpeg", "stream", "codec")):
                    return ytid, msg            # video gone / blocked: another download method will not help, return now
                first_err = first_err or msg
                continue
            files = [f for f in Path(tmp).glob("clip.*") if f.suffix != ".part"]
            if not files:
                first_err = first_err or "FAIL: no file"
                continue
            # Convert to 16 kHz mono WAV, 10 s (same format as the AI clips); for full downloads, cut from start seconds
            seek = ["-ss", str(start)] if whole else []
            cmd = ["ffmpeg", "-y", "-loglevel", "error", *seek, "-i", str(files[0]),
                   "-vn", "-ac", "1", "-ar", "16000", "-t", str(end - start), str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0 and out.exists() and out.stat().st_size > 100_000:
                return ytid, "ok"
            out.unlink(missing_ok=True)
            first_err = first_err or "FAIL: ffmpeg " + r.stderr[:150]
    return ytid, first_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="download only the first N clips (for testing)")
    ap.add_argument("--shard", default="0/1", help="split the work: run 0/3, 1/3, 2/3 in three windows without overlap")
    ap.add_argument("--sleep", type=float, default=3, help="minimum seconds to wait between clips")
    ap.add_argument("--browser", default=None, help="use the browser's YouTube session: firefox / edge / chrome")
    ap.add_argument("--cookies", default=None, help="or the path to an exported cookies.txt file")
    ap.add_argument("--pause", type=float, default=60, help="minutes to pause automatically when blocked")
    ap.add_argument("--max-pauses", type=int, default=4, help="stop completely after this many automatic pauses")
    args = ap.parse_args()

    EXTRA_OPTS.update({"sleep_interval": args.sleep, "max_sleep_interval": args.sleep * 2})
    if args.browser:
        EXTRA_OPTS["cookiesfrombrowser"] = (args.browser,)
        print(f"Using the YouTube session from {args.browser} (close that browser completely first)")
    if args.cookies:
        EXTRA_OPTS["cookiefile"] = args.cookies
        print(f"Using cookies file: {args.cookies}")

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Run: winget install Gyan.FFmpeg , then close and reopen the window.")

    REAL.mkdir(parents=True, exist_ok=True)
    rows = get_list()
    failed = load_failed()
    k, n = (int(v) for v in args.shard.split("/"))
    rows = [r for i, r in enumerate(rows) if i % n == k]
    if n > 1:
        print(f"Shard {k}/{n}: handling {len(rows)} clips")
    todo = [r for r in rows if r["ytid"] not in failed and not (REAL / f"{r['ytid']}.wav").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"List: {len(rows)} clips; downloaded {len(list(REAL.glob('*.wav')))}; known missing {len(failed)}; to download now {len(todo)}\n")

    new_fail = {}
    ok = blocked = streak = pauses = 0
    recent = []
    # Download one at a time (no threads) so Ctrl+C stops at once; each failure is written to the log immediately, so nothing is lost on close
    try:
        for i, row in enumerate(todo, 1):
            ytid, status = download_one(row)
            if status == "ok":
                ok += 1; streak = 0
            elif status.startswith("FAIL") and "cookie" in status.lower():
                print("\nCannot read the browser's sign-in cookies; stopping now. Reason:", status[:120])
                print("Chrome / Edge often cannot be read on Windows. Use Firefox instead (close Firefox completely first),")
                print("or export cookies.txt with a browser extension and add --cookies cookies.txt")
                break
            elif status.startswith("FAIL") and not is_permanent(status):
                blocked += 1; streak += 1          # blocked / other temporary problem: not marked as missing, retried next time
                if blocked <= 3:
                    print("  temporary error:", status[:150])
            elif status.startswith("FAIL"):
                new_fail[ytid] = status; streak = 0
                write_header = not FAILED.exists()
                with open(FAILED, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["ytid", "reason"])
                    w.writerow([ytid, status])
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}   ok {ok}   missing videos {len(new_fail)}   blocked/temporary {blocked}")
            recent.append(status == "ok")
            stuck = streak >= 10 or (len(recent) >= 40 and not any(recent[-40:]))
            if stuck:
                why = "10 temporary errors in a row" if streak >= 10 else "none of the last 40 succeeded"
                if pauses < args.max_pauses:
                    pauses += 1
                    t = time.strftime("%H:%M")
                    print(f"\n[{t}] {why} (most likely rate-limited by YouTube). Pausing {args.pause} minutes, then continuing"
                          f" (pause {pauses}/{args.max_pauses}; Ctrl+C stops at any time)")
                    time.sleep(args.pause * 60)
                    streak = 0; recent.clear()
                    print(f"[{time.strftime('%H:%M')}] resuming downloads")
                else:
                    print(f"\n{why}; still failing after {pauses} automatic pauses, stopping for good.")
                    print("Take a screenshot of these lines to diagnose the problem.")
                    break
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopped. The next run continues where this one left off.")

    if new_fail:
        print("\nExample failure reasons:")
        for k, v in list(new_fail.items())[:5]:
            print(f"  {k}: {v}")

    total = len(list(REAL.glob("*.wav")))
    print(f"\nThis run: {ok} succeeded, {len(new_fail)} failed. The real folder now holds {total} real clips.")


if __name__ == "__main__":
    main()
