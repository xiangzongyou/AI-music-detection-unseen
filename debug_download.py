"""
Diagnoses why a downloaded file has no audio stream. Checks a single video, about half a minute.
Usage:
    python debug_download.py --browser firefox
Send a screenshot of the full output.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from paths import ROOT

import yt_dlp

ap = argparse.ArgumentParser()
ap.add_argument("--browser", default=None)
ap.add_argument("--id", default=None, help="defaults to the video that kept failing (ID starting 338iZ76huS)")
args = ap.parse_args()
if not args.id:
    import csv
    with open(ROOT / "data" / "musiccaps-public.csv", encoding="utf-8") as fh:
        args.id = next(r["ytid"] for r in csv.DictReader(fh) if r["ytid"].startswith("338iZ76huS"))

print("Python", sys.version.split()[0], "| yt-dlp", yt_dlp.version.__version__)
r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout.splitlines()[0] if r.stdout else "ffmpeg not found")
r = subprocess.run(["deno", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", shell=True)
print("deno:", (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "deno not found")

FMT = "ba[ext=m4a]/ba[acodec^=opus]/ba[acodec!*=iamf]/b[acodec!*=iamf]"
opts = {"quiet": True, "no_warnings": True, "format": FMT}
if args.browser:
    opts["cookiesfrombrowser"] = (args.browser,)
url = f"https://www.youtube.com/watch?v={args.id}"

print(f"\n== Audio formats for video {args.id}:")
with yt_dlp.YoutubeDL(opts) as ydl:
    info = ydl.extract_info(url, download=False)
for f in info.get("formats", []):
    if f.get("acodec") not in (None, "none"):
        print(f"  id={f.get('format_id'):<8} ext={f.get('ext'):<5} acodec={str(f.get('acodec')):<22} "
              f"vcodec={str(f.get('vcodec')):<10} abr={f.get('abr')} proto={f.get('protocol')}")
chosen = info.get("requested_formats") or [info]
print("\n== Selected format:", [(c.get("format_id"), c.get("ext"), c.get("acodec")) for c in chosen])

with tempfile.TemporaryDirectory() as tmp:
    opts["outtmpl"] = str(Path(tmp) / "clip.%(ext)s")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    for f in Path(tmp).iterdir():
        print(f"\n== Downloaded file: {f.name}  {f.stat().st_size / 1e6:.2f} MB")
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,sample_rate",
                            "-of", "json", str(f)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("Streams seen by ffprobe:", r.stdout.strip() or r.stderr.strip()[:300])
        print("File header:", f.read_bytes()[:16])
