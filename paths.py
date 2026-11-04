"""Where data, features and results live.

Set the environment variable FMC_ROOT to your working folder, for example
    Windows:  set FMC_ROOT=D:\\data\\FakeMusicCaps
    macOS/Linux:  export FMC_ROOT=~/data/FakeMusicCaps
If it is not set, a folder called `workspace` next to these scripts is used.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FMC_ROOT", Path(__file__).resolve().parent / "workspace"))
