"""
Downloads MediaPipe's Pose Landmarker (lite) model into models/.

  python scripts/download_models.py            # lite, ~5.5 MB (default, best for CPU laptops)
  python scripts/download_models.py --variant full

The model comes from Google's official MediaPipe model storage and is
licensed under Apache 2.0. The script only talks to that host over HTTPS,
caps the download size, checks the file looks like a .task bundle (a zip),
and writes it atomically. It prints the SHA-256 so you can pin it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

HOST = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
VARIANTS = {
    "lite": HOST + "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": HOST + "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}
MAX_BYTES = 40 * 1024 * 1024
MODELS = Path(__file__).resolve().parent.parent / "models"


def download(url: str, dest: Path) -> str:
    if not url.startswith(HOST):
        raise SystemExit("refusing to download from an unexpected host")
    req = urllib.request.Request(url, headers={"User-Agent": "posecue-model-downloader"})
    tmp = dest.with_suffix(".part")
    h = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            if not r.geturl().startswith("https://"):
                raise SystemExit("download was redirected away from HTTPS")
            while chunk := r.read(1 << 16):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise SystemExit("download is larger than expected; stopped")
                h.update(chunk)
                f.write(chunk)
        with open(tmp, "rb") as f:
            # .task bundles are zip archives; MediaPipe's start with a 2-byte prefix before the signature
            if b"PK\x03\x04" not in f.read(8):
                raise SystemExit("downloaded file is not a MediaPipe .task bundle")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="lite")
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    a = ap.parse_args()
    url = VARIANTS[a.variant]
    dest = MODELS / url.rsplit("/", 1)[1]
    MODELS.mkdir(exist_ok=True)
    if dest.exists() and not a.force:
        print(f"already there: {dest}")
        return 0
    print(f"downloading {url}")
    digest = download(url, dest)
    print(f"saved {dest} ({dest.stat().st_size / 2**20:.1f} MB)\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
