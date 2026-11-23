"""
Input and output file checks.

PoseCue only reads files the user names on the command line and only writes
the outputs they ask for. These checks stop the common mistakes and abuses:
URLs or device names passed as "files" (ffmpeg and OpenCV would happily open
http://, pipe: or concat: inputs), huge files that would exhaust an 8 GB
laptop, outputs that would overwrite an input, and output names with an
unexpected extension.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

MAX_VIDEO_BYTES = 4 * 1024**3       # 4 GB
MAX_AUDIO_BYTES = 1024**3           # 1 GB
MAX_SHOW_JSON_BYTES = 16 * 1024**2  # LumaCue's own artifact/show limit is far below this
MAX_DURATION_S = 30 * 60            # LumaCue shows are at most 60 min; 30 min of video is plenty

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"} | VIDEO_EXT
POSE_EXT = {".npz"}

# Windows reserved device names (CON, NUL, COM1 …) are not files.
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$", re.IGNORECASE)
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]+:")   # http:, pipe:, concat: … (not C:)


class FileCheckError(ValueError):
    pass


def _reject_non_path(raw: str) -> None:
    if not raw or "\x00" in raw:
        raise FileCheckError("empty or invalid file name")
    if _SCHEME.match(raw):
        raise FileCheckError(f"{raw!r} looks like a URL or protocol, not a local file")
    if _RESERVED.match(Path(raw).name):
        raise FileCheckError(f"{Path(raw).name!r} is a reserved device name")


def check_input(raw: str, kinds: set[str], max_bytes: int, what: str) -> Path:
    """An existing regular file with an allowed extension and a sane size."""
    _reject_non_path(raw)
    p = Path(raw).expanduser()
    try:
        p = p.resolve(strict=True)
    except (OSError, RuntimeError):
        raise FileCheckError(f"{what} not found: {raw}") from None
    if not p.is_file():
        raise FileCheckError(f"{what} is not a regular file: {raw}")
    if p.suffix.lower() not in kinds:
        raise FileCheckError(f"{what} must be one of {', '.join(sorted(kinds))} (got {p.suffix or 'no extension'})")
    size = p.stat().st_size
    if size == 0:
        raise FileCheckError(f"{what} is empty: {raw}")
    if size > max_bytes:
        raise FileCheckError(f"{what} is {size / 2**20:.0f} MB, over the {max_bytes / 2**20:.0f} MB limit")
    return p


def check_output(raw: str, kinds: set[str], inputs: list[Path], force: bool, what: str) -> Path:
    """A writable output path with the right extension that doesn't clobber an input."""
    _reject_non_path(raw)
    p = Path(raw).expanduser().resolve()
    if p.suffix.lower() not in kinds:
        raise FileCheckError(f"{what} must end in {', '.join(sorted(kinds))}")
    if p.is_dir():
        raise FileCheckError(f"{what} is a folder: {raw}")
    if not p.parent.is_dir():
        raise FileCheckError(f"folder for {what} does not exist: {p.parent}")
    for i in inputs:
        if i is not None and p == i:
            raise FileCheckError(f"{what} would overwrite an input file: {raw}")
    if p.exists() and not force:
        raise FileCheckError(f"{what} already exists (use --force to overwrite): {raw}")
    return p


def write_text_atomic(path: Path, text: str) -> None:
    """Writes next to the target, then renames, so a crash never leaves half a show file."""
    data = text.encode("utf-8")
    if len(data) > MAX_SHOW_JSON_BYTES:
        raise FileCheckError("show file would be larger than 16 MB; shorten the video or lower the cue density")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def safe_display_name(name: str, limit: int = 80) -> str:
    """A show / audio name for the JSON: no control characters, no folders, bounded length."""
    name = Path(name).name if ("/" in name or "\\" in name) else name
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:limit] or "PoseCue show"
