"""
Pose sequences: what the tracker produces and the move detector reads.

A sequence holds the 33 BlazePose landmarks (MediaPipe's pose model) for each
analysed frame, in image coordinates: x and y from 0 to 1 (y grows downwards),
plus a visibility score. Frames where no dancer was found are NaN.

"Left" and "right" are the dancer's own left and right, as MediaPipe reports
them for an unmirrored video.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# BlazePose landmark indices used by PoseCue.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
N_LANDMARKS = 33

# Bones drawn in the preview.
SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

# Swapping these pairs mirrors a pose (for selfie-mode videos).
_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
                 (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

MAX_POSE_FILE_BYTES = 64 * 1024 * 1024


@dataclass
class PoseSequence:
    times_ms: np.ndarray     # (T,) float, ms from the start of the video
    xy: np.ndarray           # (T, 33, 2) float, image coords 0..1, NaN where missing
    visibility: np.ndarray   # (T, 33) float 0..1
    aspect: float = 16 / 9   # frame width / height, so x and y distances compare fairly
    fps: float = 15.0        # analysis rate (video fps / frame step)

    def __post_init__(self) -> None:
        self.times_ms = np.asarray(self.times_ms, dtype=np.float64)
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.visibility = np.asarray(self.visibility, dtype=np.float64)
        t = len(self.times_ms)
        if self.xy.shape != (t, N_LANDMARKS, 2) or self.visibility.shape != (t, N_LANDMARKS):
            raise ValueError(f"pose arrays have mismatched shapes: {self.xy.shape}, {self.visibility.shape}, {t} times")
        if not (np.isfinite(self.aspect) and 0.1 < self.aspect < 10):
            raise ValueError("aspect ratio out of range")
        if not (np.isfinite(self.fps) and 0.5 <= self.fps <= 240):
            raise ValueError("fps out of range")

    def __len__(self) -> int:
        return len(self.times_ms)

    @property
    def duration_ms(self) -> float:
        return float(self.times_ms[-1] + 1000 / self.fps) if len(self) else 0.0

    def mirrored(self) -> "PoseSequence":
        xy = self.xy.copy()
        vis = self.visibility.copy()
        xy[..., 0] = 1 - xy[..., 0]
        for a, b in _MIRROR_PAIRS:
            xy[:, [a, b]] = xy[:, [b, a]]
            vis[:, [a, b]] = vis[:, [b, a]]
        return PoseSequence(self.times_ms.copy(), xy, vis, self.aspect, self.fps)

    # ---- cache files: plain arrays only, never pickles ----
    def save(self, path: Path) -> None:
        np.savez_compressed(path, times_ms=self.times_ms, xy=self.xy, visibility=self.visibility,
                            meta=np.array([self.aspect, self.fps]))

    @staticmethod
    def load(path: Path) -> "PoseSequence":
        if Path(path).stat().st_size > MAX_POSE_FILE_BYTES:
            raise ValueError(f"pose file is larger than {MAX_POSE_FILE_BYTES // 2**20} MB")
        with np.load(path, allow_pickle=False) as z:
            meta = z["meta"]
            return PoseSequence(z["times_ms"], z["xy"], z["visibility"], float(meta[0]), float(meta[1]))


def fill_gaps(xy: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fills runs of up to `max_gap` missing frames (per landmark and axis)."""
    out = xy.copy()
    t = np.arange(len(xy))
    flat = out.reshape(len(xy), -1)
    for k in range(flat.shape[1]):
        col = flat[:, k]
        bad = ~np.isfinite(col)
        if not bad.any() or bad.all():
            continue
        good = ~bad
        filled = np.interp(t, t[good], col[good])
        # only fill short interior gaps
        run_start = None
        for i in range(len(col) + 1):
            missing = i < len(col) and bad[i]
            if missing and run_start is None:
                run_start = i
            elif not missing and run_start is not None:
                if i - run_start <= max_gap and run_start > 0 and i < len(col):
                    col[run_start:i] = filled[run_start:i]
                run_start = None
    return out


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0 that ignores NaNs."""
    if window <= 1 or len(x) == 0:
        return x.copy()
    k = np.ones(window)
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    shape = x.shape
    a = filled.reshape(len(x), -1)
    v = valid.reshape(len(x), -1).astype(float)
    num = np.stack([np.convolve(a[:, i], k, mode="same") for i in range(a.shape[1])], axis=1)
    den = np.stack([np.convolve(v[:, i], k, mode="same") for i in range(v.shape[1])], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < window / 3] = np.nan
    out = out.reshape(shape)
    out[~valid] = np.nan
    return out
