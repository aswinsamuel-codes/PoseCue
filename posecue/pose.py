"""
Pose tracking: video → PoseSequence, with MediaPipe's Pose Landmarker (Tasks API).

Kept light for a CPU-only laptop: frames are shrunk to `max_width` before
tracking, only every `frame_step`-th frame is analysed, OpenCV is held to two
threads, and frames are streamed (never all held in memory). MediaPipe is
imported only here, so the rest of PoseCue (and its tests) run without it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from posecue.landmarks import N_LANDMARKS, PoseSequence
from posecue.safety import MAX_DURATION_S

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
MAX_MODEL_BYTES = 64 * 1024 * 1024


def _limit_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTEROP_THREADS",
                "TF_NUM_INTRAOP_THREADS"):
        os.environ.setdefault(var, str(n))
    import cv2

    cv2.setNumThreads(n)


def open_video(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not (1 <= fps <= 240):
        fps = 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0 or w * h > 8192 * 8192:
        cap.release()
        raise RuntimeError("video frame size is missing or too large")
    if frames and frames / fps > MAX_DURATION_S:
        cap.release()
        raise RuntimeError(f"video is longer than {MAX_DURATION_S // 60} minutes; trim it first")
    return cap, fps, frames, w, h


def track_video(video: Path, model: Path = DEFAULT_MODEL, *, max_width: int = 480, frame_step: int = 2,
                threads: int = 2, progress: Callable[[int, int], None] | None = None) -> PoseSequence:
    """Runs the pose model over the video. Raises FileNotFoundError if the model isn't downloaded."""
    model = Path(model)
    if not model.is_file():
        raise FileNotFoundError(
            f"pose model not found at {model}. Run: python scripts/download_models.py "
            "(or pass --pose with a saved .npz from an earlier run)")
    if model.stat().st_size > MAX_MODEL_BYTES or model.suffix != ".task":
        raise ValueError("pose model must be a MediaPipe .task file under 64 MB")
    _limit_threads(threads)
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    cap, fps, n_frames, w, h = open_video(video)
    frame_step = max(1, int(frame_step))
    scale = min(1.0, max_width / w)
    size = (max(2, int(round(w * scale))), max(2, int(round(h * scale))))
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    times, xys, vis = [], [], []
    max_frames = int(MAX_DURATION_S * fps)
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while i < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if i % frame_step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                t_ms = int(round(i * 1000 / fps))
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
                xy = np.full((N_LANDMARKS, 2), np.nan)
                v = np.zeros(N_LANDMARKS)
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    xy[:] = [(lm.x, lm.y) for lm in lms[:N_LANDMARKS]]
                    v[:] = [lm.visibility if lm.visibility is not None else 1.0 for lm in lms[:N_LANDMARKS]]
                times.append(t_ms)
                xys.append(xy)
                vis.append(v)
                if progress and len(times) % 30 == 0:
                    progress(i, n_frames)
            i += 1
    cap.release()
    if not times:
        raise RuntimeError("no frames could be read from the video")
    return PoseSequence(np.array(times, float), np.array(xys), np.array(vis), aspect=w / h, fps=fps / frame_step)
