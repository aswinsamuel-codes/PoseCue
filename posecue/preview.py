"""
Preview video: the dancer (with the tracked skeleton) side by side with a
simulated LED costume lit by the generated cues, plus the song.

The LED figure uses posecue.render, a port of Show Studio's digital twin, so
it shows what the costumes will do. Frames are streamed straight to disk.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from posecue.audio import find_ffmpeg
from posecue.landmarks import SKELETON, PoseSequence
from posecue.moves import Move
from posecue.render import Layout, render_dancer
from posecue.zones import Costume

_BG = (12, 12, 16)
_SILHOUETTE = (29, 22, 22)   # BGR of LumaCue's #16161d
_LIMBS = [([(35, 46), (22, 78), (13, 110)], 9), ([(65, 46), (78, 78), (87, 110)], 9),
          ([(44, 108), (41, 152), (39, 192)], 11), ([(56, 108), (59, 152), (61, 192)], 11)]
_LABEL = {"arm_raise": "ARM UP", "jump": "JUMP", "spin": "SPIN", "hit": "HIT", "still": "FREEZE"}


def _draw_figure(cv2, panel: np.ndarray, leds: np.ndarray, layout: Layout, scale: float, ox: float, oy: float) -> None:
    def P(x, y):
        return int(round(ox + x * scale)), int(round(oy + y * scale))

    cv2.circle(panel, P(50, 21), int(13 * scale), _SILHOUETTE, -1, cv2.LINE_AA)
    torso = np.array([P(33, 41), P(67, 41), P(63, 110), P(37, 110)], np.int32)
    cv2.fillConvexPoly(panel, torso, _SILHOUETTE, cv2.LINE_AA)
    for pts, w in _LIMBS:
        cv2.polylines(panel, [np.array([P(*q) for q in pts], np.int32)], False, _SILHOUETTE,
                      max(1, int(w * scale)), cv2.LINE_AA)
    glow = np.zeros_like(panel)
    r = max(2, int(round(1.6 * scale)))
    for (x, y), (red, green, blue) in zip(layout.points, leds):
        if red + green + blue < 6:
            cv2.circle(panel, P(x, y), r, (40, 40, 48), -1, cv2.LINE_AA)
            continue
        col = (int(blue), int(green), int(red))
        cv2.circle(glow, P(x, y), r * 3, col, -1, cv2.LINE_AA)
        cv2.circle(panel, P(x, y), r, col, -1, cv2.LINE_AA)
    k = (r * 6) | 1
    panel[:] = cv2.add(panel, (cv2.GaussianBlur(glow, (k, k), 0) * 0.6).astype(np.uint8))


def _draw_skeleton(cv2, img: np.ndarray, xy: np.ndarray, vis: np.ndarray) -> None:
    h, w = img.shape[:2]
    for a, b in SKELETON:
        if vis[a] < 0.3 or vis[b] < 0.3 or not np.isfinite(xy[[a, b]]).all():
            continue
        cv2.line(img, (int(xy[a, 0] * w), int(xy[a, 1] * h)), (int(xy[b, 0] * w), int(xy[b, 1] * h)),
                 (0, 229, 255), 2, cv2.LINE_AA)
    for k in range(len(xy)):
        if vis[k] >= 0.3 and np.isfinite(xy[k]).all():
            cv2.circle(img, (int(xy[k, 0] * w), int(xy[k, 1] * h)), 3, (255, 255, 255), -1, cv2.LINE_AA)


def render_preview(out: Path, show: dict, costume: Costume, seq: PoseSequence, moves: list[Move], *,
                   video: Path | None = None, audio: Path | None = None, offset_ms: float = 0.0,
                   dancer: int = 0, height: int = 480, progress=None) -> Path:
    import cv2

    layout = Layout(costume)
    fps = seq.fps
    cap = None
    step = 1
    if video is not None:
        from posecue.pose import open_video

        cap, vfps, _, vw, vh = open_video(video)
        step = max(1, int(round(vfps / fps)))
        fps = vfps / step
        left_w = int(round(height * vw / vh))
    else:
        left_w = int(round(height * seq.aspect))
    left_w += left_w % 2
    right_w = int(height * 0.6) // 2 * 2
    size = (left_w + right_w, height)
    duration_ms = min(seq.duration_ms, show["lengthMs"] - offset_ms) if show["lengthMs"] > offset_ms else seq.duration_ms

    tmpdir = tempfile.TemporaryDirectory(prefix="posecue-")
    silent = Path(tmpdir.name) / "preview.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        tmpdir.cleanup()
        raise RuntimeError("OpenCV could not create the preview video")
    scale = height * 0.92 / 200
    ox, oy = (right_w - 100 * scale) / 2, height * 0.04
    n_out = int(duration_ms / 1000 * fps)
    i_frame = 0
    try:
        for k in range(n_out):
            t = k * 1000 / fps
            if cap is not None:
                frame = None
                while i_frame <= k * step:
                    ok, frame = cap.read()
                    i_frame += 1
                    if not ok:
                        frame = None
                        break
                if frame is None:
                    break
                left = cv2.resize(frame, (left_w, height), interpolation=cv2.INTER_AREA)
            else:
                left = np.full((height, left_w, 3), _BG, np.uint8)
            j = int(np.clip(np.searchsorted(seq.times_ms, t), 0, len(seq) - 1))
            if abs(seq.times_ms[j] - t) <= 1000 / seq.fps:
                _draw_skeleton(cv2, left, seq.xy[j], seq.visibility[j])
            labels = sorted({_LABEL[m.kind] + ("" if m.part == "body" else f" {m.part[-1].upper()}")
                             for m in moves if m.start_ms - 50 <= t <= max(m.end_ms, m.start_ms + 400)})
            for n, text in enumerate(labels[:4]):
                cv2.putText(left, text, (12, 32 + 28 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 255), 2, cv2.LINE_AA)
            right = np.full((height, right_w, 3), _BG, np.uint8)
            show_t = t + offset_ms
            _draw_figure(cv2, right, render_dancer(show, layout, dancer, show_t), layout, scale, ox, oy)
            sec = next((s["name"] for s in show.get("sections", []) if s["startMs"] <= show_t < s["endMs"]), "")
            cv2.putText(right, f"{show_t / 1000:5.1f}s {sec}", (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 210), 1, cv2.LINE_AA)
            writer.write(np.hstack([left, right]))
            if progress and k % 60 == 0:
                progress(k, n_out)
    finally:
        writer.release()
        if cap is not None:
            cap.release()
    try:
        _finish(silent, out, audio or video, offset_ms)
    finally:
        tmpdir.cleanup()
    return out


def _finish(silent: Path, out: Path, sound: Path | None, offset_ms: float) -> None:
    """Re-encodes to H.264 (plays everywhere) with the song; falls back to the silent mp4v file."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        out.write_bytes(silent.read_bytes())
        return
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-protocol_whitelist", "file",
           "-i", f"file:{silent}"]
    if sound is not None:
        if offset_ms >= 0:
            cmd += ["-ss", f"{offset_ms / 1000:.3f}"]
        else:
            cmd += ["-itsoffset", f"{-offset_ms / 1000:.3f}"]
        cmd += ["-protocol_whitelist", "file", "-i", f"file:{sound}"]
        cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-threads", "2",
            f"file:{out}"]
    proc = subprocess.run(cmd, capture_output=True, timeout=1800, check=False)
    if proc.returncode != 0:
        out.write_bytes(silent.read_bytes())
