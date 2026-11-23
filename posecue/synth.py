"""
Synthetic dancers and music, for tests and for trying PoseCue without a video
or the pose model (`python -m posecue --demo`).

A stick dancer faces the camera, grooves on the beat, and performs a script of
moves (arm raises, jumps, spins, punches, freezes). The music is a drum
pattern whose loudness follows a list of sections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posecue.landmarks import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, N_LANDMARKS, NOSE,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, PoseSequence,
)

# Standing pose in body units: origin between the hips, y up, torso = 1.
# The dancer faces the camera, so their LEFT side is on the image's right (+x).
_REST = {
    NOSE: (0, 1.45), L_SHOULDER: (0.4, 1.0), R_SHOULDER: (-0.4, 1.0), L_ELBOW: (0.5, 0.5), R_ELBOW: (-0.5, 0.5),
    L_WRIST: (0.55, 0.05), R_WRIST: (-0.55, 0.05), L_HIP: (0.2, 0), R_HIP: (-0.2, 0),
    L_KNEE: (0.22, -0.9), R_KNEE: (-0.22, -0.9), L_ANKLE: (0.22, -1.8), R_ANKLE: (-0.22, -1.8),
}
# The other BlazePose points follow a "parent" point (face → nose, hands → wrists, feet → ankles).
_PARENT = {1: NOSE, 2: NOSE, 3: NOSE, 4: NOSE, 5: NOSE, 6: NOSE, 7: NOSE, 8: NOSE, 9: NOSE, 10: NOSE,
           17: L_WRIST, 19: L_WRIST, 21: L_WRIST, 18: R_WRIST, 20: R_WRIST, 22: R_WRIST,
           29: L_ANKLE, 31: L_ANKLE, 30: R_ANKLE, 32: R_ANKLE}


@dataclass
class Step:
    kind: str        # arm_raise | jump | spin | punch | still
    start_s: float
    dur_s: float
    part: str = "body"   # arm_l / arm_r for arm_raise and punch


def _ease(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def synth_pose(duration_s: float, steps: list[Step], *, fps: float = 15.0, bpm: float = 120.0,
               groove: float = 0.15, aspect: float = 16 / 9, noise: float = 0.0, seed: int = 0) -> PoseSequence:
    t = np.arange(int(duration_s * fps)) / fps
    T = len(t)
    pts = np.zeros((T, N_LANDMARKS, 2))
    for k, (x, y) in _REST.items():
        pts[:, k] = (x, y)
    groove_amp = np.full(T, groove)

    for s in steps:
        u = (t - s.start_s) / s.dur_s
        inside = (u >= 0) & (u <= 1)
        if s.kind == "still":
            groove_amp[inside] = 0
        elif s.kind == "arm_raise":
            side = 1 if s.part == "arm_l" else -1
            elbow, wrist = (L_ELBOW, L_WRIST) if side > 0 else (R_ELBOW, R_WRIST)
            ramp = np.minimum(_ease(u / 0.2), _ease((1 - u) / 0.2))  # up over the first 20 %, down over the last 20 %
            ramp[~inside] = 0
            for k, target in ((elbow, (0.45 * side, 1.5)), (wrist, (0.5 * side, 2.0))):
                pts[:, k] += ramp[:, None] * (np.array(target) - pts[:, k])
        elif s.kind == "punch":
            side = 1 if s.part == "arm_l" else -1
            wrist, elbow = (L_WRIST, L_ELBOW) if side > 0 else (R_WRIST, R_ELBOW)
            out_t = 0.1 / s.dur_s  # snap out in 100 ms, hold, then drift back
            ramp = np.where(u < out_t, _ease(u / out_t), np.where(u < 0.6, 1.0, _ease((1 - u) / 0.4)))
            ramp[~inside] = 0
            for k, target in ((wrist, (1.3 * side, 1.0)), (elbow, (0.85 * side, 1.0))):
                pts[:, k] += ramp[:, None] * (np.array(target) - pts[:, k])
        elif s.kind == "jump":
            h = np.where(inside, 0.6 * np.sin(np.pi * np.clip(u, 0, 1)), 0)
            pts[:, :, 1] += h[:, None]
        elif s.kind == "spin":
            turns = max(1, round(s.dur_s / 1.0))
            ang = np.where(inside, 2 * np.pi * turns * _ease(u), 0)
            pts[:, :, 0] *= np.cos(ang)[:, None]

    # groove: a small bounce on the beat and a sway every two beats
    beat_hz = bpm / 60
    pts[:, :, 1] += (groove_amp * np.abs(np.sin(np.pi * beat_hz * t)))[:, None]
    pts[:, :, 0] += (groove_amp * 0.6 * np.sin(np.pi * beat_hz * t / 2))[:, None]
    for child, parent in _PARENT.items():
        pts[:, child] = pts[:, parent] + (0.03 * ((child % 3) - 1), 0.03 * ((child % 2) * 2 - 1))

    rng = np.random.default_rng(seed)
    if noise:
        pts += rng.normal(0, noise, pts.shape)
    s = 0.18
    xy = np.empty_like(pts)
    xy[..., 0] = (0.5 * aspect + pts[..., 0] * s) / aspect
    xy[..., 1] = 0.55 - pts[..., 1] * s
    vis = np.full((T, N_LANDMARKS), 0.95)
    return PoseSequence(t * 1000, xy, vis, aspect=aspect, fps=fps)


def synth_music(sections: list[tuple[float, float]], *, bpm: float = 120.0, sr: int = 22050, seed: int = 0) -> np.ndarray:
    """Drum loop: kick on every beat, snare on 2 and 4, hats when loud. sections = [(seconds, loudness 0–1)]."""
    rng = np.random.default_rng(seed)
    total = sum(d for d, _ in sections)
    n = int(total * sr)
    y = np.zeros(n, dtype=np.float64)
    amp = np.zeros(n)
    pos = 0
    for d, a in sections:
        amp[pos:pos + int(d * sr)] = a
        pos += int(d * sr)
    amp[pos:] = sections[-1][1]
    beat = 60 / bpm
    kick_t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-kick_t * 30)) * kick_t) * np.exp(-kick_t * 12)
    snare = rng.normal(0, 1, int(0.15 * sr)) * np.exp(-np.arange(int(0.15 * sr)) / sr * 25) * 0.5
    hat = rng.normal(0, 1, int(0.04 * sr)) * np.exp(-np.arange(int(0.04 * sr)) / sr * 90) * 0.25
    k = 0
    while k * beat < total:
        i = int(k * beat * sr)
        a = amp[min(i, n - 1)]

        def put(sound, at, gain):
            j = min(n, at + len(sound))
            if at < n:
                y[at:j] += sound[: j - at] * gain

        put(kick, i, a)
        if k % 2 == 1:
            put(snare, i, a)
        if a > 0.6:
            put(hat, i + int(beat / 2 * sr), a)
        k += 1
    y += rng.normal(0, 0.002, n)   # a little room noise so silence isn't digital zero
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32)


def demo_script() -> tuple[float, list[Step], list[tuple[float, float]]]:
    """A 32 s routine on a 120 BPM track: calm intro, build, drop, outro."""
    steps = [
        Step("still", 0.0, 2.0),
        Step("arm_raise", 4.0, 2.0, "arm_l"),
        Step("arm_raise", 6.0, 2.0, "arm_r"),
        Step("punch", 9.0, 1.0, "arm_r"),
        Step("punch", 10.5, 1.0, "arm_l"),
        Step("spin", 12.0, 1.5),
        Step("jump", 16.0, 0.5),
        Step("arm_raise", 18.0, 2.0, "arm_l"),
        Step("arm_raise", 18.0, 2.0, "arm_r"),
        Step("jump", 21.0, 0.5),
        Step("punch", 23.0, 1.0, "arm_l"),
        Step("spin", 25.0, 2.0),
        Step("still", 29.5, 2.5),
    ]
    music = [(8.0, 0.25), (8.0, 0.55), (12.0, 1.0), (4.0, 0.2)]
    return 32.0, steps, music
