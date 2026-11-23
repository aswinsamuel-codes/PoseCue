"""
Key-move detection: pure functions on a PoseSequence.

Everything is measured in "torsos" (the median shoulder-to-hip distance in the
video), so the thresholds work whether the dancer fills the frame or is a
small figure on a wide stage shot. Times are in ms.

Detected moves
  arm_raise  a wrist goes clearly above its shoulder (per side)
  jump       hips and both ankles leave their usual height together
  spin       the shoulders swap sides twice within a couple of seconds
  hit        a limb moves fast then stops sharply ("pop", "lock", a punch)
  still      the whole body barely moves for over a second
and a 0–1 dance energy curve from overall body speed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from posecue.landmarks import (
    L_ANKLE, L_HIP, L_SHOULDER, L_WRIST, L_ELBOW, L_KNEE,
    R_ANKLE, R_HIP, R_SHOULDER, R_WRIST, R_ELBOW, R_KNEE,
    PoseSequence, fill_gaps, smooth,
)

# Body parts a move can belong to. cues.py maps them to costume zones.
PARTS = ("arm_l", "arm_r", "leg_l", "leg_r", "body")

_LIMB_TIP = {"arm_l": L_WRIST, "arm_r": R_WRIST, "leg_l": L_ANKLE, "leg_r": R_ANKLE}
_MOTION_POINTS = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
                  L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]


@dataclass
class Move:
    kind: str          # arm_raise | jump | spin | hit | still
    part: str          # one of PARTS
    start_ms: float
    end_ms: float
    strength: float    # 0–1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Tuning:
    """Thresholds, in torsos and seconds. Defaults were set on synthetic and test clips."""
    raise_up: float = 0.15          # wrist this far above the shoulder → raised
    raise_down: float = 0.0         # back below this → lowered (hysteresis)
    raise_min_s: float = 0.25
    jump_height: float = 0.22       # hips rise this far above their running baseline
    jump_ankle: float = 0.15        # both ankles must rise at least this much too
    jump_max_s: float = 1.2
    spin_width: float = 0.15        # |shoulder width| needed to count a side as "facing"
    spin_max_s: float = 2.5
    hit_speed: float = 5.0          # limb tip speed (torsos/s) a hit must reach
    hit_stop: float = 0.35          # … then fall to this fraction of the peak
    hit_stop_s: float = 0.2         # … within this time
    hit_gap_s: float = 0.25
    still_speed: float = 0.35       # mean body speed below this is "still"
    still_min_s: float = 1.2
    energy_ref: float = 2.5         # body speed that counts as full energy


@dataclass
class Analysis:
    moves: list[Move]
    energy: np.ndarray      # (T,) 0–1
    times_ms: np.ndarray    # (T,)
    torso: float            # torso length in image-height units

    def energy_between(self, start_ms: float, end_ms: float) -> float:
        m = (self.times_ms >= start_ms) & (self.times_ms < end_ms)
        v = self.energy[m]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")


# ---------------------------------------------------------------- geometry

def _prepare(seq: PoseSequence) -> tuple[np.ndarray, float]:
    """Image coords → (T, 33, 2) in image-height units with y pointing UP, gaps filled, lightly smoothed."""
    xy = seq.xy.copy()
    low_vis = seq.visibility < 0.3
    xy[low_vis] = np.nan
    xy = fill_gaps(xy, max_gap=max(1, int(round(seq.fps * 0.5))))
    xy[..., 0] *= seq.aspect
    xy[..., 1] = -xy[..., 1]
    xy = fix_label_swaps(xy, _torso(xy))
    xy = smooth(xy, max(1, int(round(seq.fps * 0.1)) | 1))
    return xy, _torso(xy)


def _torso(xy: np.ndarray) -> float:
    """Median shoulder-centre to hip-centre distance."""
    mid_sh = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    mid_hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    t = np.linalg.norm(mid_sh - mid_hip, axis=1)
    t = t[np.isfinite(t)]
    return float(np.median(t)) if t.size else float("nan")


_LR_PAIRS = [(L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
             (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE)]


def fix_label_swaps(xy: np.ndarray, torso: float, ratio: float = 0.5, jump: float = 0.3) -> np.ndarray:
    """Undo frames where the tracker swapped the dancer's left and right.

    Pose models can't always tell front from back, and then flip every left/right
    label for a few frames. Real bodies move continuously, so if keeping the labels
    means the joints teleported (over `jump` torsos each on average in one frame)
    and swapping them fits the previous frame much better, it was a swap. A real
    turn crosses left and right gradually, so it never trips this.
    """
    if not np.isfinite(torso) or torso <= 0:
        return xy.copy()
    out = xy.copy()
    a = [i for i, _ in _LR_PAIRS]
    b = [j for _, j in _LR_PAIRS]
    all_pairs = [(i, i + 1) for i in range(11, 33, 2)] + [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)]
    for t in range(1, len(out)):
        prev, cur = out[t - 1], out[t]
        if not (np.isfinite(prev[a + b]).all() and np.isfinite(cur[a + b]).all()):
            continue
        keep = np.linalg.norm(cur[a] - prev[a], axis=1).sum() + np.linalg.norm(cur[b] - prev[b], axis=1).sum()
        swap = np.linalg.norm(cur[b] - prev[a], axis=1).sum() + np.linalg.norm(cur[a] - prev[b], axis=1).sum()
        if keep > jump * torso * 2 * len(a) and swap < ratio * keep:
            for i, j in all_pairs:
                out[t, [i, j]] = out[t, [j, i]]
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _hysteresis(signal: np.ndarray, up: float, down: float) -> np.ndarray:
    on = False
    out = np.zeros(len(signal), dtype=bool)
    for i, v in enumerate(signal):
        if not np.isfinite(v):
            on = False
        elif on and v < down:
            on = False
        elif not on and v > up:
            on = True
        out[i] = on
    return out


def _speed(p: np.ndarray, fps: float, torso: float) -> np.ndarray:
    """Speed (torsos/s) of a (T, 2) track; NaN where unknown."""
    v = np.full(len(p), np.nan)
    if len(p) > 1:
        d = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps / torso
        v[1:] = d
        v[0] = d[0]
    return v


# ---------------------------------------------------------------- detectors

def detect_arm_raises(xy: np.ndarray, t_ms: np.ndarray, fps: float, torso: float, tu: Tuning) -> list[Move]:
    out = []
    for part, wrist, shoulder in (("arm_l", L_WRIST, L_SHOULDER), ("arm_r", R_WRIST, R_SHOULDER)):
        height = (xy[:, wrist, 1] - xy[:, shoulder, 1]) / torso
        raised = _hysteresis(height, tu.raise_up, tu.raise_down)
        for a, b in _runs(raised):
            if (b - a) / fps < tu.raise_min_s:
                continue
            peak = float(np.nanmax(height[a:b]))
            out.append(Move("arm_raise", part, float(t_ms[a]), float(t_ms[b - 1] + 1000 / fps),
                            float(np.clip(peak / 1.2, 0.2, 1.0))))
    return out


def _running_baseline(y: np.ndarray, window: int) -> np.ndarray:
    """Running low percentile of y: the usual standing height, robust to short jumps."""
    out = np.full(len(y), np.nan)
    half = window // 2
    for i in range(len(y)):
        seg = y[max(0, i - half): i + half + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.percentile(seg, 30)
    return out


def detect_jumps(xy: np.ndarray, t_ms: np.ndarray, fps: float, torso: float, tu: Tuning) -> list[Move]:
    hip = (xy[:, L_HIP, 1] + xy[:, R_HIP, 1]) / 2
    win = max(3, int(fps * 3))
    lift = (hip - _running_baseline(hip, win)) / torso
    lift_l = (xy[:, L_ANKLE, 1] - _running_baseline(xy[:, L_ANKLE, 1], win)) / torso
    lift_r = (xy[:, R_ANKLE, 1] - _running_baseline(xy[:, R_ANKLE, 1], win)) / torso
    airborne = (lift > tu.jump_height / 2) & (np.fmin(lift_l, lift_r) > tu.jump_ankle / 2)
    out = []
    for a, b in _runs(np.nan_to_num(airborne, nan=0).astype(bool)):
        peak = float(np.nanmax(lift[a:b]))
        ankles = float(np.nanmax(np.fmin(lift_l, lift_r)[a:b]))
        dur = (b - a) / fps
        if peak < tu.jump_height or ankles < tu.jump_ankle or dur > tu.jump_max_s:
            continue
        if ankles < 0.5 * peak:
            continue   # hips moved but the feet didn't follow: a tracking wobble, not a jump
        out.append(Move("jump", "body", float(t_ms[a]), float(t_ms[b - 1] + 1000 / fps),
                        float(np.clip(peak / 0.8, 0.3, 1.0))))
    return out


def detect_spins(xy: np.ndarray, t_ms: np.ndarray, fps: float, torso: float, tu: Tuning) -> list[Move]:
    # Dancer's left shoulder minus right shoulder, along x. Facing the camera it is
    # positive (their left is on the image's right); facing away it is negative.
    width = (xy[:, L_SHOULDER, 0] - xy[:, R_SHOULDER, 0]) / torso
    side = np.where(width > tu.spin_width, 1, np.where(width < -tu.spin_width, -1, 0))
    # times at which the facing side changed (ignoring the "edge-on" frames in between)
    typical = np.nanmedian(np.abs(width)) if np.isfinite(width).any() else 0.0
    edge_on = np.abs(width) < max(tu.spin_width * 1.5, 0.45 * typical)
    near = max(1, int(round(0.4 * fps)))
    flips: list[int] = []
    last = 0
    for i, s in enumerate(side):
        if s == 0:
            continue
        # a real turn shows the shoulders edge-on around the flip; an instant flip is a tracking glitch
        if last != 0 and s != last and edge_on[max(0, i - near): i + 1].any():
            flips.append(i)
        last = s
    out = []
    i = 0
    while i + 1 < len(flips):
        a, b = flips[i], flips[i + 1]
        if (b - a) / fps <= tu.spin_max_s:
            # extend back over the edge-on frames to where the turn began
            start = a
            while start > 0 and side[start - 1] == 0 and (a - start) / fps < 0.6:
                start -= 1
            n_turns = 1
            j = i + 2
            while j + 1 < len(flips) and (flips[j + 1] - flips[j]) / fps <= tu.spin_max_s and (flips[j] - flips[j - 1]) / fps <= tu.spin_max_s:
                n_turns += 1
                j += 2
            end = flips[j - 1]
            out.append(Move("spin", "body", float(t_ms[start]), float(t_ms[end] + 1000 / fps),
                            float(np.clip(0.6 + 0.2 * n_turns, 0.6, 1.0))))
            i = j
        else:
            i += 1
    return out


def detect_hits(xy: np.ndarray, t_ms: np.ndarray, fps: float, torso: float, tu: Tuning) -> list[Move]:
    out = []
    stop_n = max(1, int(round(tu.hit_stop_s * fps)))
    gap_n = max(1, int(round(tu.hit_gap_s * fps)))
    for part, tip in _LIMB_TIP.items():
        # speed relative to the hips, so walking across the stage isn't a hit
        hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
        v = _speed(xy[:, tip] - hip, fps, torso)
        if not np.isfinite(v).any():
            continue
        last = -10**9
        for i in range(1, len(v) - 1):
            if not np.isfinite(v[i]) or v[i] < tu.hit_speed:
                continue
            if v[i] < np.nanmax(v[max(0, i - 1): i + 2]):
                continue   # not the local peak
            after = v[i + 1: i + 1 + stop_n]
            if not after.size or np.nanmin(after) > tu.hit_stop * v[i]:
                continue   # didn't stop sharply
            if i - last < gap_n:
                continue
            last = i
            stop_i = i + 1 + int(np.nanargmin(after))
            out.append(Move("hit", part, float(t_ms[stop_i]), float(t_ms[stop_i] + 150),
                            float(np.clip(v[i] / (tu.hit_speed * 2.5), 0.3, 1.0))))
    return out


def body_speed(xy: np.ndarray, fps: float, torso: float) -> np.ndarray:
    speeds = np.stack([_speed(xy[:, k], fps, torso) for k in _MOTION_POINTS], axis=1)
    with np.errstate(all="ignore"):
        return np.nanmean(speeds, axis=1)


def detect_stillness(speed: np.ndarray, t_ms: np.ndarray, fps: float, tu: Tuning) -> list[Move]:
    out = []
    calm = smooth(speed, max(1, int(fps * 0.3)) | 1)
    for a, b in _runs(np.nan_to_num(calm, nan=np.inf) < tu.still_speed):
        if (b - a) / fps >= tu.still_min_s:
            out.append(Move("still", "body", float(t_ms[a]), float(t_ms[b - 1] + 1000 / fps), 1.0))
    return out


def energy_curve(speed: np.ndarray, fps: float, tu: Tuning) -> np.ndarray:
    return np.clip(smooth(speed, max(1, int(fps)) | 1) / tu.energy_ref, 0, 1)


def _part_of_bigger_move(hit: Move, moves: list[Move]) -> bool:
    """A jump or spin moves every limb fast, and an arm snapping up or down is the raise itself:
    those limb stops are not separate hits."""
    for m in moves:
        if m.kind in ("jump", "spin") and m.start_ms - 200 <= hit.start_ms <= m.end_ms + 300:
            return True
        if m.kind == "arm_raise" and m.part == hit.part and (
                abs(hit.start_ms - m.start_ms) <= 500 or abs(hit.start_ms - m.end_ms) <= 500):
            return True
    return False


def analyze_moves(seq: PoseSequence, tuning: Tuning | None = None) -> Analysis:
    tu = tuning or Tuning()
    if len(seq) < 3:
        return Analysis([], np.zeros(len(seq)), seq.times_ms, float("nan"))
    xy, torso = _prepare(seq)
    if not np.isfinite(torso) or torso <= 1e-4:
        return Analysis([], np.full(len(seq), np.nan), seq.times_ms, float("nan"))
    t, fps = seq.times_ms, seq.fps
    speed = body_speed(xy, fps, torso)
    moves = (detect_arm_raises(xy, t, fps, torso, tu) + detect_jumps(xy, t, fps, torso, tu)
             + detect_spins(xy, t, fps, torso, tu) + detect_hits(xy, t, fps, torso, tu)
             + detect_stillness(speed, t, fps, tu))
    spins = [m for m in moves if m.kind == "spin"]
    moves = [m for m in moves if not (m.kind == "jump" and any(sp.start_ms <= m.start_ms <= sp.end_ms for sp in spins))]
    moves = [m for m in moves if not (m.kind == "hit" and _part_of_bigger_move(m, moves))]
    moves.sort(key=lambda m: (m.start_ms, m.kind, m.part))
    return Analysis(moves, energy_curve(speed, fps, tu), t, torso)

