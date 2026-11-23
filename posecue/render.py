"""
LED simulation for the preview: a Python port of LumaCue's digital twin
(lumacue/lib/show/render.ts, body.ts sampleZone, canvas.ts). Given a show and
a costume it returns the colour of every LED at a time t, following the same
rules as Show Studio, so the preview shows what the costume will do.
"""
from __future__ import annotations

import math

import numpy as np

from posecue.zones import Costume

ALL_DANCERS = -1
BAND_W = 48
_CENTRE_X, _CENTRE_Y = 128, 90
CANVAS_EFFECTS = ("rise", "sweep", "ripple")


def speed_hz(speed: float) -> float:
    return 0.25 * 2 ** ((speed / 100) * 5)


def lap_ms(speed: float) -> float:
    return 2000 / speed_hz(speed)


def speed_for(hz: float) -> float:
    """Inverse of speed_hz (autocue.ts speedFor): the 0–100 speed giving `hz`."""
    return max(0.0, min(100.0, (100 / 5) * math.log2(max(1e-9, hz) / 0.25)))


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    n = int(h.lstrip("#")[:6].ljust(6, "0"), 16)
    return (n >> 16) & 255, (n >> 8) & 255, n & 255


# ---------------------------------------------------------------- layout (body.ts)

def _length(path) -> float:
    return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))


def _point_at(path, d: float):
    for i in range(1, len(path)):
        seg = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if d <= seg or i == len(path) - 1:
            t = min(1.0, d / seg) if seg else 0.0
            return (path[i - 1][0] + (path[i][0] - path[i - 1][0]) * t, path[i - 1][1] + (path[i][1] - path[i - 1][1]) * t)
        d -= seg
    return path[-1]


def sample_zone(lines, count: int) -> list[tuple[float, float]]:
    lengths = [max(0.001, _length(p)) for p in lines]
    total = sum(lengths)
    counts = [int(math.floor(count * L / total)) for L in lengths]
    i = 0
    while sum(counts) < count:
        counts[i] += 1
        i = (i + 1) % len(counts)
    pts = []
    for path, c in zip(lines, counts):
        L = _length(path)
        pts += [_point_at(path, (j + 0.5) / c * L) for j in range(c)]
    return pts


class Layout:
    def __init__(self, costume: Costume):
        self.zones = []
        start = 0
        for z in costume.zones:
            pts = sample_zone(z.lines(), z.leds)
            self.zones.append((z.id, start, z.leds, pts))
            start += z.leds
        self.total = start
        self.points = np.array([p for _, _, _, pts in self.zones for p in pts], dtype=np.float64)


# ---------------------------------------------------------------- effects (render.ts)

def _hsv(h: float, s: float, v: float):
    i = math.floor(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return r * 255, g * 255, b * 255


def _rand(a: int, b: int) -> float:
    def imul(x, y):
        return ((x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)) & 0xFFFFFFFF

    def s32(x):
        x &= 0xFFFFFFFF
        return x - (1 << 32) if x & 0x80000000 else x

    h = s32(a * 374761393 + b * 668265263)
    h = s32(imul(h ^ ((h & 0xFFFFFFFF) >> 13), 1274126177))
    return ((h ^ ((h & 0xFFFFFFFF) >> 16)) & 0xFFFFFFFF) / 4294967296


def _band_level(elapsed: int, lap: int, v: int) -> int:
    if lap <= 0:
        return 0
    centre = -BAND_W + ((elapsed % lap) * (255 + 2 * BAND_W)) // lap
    d = abs(v - centre)
    return 255 - (d * 255) // BAND_W if d < BAND_W else 0


def _canvas_coord(effect: str, x: int, y: int) -> int:
    if effect == "rise":
        return 255 - y
    if effect == "sweep":
        return x
    return min(255, abs(x - _CENTRE_X) + abs(y - _CENTRE_Y))


def eval_effect(c: dict, tl: float, i: int, n: int, pos) -> tuple[float, float, float, float]:
    """Colour (r, g, b) and opacity of LED i of n for clip c, tl ms after it started."""
    r, g, b = hex_to_rgb(c["color"])
    eff = c["effect"]
    if eff in CANVAS_EFFECTS:
        x = max(0, min(255, round(pos[0] / 100 * 255)))
        y = max(0, min(255, round(pos[1] / 200 * 255)))
        k = _band_level(max(0, int(tl)), round(lap_ms(c["speed"])), _canvas_coord(eff, x, y))
        return r, g, b, k / 255
    hz = speed_hz(c["speed"])
    t = tl / 1000
    k, a = 1.0, 1.0
    col = (r, g, b)
    if eff == "fade":
        u = min(1.0, tl / max(1, c["durationMs"]))
        r2, g2, b2 = hex_to_rgb(c["color2"])
        col = (r + (r2 - r) * u, g + (g2 - g) * u, b + (b2 - b) * u)
    elif eff == "pulse":
        k = 0.5 - 0.5 * math.cos(2 * math.pi * (hz / 2) * t)
    elif eff == "strobe":
        a = 1.0 if (t * hz * 2) % 1 < 0.25 else 0.0
    elif eff == "chase":
        head = ((t * hz) / 2 % 1) * n
        tail = max(1, n / 6)
        d = (head - i + n) % n
        k = 1 - d / tail if d < tail else 0.0
        a = k
    elif eff == "wipe":
        u = min(1.0, tl / max(1, c["durationMs"]))
        a = 1.0 if (i + 0.5) / n <= u else 0.0
    elif eff == "rainbow":
        col = _hsv((i / n + t * hz / 4) % 1, 1, 1)
    elif eff == "twinkle":
        v = _rand(i, math.floor(t * hz * 4))
        k = (v - 0.7) / 0.3 if v > 0.7 else 0.0
        a = k
    elif eff == "blackout":
        col = (0, 0, 0)
    return col[0] * k, col[1] * k, col[2] * k, a


def active_clips(show: dict, dancer: int, t: float) -> list[dict]:
    hit = [c for c in show["clips"] if c["startMs"] <= t < c["startMs"] + c["durationMs"]]
    every = sorted((c for c in hit if c["track"] == ALL_DANCERS), key=lambda c: c["startMs"])
    own = sorted((c for c in hit if c["track"] == dancer), key=lambda c: c["startMs"])
    return every + own


def render_dancer(show: dict, layout: Layout, dancer: int, t: float) -> np.ndarray:
    """(layout.total, 3) LED colours 0–255 for one dancer at time t (ms)."""
    frame = np.zeros((layout.total, 3))
    for clip in active_clips(show, dancer, t):
        tl = t - clip["startMs"]
        for zid, start, count, pts in layout.zones:
            if clip["zones"] and zid not in clip["zones"]:
                continue
            for i in range(count):
                r, g, b, a = eval_effect(clip, tl, i, count, pts[i])
                if a <= 0:
                    continue
                o = frame[start + i]
                o += (np.array((r, g, b)) - o) * a
    for s in show.get("sections") or []:
        if s["startMs"] <= t < s["endMs"] and s["darkZones"]:
            for zid, start, count, _ in layout.zones:
                if zid in s["darkZones"]:
                    frame[start:start + count] = 0
    return frame
