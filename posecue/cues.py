"""
Cue generation: music analysis + detected moves → LumaCue clips.

Two layers, both ordinary LumaCue clips the light engineer can edit:

1. Base layer (whole costume), one look per 4-bar phrase, following LumaCue's
   own autocue rules (lib/show/autocue.ts): calm parts breathe (pulse /
   twinkle, one breath per bar), building parts chase (one lap per bar), high
   parts chase faster or cycle a rainbow, and a drop gets a beat of blackout
   then a bar of strobe on the beat. PoseCue picks each section's level from
   the music AND how hard the dancer is moving in it, and dims the base colour
   so the move cues stand out.

2. Move layer, on the zones of the body part that moved:
     arm raise  wipe along that arm (+ hand band), then hold while it's up
     jump       a band of light rises up the whole body ('rise')
     spin       a band sweeps across the body ('sweep'; rainbow when energetic)
     hit        a short accent flash on that limb
     stillness  everything dark except a slow chest breath
   Move starts within 120 ms of a beat are snapped onto it: dancers hit the
   beat, and pose timing jitters by a frame or two.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from posecue.audio import AudioAnalysis
from posecue.moves import Analysis, Move
from posecue.render import ALL_DANCERS, speed_for
from posecue.zones import Costume

LEVELS = ("low", "mid", "high")
STYLES = ("calm", "balanced", "energetic")

# Same palettes as LumaCue's autocue (lib/show/autocue.ts PALETTES).
PALETTES: dict[str, list[str]] = {
    "neon": ["#00E5FF", "#FF00CC", "#7C3AED", "#FFFFFF"],
    "fire": ["#FF3300", "#FF8800", "#FFD000", "#FFFFFF"],
    "ice": ["#00E5FF", "#2563EB", "#A5F3FC", "#FFFFFF"],
    "jungle": ["#00FF66", "#FFD000", "#00E5FF", "#FFFFFF"],
    "royal": ["#7C3AED", "#FFD000", "#FF0055", "#FFFFFF"],
    "bench": ["#FF0000", "#00FF00", "#FF2000", "#FFB000"],
}
STYLE_PALETTE = {"calm": "ice", "balanced": "neon", "energetic": "fire"}
_BASE_DIM = {"calm": 0.45, "balanced": 0.55, "energetic": 0.75}
SNAP_MS = 120
MIN_CLIP_MS = 50


@dataclass
class CueOptions:
    style: str = "balanced"
    palette: str | None = None      # None → by style
    track: int = ALL_DANCERS        # -1 = every costume, 0.. = one dancer
    base: bool = True               # include the music-driven base layer
    offset_ms: float = 0.0          # song time = video time + offset
    seed: int = 7                   # clip ids are reproducible

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {', '.join(STYLES)}")
        if self.palette is not None and self.palette not in PALETTES:
            raise ValueError(f"palette must be one of {', '.join(PALETTES)}")


def dim(hex_color: str, k: float) -> str:
    n = int(hex_color[1:7], 16)
    r, g, b = (n >> 16) & 255, (n >> 8) & 255, n & 255
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(v * k))) for v in (r, g, b)))


def snap_to_beat(t: float, beats: list[float], tol: float = SNAP_MS) -> float:
    if not beats:
        return t
    import bisect

    i = bisect.bisect_left(beats, t)
    best = min((beats[j] for j in (i - 1, i) if 0 <= j < len(beats)), key=lambda b: abs(b - t))
    return best if abs(best - t) <= tol else t


def section_level(music_level: str, dance_energy: float, style: str) -> str:
    """Music level blended with how hard the dancer moves (0–1), shifted by style."""
    m = LEVELS.index(music_level)
    if dance_energy != dance_energy:   # NaN: no dancer found in this part
        combined = m
    else:
        d = 0 if dance_energy < 0.33 else 1 if dance_energy < 0.66 else 2
        combined = int((m + d) / 2 + 0.5)
    shift = -1 if style == "calm" else 1 if style == "energetic" else 0
    return LEVELS[max(0, min(2, combined + shift))]


class _Builder:
    def __init__(self, length_ms: float, track: int, seed: int):
        self.length = int(round(length_ms))
        self.track = track
        self.clips: list[dict] = []
        self._rng = random.Random(seed)
        self._ids: set[str] = set()

    def _id(self) -> str:
        while True:
            i = "".join(self._rng.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(7))
            if i not in self._ids:
                self._ids.add(i)
                return i

    def add(self, start: float, dur: float, effect: str, color: str, *, color2: str = "#000000",
            zones: list[str] | None = None, speed: float = 50) -> None:
        s = max(0, int(round(start)))
        d = min(self.length - s, int(round(dur)))
        if s >= self.length or d < MIN_CLIP_MS:
            return
        self.clips.append({
            "id": self._id(), "track": self.track, "startMs": s, "durationMs": d, "effect": effect,
            "color": color, "color2": color2, "zones": list(zones or []),
            "speed": round(max(0.0, min(100.0, speed)), 3),
        })


def generate_clips(audio: AudioAnalysis, analysis: Analysis, costume: Costume, length_ms: float,
                   opts: CueOptions) -> list[dict]:
    pal = PALETTES[opts.palette or STYLE_PALETTE[opts.style]]
    accent = pal[-1]
    beat = audio.beatMs
    bar = beat * 4
    bar_s = bar / 1000
    b = _Builder(length_ms, opts.track, opts.seed)
    off = opts.offset_ms

    strobe_on_beat = speed_for(1000 / beat / 2)
    breath_per_bar = speed_for(2 / bar_s)
    lap_per_bar = speed_for(2 / bar_s)
    two_laps_per_bar = speed_for(4 / bar_s)
    rainbow_per_bar = speed_for(4 / bar_s)
    k_dim = _BASE_DIM[opts.style]

    # ---------------- base layer ----------------
    if opts.base:
        for i, s in enumerate(audio.sections):
            level = section_level(s.level, analysis.energy_between(s.startMs - off, s.endMs - off), opts.style)
            color = dim(pal[i % (len(pal) - 1)], k_dim)
            nxt = dim(pal[(i + 1) % (len(pal) - 1)], k_dim)
            phrases = []
            p = s.startMs
            while p < s.endMs - beat:
                phrases.append((p, min(s.endMs, p + bar * 4)))
                p += bar * 4
            if not phrases:
                phrases.append((s.startMs, s.endMs))
            intro = i == 0 and s.startMs < bar
            if intro:
                b.add(0, min(bar * 2, s.endMs), "fade", "#000000", color2=color)
            for k, (p0, p1) in enumerate(phrases):
                start = min(bar * 2, p1) if intro and k == 0 else p0
                if level == "low":
                    if k % 2 == 0:
                        b.add(start, p1 - start, "pulse", color, speed=breath_per_bar)
                    else:
                        b.add(start, p1 - start, "twinkle", color, speed=45)
                elif level == "mid":
                    b.add(start, p1 - start, "chase", nxt if k % 2 else color, speed=lap_per_bar)
                else:
                    frm = start
                    if s.drop and k == 0:
                        b.add(s.startMs - beat, beat, "blackout", "#000000")
                        b.add(s.startMs, bar, "strobe", accent, speed=strobe_on_beat)
                        frm = s.startMs + bar
                    if k % 2 == 0 or opts.palette == "bench":
                        b.add(frm, p1 - frm, "chase", color if k % 2 == 0 else nxt, speed=two_laps_per_bar)
                    else:
                        b.add(frm, p1 - frm, "rainbow", "#FFFFFF", speed=rainbow_per_bar)
        outro = min(bar * 2, 4000, length_ms / 4)
        last = dim(pal[max(0, len(audio.sections) - 1) % (len(pal) - 1)], k_dim)
        b.add(length_ms - outro, outro, "fade", last, color2="#000000")

    # ---------------- move layer (added after, so it draws on top) ----------------
    colors = pal[:-1]

    def move_colors(t: float) -> list[str]:
        """Palette colours other than the base layer's at time t, so moves never vanish into it."""
        i = next((k for k, s in enumerate(audio.sections) if s.startMs <= t < s.endMs), 0)
        base = {i % len(colors), (i + 1) % len(colors)} if opts.base else set()
        rest = [c for k, c in enumerate(colors) if k not in base] or colors
        return rest + colors

    n_base = len(b.clips)
    last_hit: dict[str, float] = {}
    for m in analysis.moves:
        start = snap_to_beat(m.start_ms + off, audio.beats)
        end = m.end_ms + off
        dur = max(end - start, MIN_CLIP_MS)
        if m.kind == "arm_raise":
            zones = costume.zones_for(m.part, "hands")
            mc = move_colors(start)
            color = mc[0] if m.part == "arm_l" else mc[1] if mc[1] != mc[0] else accent
            rise = max(150.0, min(beat, dur))
            b.add(start, rise, "wipe", color, zones=zones)
            hold = min(dur - rise, bar * 4)
            if hold >= 100:
                b.add(start + rise, hold, "solid", color, zones=zones)
        elif m.kind == "jump":
            d = max(dur, beat)
            b.add(start, d, "rise", accent, speed=speed_for(2000 / d))
        elif m.kind == "spin":
            d = max(dur, beat)
            if opts.style == "energetic" and opts.palette != "bench":
                b.add(start, d, "rainbow", "#FFFFFF", speed=rainbow_per_bar)
            else:
                b.add(start, d, "sweep", move_colors(start)[0], speed=speed_for(2000 / d))
        elif m.kind == "hit":
            if opts.style == "calm" and m.strength < 0.5:
                continue
            if start - last_hit.get(m.part, -1e9) < beat / 2:
                continue
            last_hit[m.part] = start
            b.add(start, 150, "solid", accent, zones=costume.zones_for(m.part))
        elif m.kind == "still":
            chest = costume.zones_for("chest")
            others = [z for z in costume.zone_ids() if z not in chest]
            b.add(start, dur, "blackout", "#000000", zones=others)
            b.add(start, dur, "pulse", dim(move_colors(start)[0], 0.7), zones=chest, speed=breath_per_bar)
    return _keep_moves_on_top(b, n_base)


def _keep_moves_on_top(b: _Builder, n_base: int) -> list[dict]:
    """LumaCue draws clips on a track in order of start time, so a base clip that starts while a
    move clip is still running would cover it. Steady move clips are cut at those base starts
    (the pieces look the same); moving effects are short, so they are only trimmed in rare cases."""
    base, moves = b.clips[:n_base], b.clips[n_base:]
    starts = sorted({c["startMs"] for c in base})
    out = list(base)
    for c in moves:
        end = c["startMs"] + c["durationMs"]
        cuts = [t for t in starts if c["startMs"] < t < end]
        if not cuts or c["effect"] not in ("solid", "blackout", "pulse", "wipe"):
            out.append(c)
            continue
        edges = [c["startMs"], *cuts, end]
        for k, (s0, s1) in enumerate(zip(edges, edges[1:])):
            if s1 - s0 < MIN_CLIP_MS:
                continue
            piece = dict(c, startMs=s0, durationMs=s1 - s0)
            if c["effect"] == "wipe" and k > 0:
                piece["effect"] = "solid"   # the wipe already filled the strip
            if k > 0:
                piece["id"] = b._id()
            out.append(piece)
    return out


def show_sections(audio: AudioAnalysis, analysis: Analysis, length_ms: float, opts: CueOptions) -> list[dict]:
    """Named parts of the show (LumaCue ShowSection). No zones are forced dark."""
    rng = random.Random(opts.seed + 1)
    out = []
    for i, s in enumerate(audio.sections):
        level = section_level(s.level, analysis.energy_between(s.startMs - opts.offset_ms, s.endMs - opts.offset_ms), opts.style)
        name = f"Part {i + 1}: {level}{' (drop)' if s.drop and level == 'high' else ''}"
        start, end = int(round(s.startMs)), int(round(min(s.endMs, length_ms)))
        if end > start:
            out.append({"id": "".join(rng.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(7)),
                        "name": name, "startMs": start, "endMs": end, "darkZones": []})
    return out


def describe_moves(moves: list[Move]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in moves:
        key = f"{m.kind}:{m.part}" if m.kind in ("arm_raise", "hit") else m.kind
        counts[key] = counts.get(key, 0) + 1
    return counts
