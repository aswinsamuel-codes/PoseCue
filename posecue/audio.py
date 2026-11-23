"""
Music analysis: tempo, beat grid, bars, energy sections, drops and strong hits.

This is a NumPy port of LumaCue Show Studio's own analysis
(lumacue/lib/show/analyze.ts), so a cue sheet made by PoseCue sits on the same
beat grid the Show Studio would find for the same song. Same method, same
constants: full-band and bass energy every ~10 ms → onset strength from rises
in log energy → tempo by autocorrelation (70–190 BPM, gently preferring 120)
→ beats by dynamic programming → bars from bass hits → energy per 4-bar phrase.

Decoding uses ffmpeg (the copy bundled with imageio-ffmpeg, or one on PATH),
called with an argument list (never a shell) and the input forced to the
`file:` protocol. WAV files are read directly with SciPy when ffmpeg is absent.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from posecue.safety import MAX_DURATION_S

SAMPLE_RATE = 22050
TARGET_HOP_S = 0.01
TIGHTNESS = 100


@dataclass
class Section:
    startMs: float
    endMs: float
    level: str        # 'low' | 'mid' | 'high'
    energy: float     # 0–1, relative to the rest of the song
    drop: bool


@dataclass
class AudioAnalysis:
    bpm: float
    beatMs: float
    beats: list[float]
    downbeats: list[float]
    sections: list[Section]
    accents: list[float]
    lengthMs: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def find_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 samples at `sr` Hz from any audio or video file ffmpeg can read."""
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file",
               "-i", f"file:{Path(path).resolve()}", "-t", str(MAX_DURATION_S),
               "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(f"could not decode audio: {msg[-1] if msg else 'ffmpeg failed'}")
        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        if samples.size == 0:
            raise RuntimeError("the file has no audio track; pass the song with --audio")
        return samples.copy()
    if Path(path).suffix.lower() == ".wav":
        return _load_wav(path, sr)
    raise RuntimeError("ffmpeg not found: install imageio-ffmpeg (pip) or use a .wav file")


def _load_wav(path: Path, sr: int) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, data = wavfile.read(path)
    x = data.astype(np.float32)
    if data.dtype.kind in "iu":
        x /= float(np.iinfo(data.dtype).max)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x[: int(MAX_DURATION_S * rate)]
    if rate != sr:
        g = np.gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
    return x


def _percentile(values, p: float) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    if not len(v):
        return 0.0
    return float(v[min(len(v) - 1, max(0, int(np.floor((p / 100) * (len(v) - 1)))))])


def analyze_audio(samples: np.ndarray, sr: int = SAMPLE_RATE) -> AudioAnalysis:
    x = np.asarray(samples, dtype=np.float64)
    hop = max(1, round(sr * TARGET_HOP_S))
    hop_s = hop / sr
    frames = len(x) // hop
    length_ms = len(x) / sr * 1000
    if frames < 50:
        raise ValueError("audio is too short to analyse (under half a second)")
    x = x[: frames * hop]

    # ---- energy per 10 ms frame (full band and bass below ~150 Hz) ----
    lp_a = np.exp(-2 * np.pi * 150 / sr)
    low = lfilter([1 - lp_a], [1, -lp_a], x)
    e_full = (x * x).reshape(frames, hop).mean(axis=1)
    e_low = (low * low).reshape(frames, hop).mean(axis=1)

    # ---- onset strength: rises in log energy, minus the local average ----
    d_full = np.diff(np.log(1e-9 + e_full), prepend=np.log(1e-9 + e_full[0]))
    d_low = np.diff(np.log(1e-9 + e_low), prepend=np.log(1e-9 + e_low[0]))
    raw_low = np.maximum(0, d_low)
    raw = np.maximum(0, d_full) + raw_low
    raw[0] = raw_low[0] = 0
    w = 25
    prefix = np.concatenate([[0.0], np.cumsum(raw)])
    idx = np.arange(frames)
    lo_i = np.maximum(0, idx - w)
    hi_i = np.minimum(frames, idx + w + 1)
    onset = np.maximum(0, raw - (prefix[hi_i] - prefix[lo_i]) / (hi_i - lo_i))

    # ---- tempo: autocorrelation over 70–190 BPM, preferring ~120 ----
    min_lag = int(np.floor(60 / 190 / hop_s))
    max_lag = int(np.ceil(60 / 70 / hop_s))
    n_fft = 1 << int(np.ceil(np.log2(2 * frames)))
    spec = np.fft.rfft(onset, n_fft)
    full_ac = np.fft.irfft(spec * np.conj(spec), n_fft)[: max_lag * 2 + 2]
    r = np.zeros(max_lag * 2 + 2)
    for L in range(min_lag, len(r)):
        r[L] = full_ac[L] / max(1, frames - L) if L < frames else 0.0
    best_lag, best_score = 50, -1.0
    for L in range(min_lag, max_lag + 1):
        bpm = 60 / (L * hop_s)
        prior = np.exp(-0.5 * (np.log2(bpm / 120) / 0.9) ** 2)
        score = (r[L] + 0.5 * (r[2 * L] if 2 * L < len(r) else 0)) * prior
        if score > best_score:
            best_score, best_lag = score, L
    y0, y1, y2 = r[best_lag - 1], r[best_lag], r[best_lag + 1]
    denom = y0 - 2 * y1 + y2
    period = best_lag + max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom)) if best_score > 0 and denom != 0 else 50.0

    # ---- beats: best path of onsets about one period apart (Ellis-style DP) ----
    sd = float(np.std(onset)) or 1.0
    score = np.zeros(frames)
    back = np.full(frames, -1, dtype=np.int64)
    min_gap = max(1, round(period / 2))
    max_gap = round(period * 2)
    gaps = np.arange(min_gap, max_gap + 1)
    penalty = TIGHTNESS * np.log(gaps / period) ** 2
    norm_onset = onset / sd
    for t in range(frames):
        taus = t - gaps
        ok = taus >= 0
        if ok.any():
            vals = score[taus[ok]] - penalty[ok]
            k = int(np.argmax(vals))
            best = vals[k]
            arg = int(taus[ok][k])
            score[t] = norm_onset[t] + max(0.0, best)
            back[t] = arg if best > 0 else -1
        else:
            score[t] = norm_onset[t]
    end = max(0, frames - 1)
    for t in range(max(0, frames - int(np.ceil(period))), frames):
        if score[t] > score[end]:
            end = t
    beat_frames = []
    t = end
    while t >= 0:
        beat_frames.append(t)
        t = back[t]
    beat_frames.reverse()
    beats = [f * hop_s * 1000 for f in beat_frames]
    intervals = np.diff(beats)
    median = _percentile(intervals, 50) if len(intervals) else period * hop_s * 1000
    span = (beats[-1] - beats[0]) / (len(beats) - 1) if len(beats) > 1 else median
    beat_ms = span if abs(span - median) < median * 0.03 else median

    # ---- bars: which of the 4 beat positions carries the most bass hits ----
    bar_offset, bar_score = 0, -1.0
    for j in range(4):
        s = sum(raw_low[beat_frames[k]] for k in range(j, len(beat_frames), 4))
        if s > bar_score:
            bar_score, bar_offset = s, j
    downbeats = [b for k, b in enumerate(beats) if k >= bar_offset and (k - bar_offset) % 4 == 0]

    # ---- section energy per 4-bar phrase ----
    bar_ms = beat_ms * 4
    cum_e = np.concatenate([[0.0], np.cumsum(e_full)])

    def rms_db(a: float, b: float) -> float:
        fa = max(0, int(np.floor(a / 1000 / hop_s)))
        fb = min(frames, int(np.ceil(b / 1000 / hop_s)))
        s = cum_e[fb] - cum_e[fa] if fb > fa else 0.0
        return 10 * np.log10(1e-9 + s / max(1, fb - fa))

    phrase_starts = [d for i, d in enumerate(downbeats) if i % 4 == 0]
    if not phrase_starts or phrase_starts[0] > 0:
        phrase_starts.insert(0, 0.0)
    phrases = []
    for i, start in enumerate(phrase_starts):
        stop = phrase_starts[i + 1] if i + 1 < len(phrase_starts) else length_ms
        if stop - start > bar_ms / 4:
            phrases.append((start, stop, rms_db(start, stop)))
    lo = _percentile([p[2] for p in phrases], 10)
    hi = _percentile([p[2] for p in phrases], 95)

    def level_of(e: float) -> str:
        return "low" if e < 0.4 else "mid" if e < 0.75 else "high"

    sections: list[Section] = []
    for start, stop, db in phrases:
        energy = max(0.0, min(1.0, (db - lo) / max(1e-6, hi - lo)))
        level = level_of(energy)
        last = sections[-1] if sections else None
        if last and last.level == level:
            w1, w2 = last.endMs - last.startMs, stop - start
            last.energy = (last.energy * w1 + energy * w2) / (w1 + w2)
            last.endMs = stop
        else:
            sections.append(Section(start, stop, level, energy, False))
    for i in range(1, len(sections)):
        prev, s = sections[i - 1], sections[i]
        s.drop = s.level == "high" and prev.level != "high" and s.energy - prev.energy > 0.25
    if sections:
        sections[0].startMs = 0.0
        sections[-1].endMs = length_ms

    # ---- accents: strongest onset peaks, at least 250 ms apart ----
    pos = onset[onset > 0]
    threshold = _percentile(pos, 97) if len(pos) else 0.0
    accents: list[float] = []
    for f in range(15, frames - 15):
        v = onset[f]
        if v < threshold or v <= 0:
            continue
        if onset[f - 15 : f + 16].max() > v:
            continue
        ms = f * hop_s * 1000
        if not accents or ms - accents[-1] >= 250:
            accents.append(ms)

    return AudioAnalysis(60000 / beat_ms, beat_ms, beats, downbeats, sections, accents, length_ms)
