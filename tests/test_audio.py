import numpy as np
import pytest

from posecue.audio import analyze_audio
from posecue.synth import synth_music


@pytest.mark.parametrize("bpm", [96.0, 120.0, 140.0])
def test_tempo_and_beat_grid(bpm):
    a = analyze_audio(synth_music([(20.0, 0.8)], bpm=bpm))
    assert a.bpm == pytest.approx(bpm, rel=0.02)
    beat = 60000 / bpm
    # beats land on the kicks (every beat from 0), within ~one analysis frame
    phase = np.array(a.beats) % beat
    err = np.minimum(phase, beat - phase)
    assert np.median(err) < 25
    assert len(a.beats) >= 0.9 * 20000 / beat


def test_downbeats_are_every_fourth_beat():
    a = analyze_audio(synth_music([(16.0, 0.8)]))
    assert len(a.downbeats) >= 6
    gaps = np.diff(a.downbeats)
    assert np.allclose(gaps, 4 * a.beatMs, rtol=0.05)


def test_quiet_then_loud_sections_and_drop():
    a = analyze_audio(synth_music([(16.0, 0.12), (16.0, 1.0)]))
    assert a.sections[0].startMs == 0 and a.sections[-1].endMs == pytest.approx(a.lengthMs)
    assert a.sections[0].level == "low"
    assert a.sections[-1].level == "high"
    loud = [s for s in a.sections if s.level == "high"]
    assert loud[0].startMs == pytest.approx(16000, abs=2100)   # phrase boundary near the change
    assert loud[0].drop
    # sections tile the song without gaps
    for s0, s1 in zip(a.sections, a.sections[1:]):
        assert s0.endMs == s1.startMs


def test_accents_are_spaced():
    a = analyze_audio(synth_music([(12.0, 0.9)]))
    assert a.accents
    assert np.all(np.diff(a.accents) >= 250)


def test_too_short_audio_is_refused():
    with pytest.raises(ValueError):
        analyze_audio(np.zeros(1000, dtype=np.float32))
