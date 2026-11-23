import numpy as np
import pytest

from posecue.cues import CueOptions, dim, generate_clips, section_level, snap_to_beat
from posecue.moves import Analysis, Move, analyze_moves
from posecue.render import ALL_DANCERS, Layout, render_dancer
from posecue.zones import get_costume


def empty_analysis(length_ms=32000):
    t = np.arange(0, length_ms, 1000 / 15)
    return Analysis([], np.full(len(t), np.nan), t, 0.18)


def with_moves(moves, length_ms=32000, energy=0.5):
    t = np.arange(0, length_ms, 1000 / 15)
    return Analysis(moves, np.full(len(t), energy), t, 0.18)


def test_snap_to_beat():
    beats = [0, 500, 1000, 1500]
    assert snap_to_beat(540, beats) == 500
    assert snap_to_beat(760, beats) == 760      # too far from any beat
    assert snap_to_beat(1490, []) == 1490


def test_section_level_blends_music_dance_and_style():
    assert section_level("low", 0.9, "balanced") == "mid"
    assert section_level("high", 0.1, "balanced") == "mid"
    assert section_level("high", 0.9, "calm") == "mid"
    assert section_level("low", 0.1, "energetic") == "mid"
    assert section_level("mid", float("nan"), "balanced") == "mid"


def test_dim():
    assert dim("#FF8000", 0.5) == "#804000"


@pytest.mark.parametrize("costume,expect", [("split", ["a", "b"]), ("full", ["a", "b"]), ("bench", ["a"])])
def test_left_arm_raise_drives_left_arm_zone(demo_audio, costume, expect):
    c = get_costume(costume)
    a = with_moves([Move("arm_raise", "arm_l", 4000, 6000, 0.8)])
    clips = generate_clips(demo_audio, a, c, 32000, CueOptions(base=False))
    wipes = [x for x in clips if x["effect"] == "wipe"]
    assert len(wipes) == 1 and wipes[0]["zones"] == expect
    assert any(x["effect"] == "solid" and x["zones"] == expect for x in clips)


def test_split_costume_left_and_right_are_different_zones(demo_audio):
    c = get_costume("split")
    a = with_moves([Move("arm_raise", "arm_l", 4000, 6000, 0.8), Move("arm_raise", "arm_r", 9000, 11000, 0.8),
                    Move("hit", "leg_r", 14000, 14150, 0.9)])
    clips = generate_clips(demo_audio, a, c, 32000, CueOptions(base=False))
    zones = [tuple(x["zones"]) for x in clips if x["effect"] == "wipe"]
    assert zones == [("a", "b"), ("a", "g")]
    hit = [x for x in clips if x["startMs"] >= 13800 and x["effect"] == "solid"]
    assert hit and hit[0]["zones"] == ["h"] and hit[0]["durationMs"] == 150


def test_effects_by_move_type(demo_audio):
    c = get_costume("split")
    a = with_moves([Move("jump", "body", 5000, 5500, 0.8), Move("spin", "body", 10000, 11500, 0.9),
                    Move("still", "body", 20000, 23000, 1.0)])
    clips = generate_clips(demo_audio, a, c, 32000, CueOptions(base=False))
    eff = {x["effect"] for x in clips}
    assert {"rise", "sweep", "blackout", "pulse"} <= eff
    blackout = next(x for x in clips if x["effect"] == "blackout")
    assert "c" not in blackout["zones"] and len(blackout["zones"]) == 7     # chest keeps breathing
    energetic = generate_clips(demo_audio, a, c, 32000, CueOptions(style="energetic", base=False))
    assert any(x["effect"] == "rainbow" for x in energetic)


def test_moves_snap_to_beats(demo_audio):
    beat = demo_audio.beats[10]
    a = with_moves([Move("jump", "body", beat + 80, beat + 600, 0.8)])
    clip = generate_clips(demo_audio, a, get_costume("split"), 32000, CueOptions(base=False))[0]
    assert clip["startMs"] == round(beat)


def test_style_changes_density(demo_audio, demo_pose):
    a = analyze_moves(demo_pose)
    c = get_costume("split")
    calm = generate_clips(demo_audio, a, c, 32000, CueOptions(style="calm"))
    hot = generate_clips(demo_audio, a, c, 32000, CueOptions(style="energetic"))
    fast = lambda cl: sum(x["effect"] in ("strobe", "rainbow", "chase") for x in cl)  # noqa: E731
    assert fast(hot) > fast(calm)


def test_clips_are_valid_and_inside_the_show(demo_audio, demo_pose):
    clips = generate_clips(demo_audio, analyze_moves(demo_pose), get_costume("split"), 32000, CueOptions())
    ids = [c["id"] for c in clips]
    assert len(ids) == len(set(ids))
    for c in clips:
        assert 0 <= c["startMs"] < 32000 and c["durationMs"] >= 50 and c["startMs"] + c["durationMs"] <= 32000
        assert c["track"] == ALL_DANCERS and 0 <= c["speed"] <= 100


def test_track_option_puts_cues_on_one_dancer(demo_audio):
    a = with_moves([Move("jump", "body", 5000, 5500, 0.8)])
    clips = generate_clips(demo_audio, a, get_costume("split"), 32000, CueOptions(track=3))
    assert {c["track"] for c in clips} == {3}


def test_offset_shifts_moves(demo_audio):
    a = with_moves([Move("jump", "body", 5000, 5500, 0.8)])
    clip = generate_clips(demo_audio, a, get_costume("split"), 32000, CueOptions(base=False, offset_ms=2000))[0]
    assert abs(clip["startMs"] - 7000) <= 120


def test_held_arm_stays_on_top_of_base_layer(demo_audio):
    """A base clip starting while the arm is up must not hide it (LumaCue layers by start time)."""
    c = get_costume("split")
    phrase = next(x for x in generate_clips(demo_audio, empty_analysis(), c, 32000, CueOptions()) if x["startMs"] > 6000)
    t0 = phrase["startMs"]
    a = with_moves([Move("arm_raise", "arm_l", t0 - 2000, t0 + 2000, 0.8)])
    show = {"clips": generate_clips(demo_audio, a, c, 32000, CueOptions()), "sections": []}
    layout = Layout(c)
    frame = render_dancer(show, layout, 0, t0 + 1000)
    zid, start, count, _ = next(z for z in layout.zones if z[0] == "b")
    arm = frame[start:start + count]
    solid = next(x for x in show["clips"] if x["effect"] == "solid" and x["zones"] == ["a", "b"])
    from posecue.render import hex_to_rgb

    assert np.allclose(arm, hex_to_rgb(solid["color"]), atol=1)


def test_bad_options():
    with pytest.raises(ValueError):
        CueOptions(style="wild")
    with pytest.raises(ValueError):
        CueOptions(palette="nope")
