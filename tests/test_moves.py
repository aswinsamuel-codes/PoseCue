import numpy as np
import pytest

from posecue.landmarks import L_WRIST, PoseSequence
from posecue.moves import analyze_moves
from posecue.synth import Step, synth_pose


def kinds(moves, kind, part=None):
    return [m for m in moves if m.kind == kind and (part is None or m.part == part)]


def test_demo_routine_finds_every_scripted_move(demo, demo_pose):
    _, steps, _ = demo
    moves = analyze_moves(demo_pose).moves
    for s in steps:
        if s.kind == "still":
            continue
        kind = {"punch": "hit"}.get(s.kind, s.kind)
        part = s.part if kind in ("arm_raise", "hit") else None
        found = [m for m in kinds(moves, kind, part)
                 if s.start_s * 1000 - 300 <= m.start_ms <= (s.start_s + s.dur_s) * 1000]
        assert found, f"missed {s}"


def test_left_arm_raise_is_the_dancers_left():
    seq = synth_pose(6, [Step("arm_raise", 2.0, 2.0, "arm_l")])
    moves = analyze_moves(seq).moves
    raises = kinds(moves, "arm_raise")
    assert [m.part for m in raises] == ["arm_l"]
    m = raises[0]
    assert 2000 <= m.start_ms <= 2600 and 3400 <= m.end_ms <= 4100
    # the dancer's left wrist is on the image's right when they face the camera
    assert np.nanmean(seq.xy[:, L_WRIST, 0]) > 0.5


def test_mirrored_video_swaps_sides():
    seq = synth_pose(6, [Step("arm_raise", 2.0, 2.0, "arm_l")]).mirrored()
    assert [m.part for m in kinds(analyze_moves(seq).moves, "arm_raise")] == ["arm_r"]


def test_jump_detected_without_arm_hits():
    moves = analyze_moves(synth_pose(6, [Step("jump", 3.0, 0.5)])).moves
    jumps = kinds(moves, "jump")
    assert len(jumps) == 1 and 2900 <= jumps[0].start_ms <= 3300
    assert not kinds(moves, "hit")
    assert not kinds(moves, "arm_raise")


def test_spin_detected_from_shoulder_flip():
    moves = analyze_moves(synth_pose(6, [Step("spin", 2.0, 1.0)])).moves
    spins = kinds(moves, "spin")
    assert len(spins) == 1
    assert 1900 <= spins[0].start_ms <= 2600 and spins[0].end_ms <= 3300


def test_punch_is_a_hit_on_that_arm():
    moves = analyze_moves(synth_pose(6, [Step("punch", 3.0, 1.0, "arm_r")])).moves
    hits = kinds(moves, "hit")
    assert [h.part for h in hits] == ["arm_r"]
    assert 3000 <= hits[0].start_ms <= 3300


def test_stillness_and_energy():
    seq = synth_pose(10, [Step("still", 0.0, 4.0)], groove=0.3)
    a = analyze_moves(seq)
    still = kinds(a.moves, "still")
    assert len(still) == 1 and still[0].start_ms == 0 and 3000 <= still[0].end_ms <= 4300
    assert a.energy_between(500, 3000) < 0.1
    assert a.energy_between(5000, 9500) > a.energy_between(500, 3000) + 0.2


def test_scale_invariance_small_dancer():
    """Same routine, dancer at a third of the size (wide stage shot): same moves."""
    seq = synth_pose(6, [Step("arm_raise", 2.0, 2.0, "arm_r"), Step("jump", 4.5, 0.5)])
    small = seq.xy.copy()
    small = 0.5 + (small - 0.5) / 3
    seq_small = PoseSequence(seq.times_ms, small, seq.visibility, seq.aspect, seq.fps)
    big = [(m.kind, m.part) for m in analyze_moves(seq).moves]
    little = [(m.kind, m.part) for m in analyze_moves(seq_small).moves]
    assert big == little


def test_missing_frames_and_noise_are_tolerated():
    seq = synth_pose(8, [Step("arm_raise", 2.0, 3.0, "arm_l")], noise=0.002, seed=3)
    xy = seq.xy.copy()
    xy[30:33] = np.nan           # dancer lost for 3 frames
    seq = PoseSequence(seq.times_ms, xy, seq.visibility, seq.aspect, seq.fps)
    raises = kinds(analyze_moves(seq).moves, "arm_raise", "arm_l")
    assert len(raises) == 1


def test_no_dancer_gives_no_moves():
    seq = synth_pose(3, [])
    empty = PoseSequence(seq.times_ms, np.full_like(seq.xy, np.nan), np.zeros_like(seq.visibility), seq.aspect, seq.fps)
    a = analyze_moves(empty)
    assert a.moves == []


def test_pose_cache_roundtrip(tmp_path):
    seq = synth_pose(2, [])
    seq.save(tmp_path / "p.npz")
    back = PoseSequence.load(tmp_path / "p.npz")
    assert np.allclose(back.xy, seq.xy) and back.fps == seq.fps


def test_bad_pose_arrays_rejected():
    with pytest.raises(ValueError):
        PoseSequence(np.zeros(3), np.zeros((3, 10, 2)), np.zeros((3, 33)))


def test_tracker_left_right_flicker_is_repaired():
    """Pose models sometimes swap every left/right label for single frames (front/back confusion).
    That must not become a fake spin or move the arm raise to the wrong side."""
    seq = synth_pose(8, [Step("arm_raise", 2.0, 3.0, "arm_l")])
    flicker = seq.mirrored()                     # mirrored() swaps labels; undo its x flip to keep positions
    flicker.xy[..., 0] = 1 - flicker.xy[..., 0]
    xy = seq.xy.copy()
    for i in [i for i in range(32, 72) if i % 4 in (0, 1)]:   # half the frames of the raise are label-swapped
        xy[i] = flicker.xy[i]
    moves = analyze_moves(PoseSequence(seq.times_ms, xy, seq.visibility, seq.aspect, seq.fps)).moves
    assert not kinds(moves, "spin")
    assert [m.part for m in kinds(moves, "arm_raise")] == ["arm_l"]
