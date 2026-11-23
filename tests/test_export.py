"""The show file must match LumaCue's ShowProject / SavedCostume exactly."""
import copy
import json
import os
import re
from pathlib import Path

import pytest

from posecue.cues import CueOptions, generate_clips, show_sections
from posecue.export import CLIP_KEYS, EFFECT_IDS, PLACEMENT_IDS, SECTION_KEYS, build_show, show_json, validate_show
from posecue.moves import analyze_moves
from posecue.zones import COSTUMES, get_costume

# Where a LumaCue checkout lives, to compare against its model.ts (tests skip without it).
LUMACUE = Path(os.environ.get("LUMACUE_DIR", Path(__file__).resolve().parents[3] / "Robotics-Embedded" / "LumaCue" / "lumacue"))
MODEL_TS = LUMACUE / "lib" / "show" / "model.ts"
BODY_TS = LUMACUE / "lib" / "show" / "body.ts"


@pytest.fixture(scope="module")
def show(demo_audio, demo_pose):
    c = get_costume("split")
    a = analyze_moves(demo_pose)
    o = CueOptions()
    clips = generate_clips(demo_audio, a, c, demo_audio.lengthMs, o)
    return build_show(name="Test", dancers=10, length_ms=demo_audio.lengthMs, audio_name="song.mp3", clips=clips,
                      sections=show_sections(demo_audio, a, demo_audio.lengthMs, o), costume=c.to_json())


def test_generated_show_validates(show):
    assert validate_show(show) == []
    assert list(show) == ["version", "name", "dancers", "lengthMs", "audioName", "clips", "sections", "costume"]
    assert all(list(c) == CLIP_KEYS for c in show["clips"])
    assert all(list(s) == SECTION_KEYS for s in show["sections"])
    assert json.loads(show_json(show)) == show


@pytest.mark.parametrize("name", sorted(COSTUMES))
def test_costumes_validate_and_have_unique_ids(name):
    c = get_costume(name)
    j = c.to_json()
    errs, ids = __import__("posecue.export", fromlist=["_validate_costume"])._validate_costume(j)
    assert errs == []
    assert len(ids) == len(j["zones"])
    for part in ("arm_l", "arm_r", "leg_l", "leg_r", "hands", "chest"):
        assert c.zones_for(part), part


def test_split_costume_keeps_full_suit_ids():
    """Ids a–f mean the same body parts as LumaCue's Full suit template, so clips survive a template switch."""
    split, full = get_costume("split"), get_costume("full")
    for z_full in full.zones:
        z_split = next(z for z in split.zones if z.id == z_full.id)
        assert z_split.placement == z_full.placement or z_split.placement == "custom"


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s["clips"][0].__setitem__("effect", "laser"), "unknown effect"),
    (lambda s: s["clips"][0].__setitem__("color", "red"), "#RRGGBB"),
    (lambda s: s["clips"][0].__setitem__("speed", 140), "speed"),
    (lambda s: s["clips"][0].__setitem__("startMs", 10**9), "outside"),
    (lambda s: s["clips"][0].__setitem__("zones", ["zz"]), "not in the costume"),
    (lambda s: s["clips"][1].__setitem__("id", s["clips"][0]["id"]), "repeated"),
    (lambda s: s["clips"][0].pop("color2"), "keys"),
    (lambda s: s.__setitem__("version", 2), "version"),
    (lambda s: s.__setitem__("dancers", 51), "dancers"),
    (lambda s: s.__setitem__("extra", 1), "unknown top-level"),
    (lambda s: s["costume"]["zones"][1]["paths"][0]["points"].append({"x": 500, "y": 1}), "outside the 100"),
    (lambda s: s["costume"]["zones"][0].__setitem__("leds", 0), "leds"),
])
def test_validator_catches_what_lumacue_would_repair(show, mutate, needle):
    bad = copy.deepcopy(show)
    mutate(bad)
    errs = validate_show(bad)
    assert any(needle in e for e in errs), errs


# ---------------- against LumaCue's own source ----------------

needs_lumacue = pytest.mark.skipif(not MODEL_TS.is_file(), reason="LumaCue checkout not found (set LUMACUE_DIR)")


def _interface_keys(src: str, name: str) -> list[str]:
    body = re.search(r"export interface " + name + r" \{(.*?)\n\}", src, re.S).group(1)
    return re.findall(r"^\s+(\w+)\??:", body, re.M)


@needs_lumacue
def test_clip_and_section_keys_match_model_ts():
    src = MODEL_TS.read_text(encoding="utf-8")
    assert _interface_keys(src, "Clip") == CLIP_KEYS
    assert _interface_keys(src, "ShowSection") == SECTION_KEYS
    show_keys = _interface_keys(src, "ShowProject")
    assert set(show_keys) == {"version", "name", "dancers", "lengthMs", "audioName", "audioUrl", "clips", "sections"}


@needs_lumacue
def test_effect_ids_match_model_ts():
    src = MODEL_TS.read_text(encoding="utf-8")
    union = re.search(r"export type EffectId = (.*)", src).group(1)
    assert set(re.findall(r"'(\w+)'", union)) == EFFECT_IDS


@needs_lumacue
def test_placements_match_body_ts():
    src = BODY_TS.read_text(encoding="utf-8")
    union = re.search(r"export type Placement = (.*)", src).group(1)
    assert set(re.findall(r"'(\w+)'", union)) == PLACEMENT_IDS
    # the preset LED lines PoseCue draws must be the ones Show Studio uses
    from posecue.zones import PLACEMENT_PATHS

    for pid in ("arms", "chest", "legs", "belt"):
        line = re.search(r"\{ id: '" + pid + r"'.*?paths: (\[.*?\]) \}", src).group(1)
        pts = [(int(x), int(y)) for x, y in re.findall(r"p\((\d+), (\d+)\)", line)]
        assert pts == [p for path in PLACEMENT_PATHS[pid] for p in path]
