import numpy as np
import pytest

from posecue.render import Layout, eval_effect, hex_to_rgb, lap_ms, render_dancer, speed_for, speed_hz
from posecue.zones import get_costume


def clip(**kw):
    c = {"id": "x", "track": -1, "startMs": 0, "durationMs": 1000, "effect": "solid", "color": "#FF0000",
         "color2": "#000000", "zones": [], "speed": 50}
    c.update(kw)
    return c


def test_speed_roundtrip():
    for hz in (0.5, 1, 2, 4):
        assert speed_hz(speed_for(hz)) == pytest.approx(hz)
    assert lap_ms(50) == pytest.approx(2000 / speed_hz(50))


def test_layout_counts():
    lay = Layout(get_costume("split"))
    assert lay.total == sum(z.leds for z in get_costume("split").zones)
    assert len(lay.points) == lay.total


def test_zone_limited_clip_lights_only_that_zone():
    c = get_costume("split")
    lay = Layout(c)
    show = {"clips": [clip(zones=["b"])], "sections": []}
    f = render_dancer(show, lay, 0, 500)
    for zid, start, count, _ in lay.zones:
        lit = f[start:start + count].sum()
        assert (lit > 0) == (zid == "b")
    # zone b is the dancer's LEFT arm, drawn on the audience's right (x > 50)
    _, start, count, pts = next(z for z in lay.zones if z[0] == "b")
    assert all(x > 50 for x, _ in pts)


def test_later_clip_draws_on_top_and_sections_darken():
    lay = Layout(get_costume("bench"))
    show = {"clips": [clip(color="#00FF00"), clip(startMs=100, color="#0000FF", zones=["a"])],
            "sections": [{"id": "s", "name": "S", "startMs": 0, "endMs": 1000, "darkZones": ["c"]}]}
    f = render_dancer(show, lay, 0, 500)
    assert np.allclose(f[0], hex_to_rgb("#0000FF"))        # zone a: top clip
    assert np.allclose(f[25], hex_to_rgb("#00FF00"))       # zone b: base clip
    assert f[45].sum() == 0                                # zone c: dark section


def test_wipe_and_rise_progress():
    c = clip(effect="wipe", durationMs=1000)
    assert eval_effect(c, 100, 0, 10, (0, 0))[3] == 1 and eval_effect(c, 100, 9, 10, (0, 0))[3] == 0
    assert eval_effect(c, 999, 9, 10, (0, 0))[3] == 1
    rise = clip(effect="rise", speed=speed_for(1))
    # early in the lap the band is at the feet, not the head
    feet = eval_effect(rise, 400, 0, 1, (50, 190))[3]
    head = eval_effect(rise, 400, 0, 1, (50, 20))[3]
    assert feet > head


def test_blackout_masks():
    lay = Layout(get_costume("bench"))
    show = {"clips": [clip(), clip(startMs=10, effect="blackout", color="#000000")], "sections": []}
    assert render_dancer(show, lay, 0, 500).sum() == 0
