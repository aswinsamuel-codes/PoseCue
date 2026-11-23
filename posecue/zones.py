"""
Costumes: which LumaCue zone each body part drives.

LumaCue zones are ids ('a', 'b' …) with a placement on the body (arms, chest,
legs, head, belt, hands, or lines drawn by hand: 'custom'). The figure is a
100 × 200 front view *as the audience sees the dancer*, so the dancer's own
left arm is on the right of the figure.

PoseCue ships three costumes. Each is written into the show file's `costume`
field, which Show Studio loads together with the show:

  split   (default) the LumaCue "Full suit" template with arms and legs split
          into left and right strips, so a left-arm raise lights only the
          left arm. Ids a–f are the Full suit's (b and d become the dancer's
          LEFT arm and leg), plus g = right arm and h = right leg.
  full    LumaCue's "Full suit" template exactly (ids a–f, both arms on one
          strip). Left- and right-arm moves both drive zone b.
  bench   LumaCue's 3-zone bench costume (a arms, b chest, c legs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Preset LED lines copied from lumacue/lib/show/body.ts (PLACEMENTS). Points are (x, y).
_ARM_AUDIENCE_LEFT = [(34, 45), (22, 78), (14, 106)]    # dancer's RIGHT arm
_ARM_AUDIENCE_RIGHT = [(66, 45), (78, 78), (86, 106)]   # dancer's LEFT arm
_LEG_AUDIENCE_LEFT = [(44, 112), (41, 152), (39, 190)]  # dancer's RIGHT leg
_LEG_AUDIENCE_RIGHT = [(56, 112), (59, 152), (61, 190)]  # dancer's LEFT leg


def _circle(cx: float, cy: float, r: float, segments: int = 18) -> list[tuple[float, float]]:
    import math

    return [(cx + math.cos(-math.pi / 2 + i / segments * 2 * math.pi) * r,
             cy + math.sin(-math.pi / 2 + i / segments * 2 * math.pi) * r) for i in range(segments + 1)]


PLACEMENT_PATHS: dict[str, list[list[tuple[float, float]]]] = {
    "arms": [_ARM_AUDIENCE_LEFT, _ARM_AUDIENCE_RIGHT],
    "chest": [[(37, 46), (50, 82), (63, 46)]],
    "legs": [_LEG_AUDIENCE_LEFT, _LEG_AUDIENCE_RIGHT],
    "head": [_circle(50, 21, 12)],
    "belt": [[(36, 104), (64, 104)]],
    "hands": [_circle(13, 113, 5, 12), _circle(87, 113, 5, 12)],
}
PLACEMENTS = tuple(PLACEMENT_PATHS) + ("custom",)


@dataclass
class Zone:
    id: str
    name: str
    leds: int
    placement: str
    paths: list[list[tuple[float, float]]] | None = None   # only for placement 'custom'

    def lines(self) -> list[list[tuple[float, float]]]:
        if self.placement == "custom" and self.paths:
            return self.paths
        return PLACEMENT_PATHS.get(self.placement, PLACEMENT_PATHS["belt"])

    def to_json(self) -> dict:
        d = {"id": self.id, "name": self.name, "leds": self.leds, "placement": self.placement}
        if self.placement == "custom" and self.paths:
            d["paths"] = [{"side": "front", "points": [{"x": float(x), "y": float(y)} for x, y in path]}
                          for path in self.paths]
        return d


@dataclass
class Costume:
    id: str
    label: str
    zones: list[Zone]
    # PoseCue body part → zone ids it lights. 'hands' / 'head' etc. are extra targets.
    part_zones: dict[str, list[str]] = field(default_factory=dict)
    brightness: int = 25

    def zone_ids(self) -> list[str]:
        return [z.id for z in self.zones]

    def zones_for(self, *parts: str) -> list[str]:
        """Zone ids lit by these parts, in costume order, without repeats."""
        wanted = {zid for p in parts for zid in self.part_zones.get(p, [])}
        return [z for z in self.zone_ids() if z in wanted]

    def to_json(self) -> dict:
        """LumaCue's SavedCostume."""
        return {"zones": [z.to_json() for z in self.zones], "brightness": self.brightness}


def _split() -> Costume:
    return Costume(
        "split", "Full suit, left/right split",
        [
            Zone("a", "Hand bands", 24, "hands"),
            Zone("b", "Left arm", 25, "custom", [_ARM_AUDIENCE_RIGHT]),
            Zone("c", "Chest V", 30, "chest"),
            Zone("d", "Left leg", 25, "custom", [_LEG_AUDIENCE_RIGHT]),
            Zone("e", "Hood", 24, "head"),
            Zone("f", "Belt", 16, "belt"),
            Zone("g", "Right arm", 25, "custom", [_ARM_AUDIENCE_LEFT]),
            Zone("h", "Right leg", 25, "custom", [_LEG_AUDIENCE_LEFT]),
        ],
        {
            "arm_l": ["b"], "arm_r": ["g"], "leg_l": ["d"], "leg_r": ["h"],
            "hands": ["a"], "chest": ["c"], "head": ["e"], "belt": ["f"],
        },
    )


def _full() -> Costume:
    return Costume(
        "full", "LumaCue Full suit",
        [
            Zone("a", "Hand bands", 24, "hands"),
            Zone("b", "Arms", 50, "arms"),
            Zone("c", "Chest V", 30, "chest"),
            Zone("d", "Legs", 50, "legs"),
            Zone("e", "Hood", 24, "head"),
            Zone("f", "Belt", 16, "belt"),
        ],
        {"arm_l": ["b"], "arm_r": ["b"], "leg_l": ["d"], "leg_r": ["d"],
         "hands": ["a"], "chest": ["c"], "head": ["e"], "belt": ["f"]},
    )


def _bench() -> Costume:
    return Costume(
        "bench", "LumaCue bench test (3 zones)",
        [Zone("a", "Zone A — arms", 20, "arms"), Zone("b", "Zone B — chest", 20, "chest"),
         Zone("c", "Zone C — legs", 20, "legs")],
        {"arm_l": ["a"], "arm_r": ["a"], "leg_l": ["c"], "leg_r": ["c"], "hands": ["a"], "chest": ["b"],
         "head": ["b"], "belt": ["b"]},
    )


COSTUMES = {"split": _split, "full": _full, "bench": _bench}


def get_costume(name: str) -> Costume:
    try:
        return COSTUMES[name]()
    except KeyError:
        raise ValueError(f"unknown costume {name!r}; choose from {', '.join(COSTUMES)}") from None
