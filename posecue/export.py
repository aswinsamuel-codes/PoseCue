"""
LumaCue show-file export and validation.

The output is a LumaCue `.lumacue.json` show file: the ShowProject from
lumacue/lib/show/model.ts plus the `costume` (SavedCostume) it was made for,
exactly as Show Studio's own "Save show" writes it (showFileJson). Show
Studio's parseShowProject() reads it back.

validate_show() is a strict mirror of parseShowProject()/parseCostume(): it
reports anything Show Studio would drop, repair or refuse, so PoseCue never
writes a file that loads with warnings.
"""
from __future__ import annotations

import json
import re

from posecue import __version__

SHOW_KEYS = ["version", "name", "dancers", "lengthMs", "audioName", "clips", "sections", "costume"]
OPTIONAL_SHOW_KEYS = {"audioUrl", "sections", "costume"}
CLIP_KEYS = ["id", "track", "startMs", "durationMs", "effect", "color", "color2", "zones", "speed"]
SECTION_KEYS = ["id", "name", "startMs", "endMs", "darkZones"]
EFFECT_IDS = {"solid", "fade", "pulse", "strobe", "chase", "wipe", "rainbow", "twinkle", "blackout",
              "rise", "sweep", "ripple"}
PLACEMENT_IDS = {"arms", "chest", "legs", "head", "belt", "hands", "custom"}
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
MAX_LENGTH_MS = 60 * 60 * 1000
MAX_DANCERS = 50
MAX_PATHS = 8
MAX_PATH_POINTS = 64


def build_show(*, name: str, dancers: int, length_ms: float, audio_name: str | None, clips: list[dict],
               sections: list[dict], costume: dict) -> dict:
    show = {
        "version": 1,
        "name": name,
        "dancers": int(dancers),
        "lengthMs": int(round(length_ms)),
        "audioName": audio_name,
        "clips": sorted(clips, key=lambda c: (c["startMs"], c["track"])),
    }
    if sections:
        show["sections"] = sorted(sections, key=lambda s: s["startMs"])
    show["costume"] = costume
    return show


def show_json(show: dict) -> str:
    return json.dumps(show, indent=2, ensure_ascii=False) + "\n"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def validate_show(show: dict) -> list[str]:
    """Problems Show Studio would report or silently fix. Empty list = loads cleanly."""
    errs: list[str] = []
    if not isinstance(show, dict):
        return ["not an object"]
    extra = set(show) - set(SHOW_KEYS) - OPTIONAL_SHOW_KEYS
    if extra:
        errs.append(f"unknown top-level keys {sorted(extra)}")
    for k in ("version", "name", "dancers", "lengthMs", "audioName", "clips"):
        if k not in show:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if show["version"] != 1:
        errs.append("version must be 1")
    if not isinstance(show["name"], str) or not show["name"].strip():
        errs.append("name must be a non-empty string")
    if not (isinstance(show["dancers"], int) and 1 <= show["dancers"] <= MAX_DANCERS):
        errs.append(f"dancers must be an integer 1–{MAX_DANCERS}")
    L = show["lengthMs"]
    if not (_num(L) and 0 < L <= MAX_LENGTH_MS):
        errs.append("lengthMs must be > 0 and at most 60 min")
        L = MAX_LENGTH_MS
    if show["audioName"] is not None and not isinstance(show["audioName"], str):
        errs.append("audioName must be a string or null")
    if "audioUrl" in show and not (isinstance(show["audioUrl"], str) and show["audioUrl"].startswith("https://")):
        errs.append("audioUrl must be an https:// link")

    zone_ids: set[str] | None = None
    if "costume" in show:
        cerrs, zone_ids = _validate_costume(show["costume"])
        errs += cerrs

    if not isinstance(show["clips"], list):
        return errs + ["clips must be a list"]
    ids: set[str] = set()
    for n, c in enumerate(show["clips"]):
        where = f"clip {n}"
        if not isinstance(c, dict):
            errs.append(f"{where}: not an object")
            continue
        if list(c) != CLIP_KEYS:
            errs.append(f"{where}: keys {list(c)} != {CLIP_KEYS}")
            continue
        if not (isinstance(c["id"], str) and c["id"]) or c["id"] in ids:
            errs.append(f"{where}: id missing or repeated")
        ids.add(c["id"])
        if not (isinstance(c["track"], int) and -1 <= c["track"] < MAX_DANCERS):
            errs.append(f"{where}: track must be -1 (all) or a dancer index")
        if not (_num(c["startMs"]) and 0 <= c["startMs"] < L):
            errs.append(f"{where}: startMs outside the show")
        if not (_num(c["durationMs"]) and c["durationMs"] > 0):
            errs.append(f"{where}: durationMs must be > 0")
        if c["effect"] not in EFFECT_IDS:
            errs.append(f"{where}: unknown effect {c['effect']!r}")
        for k in ("color", "color2"):
            if not (isinstance(c[k], str) and HEX.match(c[k])):
                errs.append(f"{where}: {k} must be #RRGGBB")
        if not (isinstance(c["zones"], list) and all(isinstance(z, str) for z in c["zones"])):
            errs.append(f"{where}: zones must be a list of zone ids")
        elif zone_ids is not None and not set(c["zones"]) <= zone_ids:
            errs.append(f"{where}: zones {sorted(set(c['zones']) - zone_ids)} are not in the costume")
        if not (_num(c["speed"]) and 0 <= c["speed"] <= 100):
            errs.append(f"{where}: speed must be 0–100")

    if "sections" in show:
        if not isinstance(show["sections"], list):
            errs.append("sections must be a list")
        else:
            for n, s in enumerate(show["sections"]):
                if not isinstance(s, dict) or list(s) != SECTION_KEYS:
                    errs.append(f"section {n}: keys must be {SECTION_KEYS}")
                    continue
                if not (_num(s["startMs"]) and _num(s["endMs"]) and 0 <= s["startMs"] < min(s["endMs"], L)):
                    errs.append(f"section {n}: bad start/end")
                if not (isinstance(s["name"], str) and s["name"].strip() and isinstance(s["id"], str) and s["id"]):
                    errs.append(f"section {n}: needs an id and a name")
                if not (isinstance(s["darkZones"], list) and all(isinstance(z, str) for z in s["darkZones"])):
                    errs.append(f"section {n}: darkZones must be a list of zone ids")
    return errs


def _validate_costume(c) -> tuple[list[str], set[str] | None]:
    errs: list[str] = []
    if not isinstance(c, dict) or not isinstance(c.get("zones"), list) or not c["zones"]:
        return ["costume must have a zones list"], None
    if set(c) != {"zones", "brightness"}:
        errs.append(f"costume keys must be zones and brightness, got {sorted(c)}")
    if not (isinstance(c.get("brightness"), int) and 5 <= c["brightness"] <= 100):
        errs.append("costume brightness must be an integer 5–100")
    ids: set[str] = set()
    for n, z in enumerate(c["zones"]):
        where = f"costume zone {n}"
        if not isinstance(z, dict):
            errs.append(f"{where}: not an object")
            continue
        if not (isinstance(z.get("id"), str) and isinstance(z.get("name"), str)):
            errs.append(f"{where}: needs a string id and name")
        elif z["id"] in ids:
            errs.append(f"{where}: repeated id {z['id']!r}")
        ids.add(z.get("id"))
        if not (isinstance(z.get("leds"), int) and 1 <= z["leds"] <= 300):
            errs.append(f"{where}: leds must be an integer 1–300")
        if z.get("placement") not in PLACEMENT_IDS:
            errs.append(f"{where}: unknown placement {z.get('placement')!r}")
        if z.get("placement") == "custom":
            paths = z.get("paths")
            if not (isinstance(paths, list) and 0 < len(paths) <= MAX_PATHS):
                errs.append(f"{where}: custom zones need 1–{MAX_PATHS} drawn paths")
                continue
            for p in paths:
                pts = p.get("points") if isinstance(p, dict) else None
                if p.get("side") not in ("front", "back") or not isinstance(pts, list) or not 0 < len(pts) <= MAX_PATH_POINTS:
                    errs.append(f"{where}: bad path")
                    continue
                for q in pts:
                    if not (isinstance(q, dict) and _num(q.get("x")) and _num(q.get("y"))
                            and 0 <= q["x"] <= 100 and 0 <= q["y"] <= 200):
                        errs.append(f"{where}: path point outside the 100 × 200 figure")
                        break
        elif "paths" in z:
            errs.append(f"{where}: only custom zones carry paths")
    return errs, ids


GENERATOR = f"PoseCue {__version__}"
