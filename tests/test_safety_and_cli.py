import json

import numpy as np
import pytest
from scipy.io import wavfile

from posecue.cli import main
from posecue.export import validate_show
from posecue.safety import FileCheckError, check_input, check_output, safe_display_name
from posecue.synth import demo_script, synth_music, synth_pose


@pytest.mark.parametrize("raw", ["http://example.com/a.mp4", "pipe:0", "concat:a.mp4|b.mp4", "NUL", "con.mp4", ""])
def test_urls_protocols_and_devices_are_refused(raw):
    with pytest.raises(FileCheckError):
        check_input(raw, {".mp4"}, 10**9, "video")


def test_input_checks(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x" * 10)
    (tmp_path / "empty.mp4").write_bytes(b"")
    (tmp_path / "a.txt").write_bytes(b"x")
    assert check_input(str(tmp_path / "a.mp4"), {".mp4"}, 100, "video").name == "a.mp4"
    for name, limit in (("empty.mp4", 100), ("a.txt", 100), ("a.mp4", 5), ("missing.mp4", 100)):
        with pytest.raises(FileCheckError):
            check_input(str(tmp_path / name), {".mp4"}, limit, "video")
    with pytest.raises(FileCheckError):
        check_input(str(tmp_path), {".mp4"}, 100, "video")    # a folder


def test_output_checks(tmp_path):
    src = tmp_path / "in.json"
    src.write_text("{}")
    with pytest.raises(FileCheckError):
        check_output(str(src), {".json"}, [src.resolve()], True, "output")       # overwrite an input
    with pytest.raises(FileCheckError):
        check_output(str(tmp_path / "x.exe"), {".json"}, [], False, "output")    # wrong type
    with pytest.raises(FileCheckError):
        check_output(str(tmp_path / "nope" / "x.json"), {".json"}, [], False, "output")
    (tmp_path / "old.json").write_text("{}")
    with pytest.raises(FileCheckError):
        check_output(str(tmp_path / "old.json"), {".json"}, [], False, "output")   # needs --force
    assert check_output(str(tmp_path / "old.json"), {".json"}, [], True, "output")


def test_display_names_are_cleaned():
    assert safe_display_name("C:\\Users\\me\\Music\\song.mp3") == "song.mp3"
    assert safe_display_name("bad\x00\x1bname") == "badname"
    assert safe_display_name("   ") == "PoseCue show"
    assert len(safe_display_name("x" * 500)) == 80


def test_pose_file_must_not_contain_pickles(tmp_path):
    from posecue.landmarks import PoseSequence

    np.savez(tmp_path / "evil.npz", times_ms=np.array([object()], dtype=object), xy=np.zeros(1),
             visibility=np.zeros(1), meta=np.zeros(2))
    with pytest.raises(ValueError):
        PoseSequence.load(tmp_path / "evil.npz")


def test_cli_demo_writes_valid_show(tmp_path):
    out = tmp_path / "demo.lumacue.json"
    moves = tmp_path / "moves.json"
    assert main(["--demo", "-o", str(out), "--moves", str(moves)]) == 0
    show = json.loads(out.read_text(encoding="utf-8"))
    assert validate_show(show) == []
    assert show["dancers"] == 10 and show["costume"]["zones"][1]["name"] == "Left arm"
    m = json.loads(moves.read_text())
    assert {x["kind"] for x in m["moves"]} >= {"arm_raise", "jump", "spin", "hit"}
    # second run without --force refuses to overwrite
    assert main(["--demo", "-o", str(out)]) == 2


def test_cli_from_saved_pose_and_wav(tmp_path):
    dur, steps, music = demo_script()
    synth_pose(dur, steps).save(tmp_path / "pose.npz")
    wavfile.write(tmp_path / "song.wav", 22050, synth_music(music))
    out = tmp_path / "show.json"
    code = main([str(tmp_path / "x.mp4"), "--pose", str(tmp_path / "pose.npz"), "--audio", str(tmp_path / "song.wav"),
                 "-o", str(out), "--style", "energetic", "--costume", "bench", "--track", "2", "--dancers", "4"])
    assert code == 2   # x.mp4 doesn't exist
    code = main(["--pose", str(tmp_path / "pose.npz"), "--audio", str(tmp_path / "song.wav"),
                 "-o", str(out), "--style", "energetic", "--costume", "bench", "--track", "2", "--dancers", "4"])
    assert code == 0
    show = json.loads(out.read_text(encoding="utf-8"))
    assert validate_show(show) == []
    assert {c["track"] for c in show["clips"]} == {1}
    assert show["audioName"] == "song.wav"
    assert [z["id"] for z in show["costume"]["zones"]] == ["a", "b", "c"]


@pytest.mark.parametrize("args", [
    ["--demo", "-o", "x.json", "--dancers", "0"],
    ["--demo", "-o", "x.json", "--track", "11"],
    ["--demo", "-o", "x.json", "--frame-step", "99"],
    ["-o", "x.json"],
    ["--demo", "-o", "x.txt"],
])
def test_cli_rejects_bad_arguments(tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)
    assert main(args) == 2


def test_tracking_without_model_gives_clear_error(tmp_path):
    from posecue.pose import track_video

    with pytest.raises(FileNotFoundError, match="download_models"):
        track_video(tmp_path / "v.mp4", tmp_path / "missing.task")


def test_preview_renders_without_video(tmp_path):
    import cv2

    from posecue.preview import render_preview
    from posecue.synth import Step
    from posecue.zones import get_costume

    seq = synth_pose(2, [Step("arm_raise", 0.2, 1.5, "arm_l")])
    show = {"lengthMs": 2000, "clips": [{"id": "x", "track": -1, "startMs": 0, "durationMs": 2000, "effect": "solid",
                                          "color": "#FF0000", "color2": "#000000", "zones": ["b"], "speed": 50}],
            "sections": []}
    out = tmp_path / "p.mp4"
    render_preview(out, show, get_costume("split"), seq, [], height=240)
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[0] == 240
    right = frame[:, -144:]
    assert right[..., 2].max() > 200          # the red left-arm LEDs are lit
