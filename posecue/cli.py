"""
Command line:

  python -m posecue dance.mp4 -o show.lumacue.json [--audio song.mp3] [--style calm|balanced|energetic]
                    [--preview preview.mp4]
  python -m posecue --demo -o demo.lumacue.json --preview demo.mp4     (no video or model needed)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from posecue import __version__
from posecue.audio import analyze_audio, load_audio
from posecue.cues import PALETTES, STYLES, CueOptions, describe_moves, generate_clips, show_sections
from posecue.export import build_show, show_json, validate_show
from posecue.landmarks import PoseSequence
from posecue.moves import analyze_moves
from posecue.render import ALL_DANCERS
from posecue.safety import (
    AUDIO_EXT, MAX_AUDIO_BYTES, MAX_VIDEO_BYTES, POSE_EXT, VIDEO_EXT, FileCheckError,
    check_input, check_output, safe_display_name, write_text_atomic,
)
from posecue.zones import COSTUMES, get_costume

JSON_EXT = {".json"}
MP4_EXT = {".mp4"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="posecue",
        description="Film a dancer, get LED cues: writes a LumaCue show file from a dance video.")
    p.add_argument("video", nargs="?", help="dance video (mp4, mov, mkv, webm ...)")
    p.add_argument("-o", "--output", required=True, help="show file to write (.json, e.g. show.lumacue.json)")
    p.add_argument("--audio", help="song file; default: the video's own soundtrack")
    p.add_argument("--style", choices=STYLES, default="balanced", help="how busy the lights are (default balanced)")
    p.add_argument("--palette", choices=sorted(PALETTES), help="LumaCue colour palette (default: by style)")
    p.add_argument("--costume", choices=sorted(COSTUMES), default="split",
                   help="split = full suit with left/right arms and legs (default); full = LumaCue Full suit; "
                        "bench = 3-zone bench costume")
    p.add_argument("--dancers", type=int, default=10, help="dancers in the show (default 10)")
    p.add_argument("--track", default="all",
                   help="'all' puts the cues on every costume (default); a number 1..N puts them on one dancer")
    p.add_argument("--name", help="show name (default: from the video file name)")
    p.add_argument("--offset-ms", type=float, default=0.0,
                   help="song time minus video time, when --audio doesn't start with the video")
    p.add_argument("--no-base", action="store_true", help="only move cues, no music-driven base layer")
    p.add_argument("--mirror", action="store_true", help="the video is mirrored (selfie camera)")
    p.add_argument("--preview", help="also render a side-by-side preview video (.mp4)")
    p.add_argument("--moves", help="also write the detected moves and beats as JSON, for checking")
    p.add_argument("--save-pose", help="save the tracked pose (.npz) so later runs skip tracking")
    p.add_argument("--pose", help="use a saved pose (.npz) instead of tracking the video")
    p.add_argument("--model", help="MediaPipe pose model (.task); default models/pose_landmarker_lite.task")
    p.add_argument("--max-width", type=int, default=480, help="frame width used for tracking (default 480)")
    p.add_argument("--frame-step", type=int, default=2, help="analyse every n-th frame (default 2)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads for OpenCV (default 2)")
    p.add_argument("--demo", action="store_true", help="use a built-in synthetic dancer and song")
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    p.add_argument("--version", action="version", version=f"PoseCue {__version__}")
    return p


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> int:
    if not (1 <= args.dancers <= 50):
        raise FileCheckError("--dancers must be 1–50 (LumaCue's limit)")
    if not (160 <= args.max_width <= 1920) or not (1 <= args.frame_step <= 10) or not (1 <= args.threads <= 16):
        raise FileCheckError("--max-width 160–1920, --frame-step 1–10, --threads 1–16")
    if abs(args.offset_ms) > 10 * 60 * 1000:
        raise FileCheckError("--offset-ms must be within ±10 minutes")
    if args.track == "all":
        track = ALL_DANCERS
    else:
        try:
            track = int(args.track) - 1
        except ValueError:
            raise FileCheckError("--track must be 'all' or a dancer number") from None
        if not (0 <= track < args.dancers):
            raise FileCheckError(f"--track must be between 1 and --dancers ({args.dancers})")

    # ---- inputs ----
    video = audio = pose_in = None
    if args.demo:
        if args.video or args.pose:
            raise FileCheckError("--demo doesn't take a video or --pose")
    elif args.pose:
        pose_in = check_input(args.pose, POSE_EXT, 64 * 1024**2, "pose file")
        if args.video:
            video = check_input(args.video, VIDEO_EXT, MAX_VIDEO_BYTES, "video")
    elif args.video:
        video = check_input(args.video, VIDEO_EXT, MAX_VIDEO_BYTES, "video")
    else:
        raise FileCheckError("give a dance video (or --pose file.npz, or --demo)")
    if args.audio:
        audio = check_input(args.audio, AUDIO_EXT, MAX_AUDIO_BYTES, "audio")
    elif not args.demo and video is None:
        raise FileCheckError("with --pose and no video, pass the song with --audio")
    model = Path(args.model) if args.model else None
    if model is not None:
        model = check_input(args.model, {".task"}, 64 * 1024**2, "pose model")

    inputs = [p for p in (video, audio, pose_in, model) if p is not None]
    out = check_output(args.output, JSON_EXT, inputs, args.force, "output")
    preview = check_output(args.preview, MP4_EXT, inputs + [out], args.force, "preview") if args.preview else None
    moves_out = check_output(args.moves, JSON_EXT, inputs + [out], args.force, "moves file") if args.moves else None
    save_pose = check_output(args.save_pose, POSE_EXT, inputs, args.force, "pose file") if args.save_pose else None
    if len({p for p in (out, preview, moves_out, save_pose) if p is not None}) < sum(
            p is not None for p in (out, preview, moves_out, save_pose)):
        raise FileCheckError("two outputs have the same file name")

    tmp = tempfile.TemporaryDirectory(prefix="posecue-demo-") if args.demo else None
    try:
        t0 = time.time()
        # ---- pose ----
        if args.demo:
            from scipy.io import wavfile

            from posecue.synth import demo_script, synth_music, synth_pose

            dur, steps, music = demo_script()
            seq = synth_pose(dur, steps)
            samples = synth_music(music)
            audio = Path(tmp.name) / "demo.wav"
            wavfile.write(audio, 22050, samples)
            name = args.name or "PoseCue demo"
            audio_name = "PoseCue demo beat"
        else:
            if pose_in is not None:
                seq = PoseSequence.load(pose_in)
                _log(f"pose: {len(seq)} frames from {pose_in.name}")
            else:
                from posecue.pose import DEFAULT_MODEL, track_video

                _log(f"tracking {video.name} at {args.max_width}px, every {args.frame_step} frame(s)...")
                seq = track_video(video, model or DEFAULT_MODEL, max_width=args.max_width,
                                  frame_step=args.frame_step, threads=args.threads,
                                  progress=lambda i, n: _log(f"  frame {i}/{n or '?'}"))
                found = int(np.isfinite(seq.xy[:, 0, 0]).sum())
                _log(f"pose: dancer found in {found}/{len(seq)} frames ({time.time() - t0:.0f} s)")
            if args.mirror:
                seq = seq.mirrored()
            if save_pose is not None:
                seq.save(save_pose)
            _log("analysing music...")
            samples = load_audio(audio or video)
            name = args.name or (video or pose_in).stem
            audio_name = (audio or video).name
        if save_pose is not None and args.demo:
            seq.save(save_pose)

        analysis_audio = analyze_audio(samples)
        _log(f"music: {analysis_audio.bpm:.1f} BPM, {len(analysis_audio.sections)} sections, "
             f"{analysis_audio.lengthMs / 1000:.1f} s")
        analysis = analyze_moves(seq)
        _log(f"moves: {describe_moves(analysis.moves) or 'none found'}")

        # ---- cues ----
        costume = get_costume(args.costume)
        opts = CueOptions(style=args.style, palette=args.palette, track=track, base=not args.no_base,
                          offset_ms=args.offset_ms)
        length_ms = min(analysis_audio.lengthMs, 60 * 60 * 1000)
        clips = generate_clips(analysis_audio, analysis, costume, length_ms, opts)
        show = build_show(name=safe_display_name(name), dancers=args.dancers, length_ms=length_ms,
                          audio_name=safe_display_name(audio_name), clips=clips,
                          sections=show_sections(analysis_audio, analysis, length_ms, opts),
                          costume=costume.to_json())
        problems = validate_show(show)
        if problems:
            raise RuntimeError("internal error, show file failed validation: " + "; ".join(problems[:5]))
        write_text_atomic(out, show_json(show))
        _log(f"wrote {out.name}: {len(clips)} clips on {costume.label}")

        if moves_out is not None:
            write_text_atomic(moves_out, json.dumps({
                "audio": analysis_audio.to_dict(),
                "moves": [m.to_dict() for m in analysis.moves],
                "torso": analysis.torso,
            }, indent=1, default=float))

        if preview is not None:
            from posecue.preview import render_preview

            _log("rendering preview...")
            render_preview(preview, show, costume, seq, analysis.moves, video=video, audio=audio or video,
                           offset_ms=args.offset_ms, dancer=max(0, track),
                           progress=lambda k, n: _log(f"  preview {k}/{n}"))
            _log(f"wrote {preview.name}")
        _log(f"done in {time.time() - t0:.0f} s")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileCheckError, FileNotFoundError, ValueError, RuntimeError) as e:
        _log(f"posecue: error: {e}")
        return 2
