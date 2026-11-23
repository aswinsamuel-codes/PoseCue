import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posecue.audio import analyze_audio  # noqa: E402
from posecue.synth import demo_script, synth_music, synth_pose  # noqa: E402


@pytest.fixture(scope="session")
def demo():
    dur, steps, music = demo_script()
    return dur, steps, music


@pytest.fixture(scope="session")
def demo_pose(demo):
    dur, steps, _ = demo
    return synth_pose(dur, steps)


@pytest.fixture(scope="session")
def demo_audio(demo):
    return analyze_audio(synth_music(demo[2]))
