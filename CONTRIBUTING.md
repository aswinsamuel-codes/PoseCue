# Contributing to PoseCue

Thanks for helping! PoseCue is small on purpose: easy to read, runs on a
CPU laptop, and its output always loads cleanly in LumaCue Show Studio.

## Set up

```bash
python -m pip install -r requirements.txt
python -m pytest
```

The tests need no video and no pose model. To run on real footage, download
the model with `python scripts/download_models.py`.

## Ground rules

- **Move detection stays pure.** Functions in `posecue/moves.py` take arrays
  and return `Move`s; no file or model access. Add a synthetic test in
  `tests/test_moves.py` (extend `posecue/synth.py` if you need a new motion).
- **Thresholds are in torso lengths and seconds**, never pixels, so they work
  for any camera distance. New ones go in `Tuning`.
- **The show file must match LumaCue.** If LumaCue's `lib/show/model.ts`
  changes, update `posecue/export.py` and keep `tests/test_export.py` green.
  `validate_show()` must stay at least as strict as Show Studio's
  `parseShowProject()`.
- **Keep it CPU-light.** No GPU-only dependencies; stream frames, don't hold a
  whole video in memory.
- **New dependencies** need a good reason: say why in the pull request.
- **Never commit** dance videos, songs, downloaded models, or generated shows
  (see `.gitignore`). Dancers' videos are personal data.

## Pull requests

1. Fork, branch from `main`, keep changes focused.
2. `python -m pytest` passes.
3. Describe what changes for a dancer or light engineer, with a preview frame
   or a short clip when the lights change.

## Reporting issues

Include your OS, Python version, the command you ran, and the output of
`--moves moves.json` if detection is off. Please don't attach videos of
people who haven't agreed to it being shared.
