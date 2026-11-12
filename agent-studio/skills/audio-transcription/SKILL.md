---
name: audio-transcription
description: Transcribe WAV files or directories into text files using an ASR service.
---

Run with a 240-second timeout, resolving the script relative to this skill:

`aml-asr /absolute/path/input.wav`

Execute the command with the Bash tool. Return the actual transcript files after it completes.
Add `--recursive` only when requested. Use `--speakers FILE` only when speaker intervals are supplied.
On failure, report the error; after fixing the cause, resume with `--resume /absolute/path/run.json`.

Without the command alias, run `python3 scripts/transcribe_dir.py` relative to this skill directory.
