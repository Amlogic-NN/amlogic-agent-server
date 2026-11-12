---
name: qrcode-reader
description: Detect and decode QR codes in images or directories, returning text, positions, and annotated images.
---

Run with a 180-second timeout, resolving the script relative to this skill:

`aml-qrcode /absolute/path/input.png`

Input may be an image or directory. Add `--recursive` only when requested. Return actual decoded text and output paths from the JSON result. Empty decoded text means decoding failed; report per-file errors.

Without the command alias, run `python3 scripts/read_qrcodes.py` relative to this skill directory.
