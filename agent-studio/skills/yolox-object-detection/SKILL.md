---
name: yolox-object-detection
description: Detect and count objects in images or directories using local YOLOX inference, and generate annotated images.
---

Run with a 180-second timeout, resolving the script relative to this skill:

`aml-yolox /absolute/path/input.png`

Input may be an image or directory. Add `--recursive` only when requested. Report actual class counts and link annotated images from the JSON result. Preserve per-file errors.

Without the command alias, run `python3 scripts/detect.py` relative to this skill directory.
