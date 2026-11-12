#!/usr/bin/env python3
"""Transcribe WAV files through the portable meeting runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    skill = Path(__file__).resolve().parents[1]
    bundled = skill / 'assets/meeting/runtime/meeting.py'
    entry = os.environ.get("MEETING_RUNTIME") or str(bundled if bundled.is_file() else skill.parent.parent / 'packages/meeting/runtime/meeting.py')
    if not entry or not Path(entry).is_absolute() or not Path(entry).is_file():
        print(json.dumps({"status":"error","data":{"error":"Install the meeting runtime assets or configure MEETING_RUNTIME"},"artifacts":[]}))
        return 1
    args = sys.argv[1:]
    if not any(a.startswith(("--speakers", "--diarization")) for a in args):
        args += ["--diarization", "off"]
    return subprocess.run([sys.executable, entry, *args, "--task", "transcribe"]).returncode

if __name__ == "__main__":
    raise SystemExit(main())
