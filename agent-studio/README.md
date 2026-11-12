# Agent Studio

Agent Studio provides reusable skills and runtime assets for AI tasks on Amlogic devices. Skills use `SKILL.md` instructions and executable scripts, and work with DSH or other Agents that support skill discovery and shell execution.

## Included examples

This release includes the following example skills. You can add your own skills under `skills/`.

| Skill | Input | Output |
| --- | --- | --- |
| `qrcode-reader` | Image or directory | Decoded text, positions and annotated images |
| `yolox-object-detection` | Image or directory | Object counts and annotated images |
| `audio-transcription` | WAV or directory | Transcript files |

## Requirements

- Python 3.10+; Node.js 24 and npm when using the DSH example.
- QR/YOLOX: Linux ARM64 / A311Y3, a working ADLA driver and device-provided `libnnsdk.so`. Executables and models are included under each Skill's `assets/`.
- DSH: an OpenAI-compatible chat model with tool calling. Transcription also needs an ASR endpoint; use mono 16 kHz PCM16 WAV for the tested setup.

The Skill scripts use the Python standard library. They do not install the model/ASR servers or device drivers.

## Quick start with DSH

Run these commands on the device in Bash, starting from the cloned repository:

```bash
cd agent-studio
cp examples/dsh/env.example .env
# Edit .env: set your chat/ASR URLs, model IDs and API keys.
```

After saving `.env`:

```bash
. ./.env
STUDIO="$PWD"
python3 scripts/install-commands.py --bin-dir "$STUDIO/.bin"
export PATH="$STUDIO/.bin:$PATH"
export DSH_SKILLS_DIR="$STUDIO/skills"
export DSH_HOME="$STUDIO/.dsh"
DSH_PROFILE="$DSH_HOME/profiles/agent-studio"
mkdir -p "$DSH_PROFILE"
cp examples/dsh/{package.json,package-lock.json,cordis.patch.yml} "$DSH_PROFILE/"
npm ci --prefix "$DSH_PROFILE"

mkdir -p "$STUDIO/work"
cd "$STUDIO/work"
"$DSH_PROFILE/node_modules/.bin/dsh" --profile agent-studio \
  'Decode the QR code in /data/qr.png and return the annotated image.'
```

Replace `/data/qr.png` with an existing file on the device. Keep this checkout in its final location because command aliases point to its scripts. Each CLI invocation starts a new conversation in the same working directory. In a new shell, repeat the environment exports above; npm installation is needed only once.

Other test inputs:

```text
Detect and count the objects in /data/photo.jpg and return the annotated image.
Transcribe /data/audio.wav and return the transcript file.
```

Check for actual Bash execution and generated files. A reply saying that the Skill was loaded is not a completed task. [DSH configuration and device permissions](examples/dsh/README.md).

## Use with another Agent

Register `skills/`, or copy a complete Skill folder into your Agent's discovery directory. Preserve `scripts/` and `assets/` together. `scripts/install-commands.py` registers optional short commands from `commands.json`; scripts can also be invoked directly with Python.

To check a runtime independently of the model, run `aml-qrcode /data/qr.png`, `aml-yolox /data/photo.jpg`, or `aml-asr /data/audio.wav` after the setup above. Results are JSON with actual output paths. Use `--help` for options.

## Add a skill

Create `skills/<name>/SKILL.md` with YAML front matter containing `name` and `description`. Describe how to execute the task, and include its scripts and required assets in the same Skill folder. Start a new DSH process to discover the added Skill. A short command is optional: use the script entry point directly, or register it in `commands.json` and rerun the command installer.

Preserve each Skill's supporting files and third-party notices when copying or redistributing it.

