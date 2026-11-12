# DSH configuration

Uses stock DSH 0.1.5-rc.3 with its minimal runtime, native Skill loader, Bash and an OpenAI-compatible model adapter. Requires Node.js 24 and npm on the device.

Follow the [collection README](../../README.md) for setup and the first task. This directory contains the pinned npm manifest/lockfile, the minimal-runtime patch and an editable environment example.

| Setting | Value |
| --- | --- |
| `DSH_SKILLS_DIR` | Absolute path to this collection's `skills/` |
| `LLM_BASE_URL` / `LLM_MODEL` | Chat endpoint including `/v1`, and its actual model ID |
| `LLM_API_KEY` | Provider key; a nonempty placeholder is accepted by unauthenticated local servers |
| `ASR_BASE_URL` / `ASR_MODEL` | Transcription endpoint including `/v1`, and its ASR model ID |
| `ASR_API_KEY` | Set if the ASR server requires authentication |
| `LLM_CONTEXT_WINDOW` / `LLM_MAX_TOKENS` | Match your model; the example uses 8192 / 768 |

`npm ci` must run on the target architecture with optional native dependencies enabled. The lockfile is retained for reproducible installation. The profile enables Skill discovery/loading and Bash.

For native DSH Web, extend the minimal preset with `dsh-skill-filesystem` and `dsh-tool-skill`, and point `customSkillDirs` at your skill root. The NPU tests used **Full access** because the Workspace Write sandbox hides `/dev/adla0`; this profile's CLI also runs with the user's full OS permissions.

If a task fails, run its short command directly from the same shell. A missing `libnnsdk.so` or inaccessible `/dev/adla0` is a device runtime/permission issue; HTTP 401/404 means the service credentials, URL or model ID need checking. If the model only loads a Skill, verify that the selected chat model supports tool calling and start a new conversation with a concrete input file.
