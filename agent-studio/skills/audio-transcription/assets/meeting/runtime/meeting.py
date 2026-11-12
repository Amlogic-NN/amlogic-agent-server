#!/usr/bin/env python3
"""Meeting workflow: PCM16 WAV/text -> transcript -> checked Markdown minutes.
Standard library only. Config and credentials come from the environment.
"""
from __future__ import annotations
import argparse
import importlib.util
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_diar_spec = importlib.util.spec_from_file_location("meeting_diarization", Path(__file__).resolve().with_name("diarization.py"))
diarization = importlib.util.module_from_spec(_diar_spec)
_diar_spec.loader.exec_module(diarization)

def prompt_directory():
    if os.environ.get("MEETING_PROMPTS_DIR"):
        return Path(os.environ["MEETING_PROMPTS_DIR"])
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "skills/meeting-summary/prompts"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent / "prompts"

PROMPT_DIR = prompt_directory()

def conservative_fields(markdown, transcript):
    """Discard attributed names/deadlines absent from source; not semantic verification."""
    normalize = lambda text: " ".join(re.findall(r"\w+", text.casefold()))
    source = normalize(transcript)
    pattern = re.compile(r"^(\s*(?:[-*]\s+)?(?:\*\*)?(?:Owner|Deadline|负责人|责任人|截止日期|截止时间)(?:\*\*)?\s*[:：](?:\*\*)?\s*)(.*)$", re.I)
    lines = []
    for line in markdown.splitlines():
        match = pattern.match(line)
        if match:
            value = normalize(match[2])
            if value and value not in ("not specified", "未指定", "未提及", "未知") and value not in source:
                line = match[1] + ("Not specified" if re.search(r"Owner|Deadline", match[1], re.I) else "未指定")
        lines.append(line)
    return "\n".join(lines)

class WorkflowError(Exception):
    pass

def load_prompts():
    prompts = {}
    for name in ("summarize", "extract", "merge"):
        path = PROMPT_DIR / (name + ".md")
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise WorkflowError("Cannot read skill prompt: " + str(path)) from exc
        if not value or len(value) > 32000:
            raise WorkflowError("Skill prompt must contain 1–32000 characters: " + str(path))
        if name != "summarize" and value.count("{{text}}") != 1:
            raise WorkflowError("Skill prompt must contain exactly one {{text}} placeholder: " + str(path))
        prompts[name] = value
    return prompts

def summary_chunks(text, limit):
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        prefix_match = re.match(r"(\[[^\]\n]+\] Speaker [A-Z0-9]+: )", line)
        prefix = prefix_match[1] if prefix_match and len(prefix_match[1]) < limit else ""
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = prefix + line[limit:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = ""
        current += line
    if current:
        chunks.append(current)
    return chunks

def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()

def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def post_json_or_audio(url, body, content_type, key, timeout):
    headers = {"Content-Type": content_type}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        with urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except HTTPError as exc:
        raise WorkflowError("Service returned HTTP " + str(exc.code)) from exc
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise WorkflowError("Service request failed: " + str(exc.reason if isinstance(exc, URLError) else exc)) from exc
    if not isinstance(payload, dict) or payload.get("error"):
        raise WorkflowError("Service returned an invalid or error response")
    return payload

def asr_request(pcm_wav, cfg):
    boundary = "aml" + uuid.uuid4().hex
    fields = {"model": cfg["asr_model"], "language": cfg["language"], "response_format": "json"}
    body = b""
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="segment.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + pcm_wav + f'\r\n--{boundary}--\r\n'.encode())
    base = cfg["asr_base"].rstrip("/")
    endpoint = base if base.endswith("/audio/transcriptions") else base + "/audio/transcriptions"
    result = post_json_or_audio(endpoint, body, "multipart/form-data; boundary=" + boundary, cfg["asr_key"], cfg["timeout"])
    if not isinstance(result.get("text"), str):
        raise WorkflowError("ASR response has no text field")
    return result["text"].strip()

def configuration(args):
    base = args.llm_base or os.environ.get("MEETING_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    model = args.llm_model or os.environ.get("MEETING_MODEL") or os.environ.get("OPENAI_MODEL", "")
    if args.task == "meeting" and (not base or not model):
        raise WorkflowError("Configure OPENAI_BASE_URL and OPENAI_MODEL for summarization")
    return {"llm_base": base.rstrip("/"), "llm_model": model,
            "llm_key": os.environ.get("MEETING_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
            "asr_base": (args.asr_base or os.environ.get("ASR_BASE_URL", "")).rstrip("/"),
            "asr_model": args.asr_model or os.environ.get("ASR_MODEL", "sensevoice-small"),
            "asr_key": args.asr_key or os.environ.get("ASR_API_KEY", ""),
            "language": args.language, "timeout": args.timeout, "max_tokens": args.max_tokens}

class Workflow:
    def __init__(self, args):
        self.args = args
        self.source = args.input.expanduser().resolve(strict=True)
        self.raw = self.source.read_bytes()
        if len(self.raw) > 20 * 1024 * 1024:
            raise WorkflowError("Input exceeds 20 MiB")
        self.cfg = configuration(args)
        self.prompts = load_prompts() if args.task == "meeting" else {}
        self.stage = "input"
        self.speaker_config = None
        self.speaker_mode = "not_applicable" if self.source.suffix.lower() != ".wav" else "off"
        self.warnings = []
        resume_state = None
        if args.resume:
            resume_path = args.resume.expanduser().resolve(strict=True)
            if not resume_path.parent.is_relative_to(Path.cwd().resolve()):
                raise WorkflowError("Resume checkpoint must be inside the current workspace")
            resume_state = json.loads(resume_path.read_text(encoding="utf-8"))
            if not isinstance(resume_state, dict):
                raise WorkflowError("Invalid checkpoint object")
        schema = "aml.meeting.run.v1"
        identity = {"input_sha256": digest(self.raw), "kind": self.source.suffix.lower()}
        if args.task != "meeting":
            identity["task"] = args.task
        if self.source.suffix.lower() == ".wav":
            if args.task != "diarize" and not self.cfg["asr_base"]:
                raise WorkflowError("Configure ASR_BASE_URL before transcribing audio")
            legacy_resume = resume_state and resume_state.get("schema") == "aml.meeting.run.v1" and args.diarization == "auto"
            if args.speakers:
                if args.diarization == "off":
                    raise WorkflowError("--speakers conflicts with --diarization off")
                supplied = args.speakers.expanduser().resolve(strict=True).read_bytes()
                if len(supplied) > 1024 * 1024:
                    raise WorkflowError("Speaker file exceeds 1 MiB")
                payload = json.loads(supplied)
                if not isinstance(payload, dict) or payload.get("schema") != "aml.meeting.speakers.v1":
                    raise WorkflowError("Invalid speaker file schema")
                self.speaker_config = {"provided": payload}
                self.speaker_mode = "diarized"
                schema = "aml.meeting.run.v2"
                identity.update(speaker_file_sha256=digest(supplied), speaker_policy=diarization.POLICY)
            elif args.diarization != "off" and not legacy_resume:
                self.speaker_config, missing = diarization.resources()
                if missing and args.diarization == "required":
                    raise WorkflowError("Speaker resources unavailable: " + ", ".join(missing))
                if self.speaker_config:
                    self.speaker_mode = "diarized"
                    schema = "aml.meeting.run.v2"
                    identity.update(diarization=self.speaker_config["fingerprints"], speaker_policy=diarization.POLICY)
                else:
                    self.speaker_mode = "unavailable"
                    self.warnings.append("Speaker resources unavailable; transcript has no speaker labels. Use --diarization required when speaker separation is required.")
            elif legacy_resume:
                self.speaker_mode = "legacy"
                self.warnings.append("Resuming a legacy checkpoint without speaker labels. Start a new run to enable speaker separation.")
            if args.task != "diarize":
                identity.update(asr_base=self.cfg["asr_base"], asr_model=self.cfg["asr_model"], language=args.language, chunk_seconds=args.chunk_seconds)
            else:
                identity["chunk_seconds"] = args.chunk_seconds
        if args.resume:
            self.state_path = args.resume.expanduser().resolve(strict=True)
            self.out = self.state_path.parent
            if not self.out.is_relative_to(Path.cwd().resolve()):
                raise WorkflowError("Resume checkpoint must be inside the current workspace")
            self.state = resume_state
            if self.state.get("schema") != schema or self.state.get("identity") != identity:
                raise WorkflowError("Checkpoint does not match the input or ASR configuration")
        else:
            self.out = (args.output_dir or Path.cwd() / "meeting_results" / uuid.uuid4().hex).expanduser().resolve()
            if self.out == Path.cwd().resolve() or not self.out.is_relative_to(Path.cwd().resolve()):
                raise WorkflowError("Output directory must be a subdirectory of the current workspace")
            self.out.mkdir(parents=True, exist_ok=False)
            self.state_path = self.out / "run.json"
            self.state = {"schema": schema, "identity": identity, "input_name": self.source.name, "segments": [], "summaries": {}, "created_at": time.time()}
        if not isinstance(self.state.get("segments"), list) or not isinstance(self.state.get("summaries"), dict):
            raise WorkflowError("Invalid checkpoint structure")
        if any(not isinstance(segment, dict) or not isinstance(segment.get("text"), str) for segment in self.state["segments"]):
            raise WorkflowError("Invalid segment checkpoint")
        self.state["speaker_mode"] = self.speaker_mode
        self.state["warnings"] = self.warnings
        self.state["prompt_sha256"] = {name + ".md": digest(value) for name, value in self.prompts.items()}
        self.state["status"] = "running"
        self.calls = 0
        self.save()

    def save(self):
        self.state["stage"] = self.stage
        self.state["updated_at"] = time.time()
        atomic_json(self.state_path, self.state)

    def transcript(self):
        self.stage = "transcription"
        self.save()
        if self.source.suffix.lower() in (".txt", ".md"):
            text = self.raw.decode("utf-8-sig").strip()
        elif self.source.suffix.lower() == ".wav":
            with wave.open(io.BytesIO(self.raw), "rb") as source:
                if source.getsampwidth() != 2 or source.getnchannels() not in (1, 2) or not 8000 <= source.getframerate() <= 48000:
                    raise WorkflowError("Expected PCM16 WAV, 8–48 kHz, mono or stereo")
                rate, frames = source.getframerate(), source.getnframes()
                if frames / rate > 600:
                    raise WorkflowError("Audio exceeds 10 minutes")
                if self.speaker_config:
                    return self.speaker_transcript()
                params = source.getparams()
                step = max(1, int(self.args.chunk_seconds * rate))
                expected = (frames + step - 1) // step
                if len(self.state["segments"]) > expected:
                    raise WorkflowError("Invalid segment checkpoint")
                for index, start in enumerate(range(0, frames, step)):
                    if index < len(self.state["segments"]):
                        continue
                    source.setpos(start)
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as segment:
                        segment.setparams(params)
                        segment.writeframes(source.readframes(min(step, frames - start)))
                    value = asr_request(buffer.getvalue(), self.cfg)
                    self.state["segments"].append({"index": index, "start_seconds": start / rate, "end_seconds": min(frames, start + step) / rate, "text": value})
                    self.save()
            text = "\n".join(p["text"] for p in self.state["segments"] if p["text"]).strip()
        else:
            raise WorkflowError("Input must be PCM16 .wav or a UTF-8 .txt/.md transcript")
        if not text:
            raise WorkflowError("No speech or transcript text was returned")
        (self.out / "transcript.txt").write_text(text + "\n", encoding="utf-8")
        return text

    def speaker_transcript(self):
        self.stage = "diarization"
        self.save()
        snapshot = self.out / "speaker-input.wav"
        snapshot.write_bytes(self.raw)
        audio = diarization.normalize_audio(snapshot, self.out)
        with wave.open(str(audio), "rb") as source:
            rate, frames = source.getframerate(), source.getnframes()
            audio_sha256 = digest(source.readframes(frames))
            source.rewind()
            if "diarization" not in self.state:
                if "provided" in self.speaker_config:
                    supplied = self.speaker_config["provided"]
                    if supplied.get("audio_sha256") != audio_sha256 or supplied.get("frame_rate") != rate or supplied.get("audio_frames") != frames or supplied.get("policy") != diarization.POLICY:
                        raise WorkflowError("Speaker file does not match normalized audio or segmentation policy")
                    raw = supplied.get("raw_segments")
                else:
                    raw = diarization.run_native(audio, self.speaker_config, self.args.diar_timeout)
                result = diarization.frame_plan(raw, rate, frames, self.args.chunk_seconds)
                result["audio_sha256"] = audio_sha256
                self.state["diarization"] = result
                self.save()
            else:
                cached = self.state["diarization"]
                if not isinstance(cached, dict):
                    raise WorkflowError("Invalid diarization checkpoint")
                result = diarization.frame_plan(cached.get("raw_segments"), rate, frames, self.args.chunk_seconds)
                result["audio_sha256"] = audio_sha256
                if cached != result:
                    raise WorkflowError("Diarization checkpoint no longer matches the audio frame plan")
            if result["overlap_adjustments"]:
                self.warnings.append("Speaker segment boundaries were reconciled to avoid duplicate audio. Labels and segment times are approximate, not word-level alignment or overlap separation.")
            atomic_json(self.out / "speakers.json", {"schema": "aml.meeting.speakers.v1", "speaker_scope": "current_recording", **{key: value for key, value in result.items() if key != "plan"}})
            if self.args.task == "diarize":
                return ""
            diarization.validate_prefix(self.state["segments"], result["plan"])
            self.stage = "speaker_transcription"
            self.save()
            for segment in result["plan"][len(self.state["segments"]):]:
                source.setpos(segment["start_frame"])
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setparams(source.getparams())
                    wav.writeframes(source.readframes(segment["end_frame"] - segment["start_frame"]))
                text = asr_request(buffer.getvalue(), self.cfg)
                self.state["segments"].append({**segment, "text": text})
                self.save()
        plain = "\n".join(s["text"] for s in self.state["segments"] if s["text"]).strip()
        if not plain:
            raise WorkflowError("No speech or transcript text was returned")
        tagged = diarization.tagged_text(self.state["segments"])
        (self.out / "transcript.txt").write_text(plain + "\n", encoding="utf-8")
        (self.out / "transcript-speakers.txt").write_text(tagged + "\n", encoding="utf-8")
        atomic_json(self.out / "transcript.json", {"schema": "aml.meeting.transcript.v2", "speaker_scope": "current_recording", "speakers": result["speakers"], "segments": self.state["segments"]})
        return tagged

    def summarize_one(self, text, tokens, mode="merge"):
        user_prompt = self.prompts[mode].replace("{{text}}", text)
        signature = digest(json.dumps([self.cfg["llm_base"], self.cfg["llm_model"], self.prompts["summarize"], user_prompt, tokens], ensure_ascii=False))
        cached = self.state["summaries"].get(signature)
        if isinstance(cached, str) and cached.strip():
            return cached
        self.calls += 1
        if self.calls > 32:
            raise WorkflowError("Summary call budget exceeded; use a shorter transcript")
        body = json.dumps({"model": self.cfg["llm_model"], "temperature": 0, "max_tokens": tokens,
                           "messages": [{"role": "system", "content": self.prompts["summarize"]}, {"role": "user", "content": user_prompt}]}).encode()
        payload = post_json_or_audio(self.cfg["llm_base"] + "/chat/completions", body, "application/json", self.cfg["llm_key"], self.cfg["timeout"])
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise WorkflowError("LLM response has no answer") from exc
        if choice.get("finish_reason") != "stop":
            if isinstance(content, str):
                (self.out / "summary.partial.md").write_text(content, encoding="utf-8")
            raise WorkflowError("Summary incomplete: finish_reason=" + str(choice.get("finish_reason")) + "; increase --max-tokens or shorten input and resume")
        if not isinstance(content, str) or not content.strip():
            raise WorkflowError("LLM returned an empty summary")
        self.state["summaries"][signature] = content.strip()
        self.save()
        return content.strip()

    def run(self):
        text = self.transcript()
        if self.args.task != "meeting":
            self.stage = "complete"
            self.state["status"] = "ok"
            self.state.pop("error", None)
            self.save()
            speaker_count = len(self.state.get("diarization", {}).get("speakers", []))
            names = ["speakers.json", "run.json"] if self.args.task == "diarize" else ["transcript.txt", "run.json"] + (["speakers.json", "transcript.json", "transcript-speakers.txt"] if self.speaker_config else [])
            result = {"status": "ok", "data": {"stage": self.stage, "task": self.args.task, "speaker_count": speaker_count, "speaker_mode": self.speaker_mode, "warnings": self.warnings, "run_file": str(self.state_path)}, "artifacts": [{"path": str(self.out / name)} for name in names]}
            if self.args.task == "transcribe":
                result["files"] = [{"file": self.source.name, "status": "ok", "text": text,
                                    "transcript": str(self.out / 'transcript.txt')}]
                result["summary"] = {"total": 1, "succeeded": 1, "failed": 0}
            else:
                result["data"]["segments"] = [{"start_seconds": s["start_frame"] / self.state["diarization"]["frame_rate"], "end_seconds": s["end_frame"] / self.state["diarization"]["frame_rate"], "speaker_id": s["speaker_id"], "speaker_label": s["speaker_label"]} for s in self.state["diarization"]["effective_segments"]]
                result["data"]["frame_rate"] = self.state["diarization"]["frame_rate"]
                result["reports"] = [{"title": "Speaker intervals", "markdown": f"Detected {speaker_count} anonymous speaker groups. Segment times and labels are approximate; download speakers.json for the intervals."}]
            return result
        self.stage = "summary"
        self.save()
        notes = text
        for level in range(5):
            if len(notes) <= self.args.chunk_chars:
                break
            parts = summary_chunks(notes, self.args.chunk_chars)
            notes = "\n\n".join(self.summarize_one(part, min(1024, self.cfg["max_tokens"]), mode="extract") for part in parts)
        else:
            raise WorkflowError("Transcript cannot be reduced within the configured context budget")
        summary = conservative_fields(self.summarize_one(notes, self.cfg["max_tokens"]), text)
        (self.out / "summary.md").write_text(summary + "\n", encoding="utf-8")
        partial = self.out / "summary.partial.md"
        if partial.exists():
            partial.unlink()
        self.stage = "complete"
        self.state["status"] = "ok"
        self.state.pop("error", None)
        self.state["summary_model"] = self.cfg["llm_model"]
        self.state["max_tokens"] = self.cfg["max_tokens"]
        self.save()
        return {"status": "ok", "data": {"stage": self.stage, "transcript_characters": len((self.out / "transcript.txt").read_text(encoding="utf-8").rstrip("\n")), "speaker_mode": self.speaker_mode, "speaker_count": len(self.state.get("diarization", {}).get("speakers", [])), "warnings": self.warnings, "summary_characters": len(summary), "segments": len(self.state["segments"]), "run_file": str(self.state_path)},
                "reports": [{"title": "Meeting minutes", "markdown": summary}],
                "artifacts": [{"path": str(self.out / name)} for name in (("transcript.txt", "summary.md", "run.json") + (("speakers.json", "transcript.json", "transcript-speakers.txt") if self.speaker_config else ()))]}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--task", choices=("meeting", "transcribe", "diarize"), default="meeting")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--speakers", type=Path, help="Reuse matching speakers.json without rerunning CAM++")
    parser.add_argument("--base-url", dest="asr_base")
    parser.add_argument("--model", dest="asr_model")
    parser.add_argument("--api-key", dest="asr_key")
    parser.add_argument("--llm-base-url", dest="llm_base")
    parser.add_argument("--llm-model", dest="llm_model")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--diarization", choices=("off", "auto", "required"), default=os.environ.get("MEETING_DIARIZATION", "auto"))
    parser.add_argument("--diar-timeout", type=float, default=60)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--chunk-seconds", type=float, default=20)
    parser.add_argument("--chunk-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MEETING_MAX_TOKENS", "4096")))
    parser.add_argument("--timeout", "--max-time", type=float, default=180)
    args = parser.parse_args(argv)
    if args.task == "diarize":
        args.diarization = "required"
    workflow = None
    try:
        if args.diarization not in ("off", "auto", "required") or not 1 <= args.diar_timeout <= 180:
            raise WorkflowError("Invalid speaker mode or timeout")
        if not 1 <= args.max_tokens <= 8192 or not 1 <= args.chunk_seconds <= 20 or not 1000 <= args.chunk_chars <= 12000 or not 1 <= args.timeout <= 600:
            raise WorkflowError("Invalid token, segmentation or timeout setting")
        if args.task in ("diarize", "transcribe") and not args.input.is_dir() and args.input.suffix.lower() != ".wav":
            raise WorkflowError("This task accepts PCM16 WAV input")
        if args.input.is_dir():
            if args.task != "transcribe" or args.resume or args.speakers or args.diarization != "off":
                raise WorkflowError("Directory input is supported only for plain transcription without --resume or --speakers")
            sources = sorted(p for p in (args.input.rglob("*") if args.recursive else args.input.glob("*")) if p.is_file() and p.suffix.lower() == ".wav")
            if not sources or len(sources) > 1000:
                raise WorkflowError("Expected 1–1000 WAV files in the input directory")
            results, artifacts = [], []
            output_root = (args.output_dir or Path.cwd() / ('transcription-' + uuid.uuid4().hex[:12])).resolve()
            for source in sources:
                from copy import copy
                item_args = copy(args)
                item_args.input = source
                relative = source.relative_to(args.input)
                item_args.output_dir = output_root / relative.parent / (relative.stem + '-' + digest(relative.as_posix())[:8])
                try:
                    value = Workflow(item_args).run()
                    for item in value.get("files", []):
                        item["file"] = str(source.relative_to(args.input))
                        results.append(item)
                    artifacts.extend(value["artifacts"])
                except (WorkflowError, OSError, ValueError, wave.Error, EOFError) as exc:
                    results.append({"file": str(source.relative_to(args.input)), "status": "error", "error": str(exc)})
            succeeded = sum(item['status'] == 'ok' for item in results)
            failed = succeeded != len(results)
            print(json.dumps({"status": "partial" if failed and succeeded else "error" if failed else "ok", "files": results,
                              "summary": {"total": len(results), "succeeded": succeeded, "failed": len(results) - succeeded},
                              "artifacts": artifacts}, ensure_ascii=False))
            return 1 if failed else 0
        workflow = Workflow(args)
        result = workflow.run()
        code = 0
    except (WorkflowError, OSError, ValueError, wave.Error, EOFError) as exc:
        result = {"status": "error", "data": {"stage": workflow.stage if workflow else "input", "error": str(exc)}, "artifacts": []}
        if workflow:
            workflow.state.update(status="error", error=str(exc))
            workflow.save()
            result["data"]["run_file"] = str(workflow.state_path)
        code = 1
    print(json.dumps(result, ensure_ascii=False))
    return code

if __name__ == "__main__":
    raise SystemExit(main())
