#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml>=6.0",
# ]
# ///

"""Bootstrap tool for openai server — model listing, downloading, and configuration.

Bootstrap model files are hosted on ModelScope
(default repo ``Amlogic/amlnn-adla-models``) or HuggingFace (default repo
``Amlogic-NN/amlnn-adla-models``).  ``--baseurl`` accepts:

    modelscope:[owner/repo][@revision]  ->  https://www.modelscope.cn/models/<repo>/resolve/<rev=master>
    ms:[owner/repo][@revision]          ->  (same)
    hf:[org/repo][@revision]            ->  https://huggingface.co/<repo>/resolve/<rev=main>
    huggingface:[org/repo][@revision]   ->  (same)
    modelscope / ms / hf / huggingface  ->  that hub's default repo (no ':')
    owner/repo                          ->  treated as a ModelScope repo id
    https://… / file://… / /local/dir   ->  used verbatim (mirror or offline zoo)

and may also be provided via $AMLLLM_MODEL_ZOO.

The referenced repo root must contain a ``metadata.jsonl`` (see
``scripts/gen_metadata.py``).  Its addresses are **relative paths**
(``LLM/A311Y3/<file>``), so they are resolved against the base URL above — the
same metadata works on ModelScope, HuggingFace or a local mirror, and nothing
is ever fetched from a hub baked into the file.  Absolute URLs in older
metadata are still honored as they are.

Each metadata entry carries its own ``dir``, so a model is downloaded into
``models/<dir>/`` (its own ``model.json``, weights and side-cars together):

Examples (deps auto-installed by uv); as an installed module the same commands
run as ``uv run -m amlllm_openai_server.bootstrap …``:

    uv run scripts/bootstrap.py list
    uv run scripts/bootstrap.py --baseurl ms:Amlogic/amlnn-adla-models list
    uv run scripts/bootstrap.py download Qwen3-4B-Instruct-2507
    uv run scripts/bootstrap.py download Qwen3-VL-4B-Instruct
    uv run scripts/bootstrap.py --baseurl hf:org/zoo@main download whisper-large-v3-turbo

Config generation follows ``examples/``:

* LLM models only need a generated ``model.json`` with ``weights`` set
  (template of ``examples/llm/qwen3.json``); no remote config is required.
* VLM entries (``multimodal``) additionally get their ``mmproj`` projector
  wired into the generated ``model.json`` from the metadata ``files`` roles.
* ASR models keep the original download path: the remote ``model.json`` is
  fetched and patched with the weights/decoder/tokenizer actually downloaded
  (templates of ``examples/asr/*/model.json``), or synthesized from the
  metadata ``files`` roles when the entry carries no config URL.

Every written ``model.json`` gets a **unique lowercase ``id``** (``a-z``,
``0-9`` and ``-``), because the server keys its models by that id: a duplicate
would make one of them unreachable. It is the ``model=`` value clients send
(``Qwen3-4B-Instruct-2507`` -> ``qwen3-4b-instruct-2507``); when another
downloaded model already uses the id, ``-2``, ``-3`` … is appended.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

# Default model-zoo repositories — override with --baseurl or $AMLLLM_MODEL_ZOO.
# metadata.jsonl keeps *relative* addresses, so the base URL decides where the
# files are fetched from (ModelScope, HuggingFace or any plain-HTTP mirror).
DEFAULT_REPO_ID = "Amlogic/amlnn-adla-models"
DEFAULT_HF_REPO_ID = "Amlogic-NN/amlnn-adla-models"
DEFAULT_BASE_URL = os.environ.get("AMLLLM_MODEL_ZOO",
                                  f"modelscope:{DEFAULT_REPO_ID}")
DEFAULT_REPO_BY_SCHEME = {
    "modelscope": DEFAULT_REPO_ID,
    "ms": DEFAULT_REPO_ID,
    "hf": DEFAULT_HF_REPO_ID,
    "huggingface": DEFAULT_HF_REPO_ID,
}
DEFAULT_REVISION_BY_SCHEME = {"modelscope": "master", "ms": "master",
                              "hf": "main", "huggingface": "main"}

MS_ENDPOINT = os.environ.get("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

SERVER_YAML_TEMPLATE = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_key": None,
        "log_level": "DEBUG",
        "title": "AMLLLM OneAPI Proxy",
        "version": "0.2.0",
    },
    "models": {
        "root_dir": "../models",
        # one session per model: a second request while a model is running gets
        # 429 instead of waiting for it (set false to queue instead)
        "reject_when_busy": True,
        # filled by `download` with the directory of every downloaded model
        # (each model owns models/<dir>/model.json)
        "enabled": [],
    },
}

# Config templates (mirroring examples/llm/qwen3.json and examples/asr/*)
# used when a metadata entry carries no downloadable config URL.
ASR_MODEL_TYPES = ("asr", "whisper", "sensevoice")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_base_url(value: str | None) -> str:
    """Translate a --baseurl value (scheme shorthand, URL or local dir) to a base URL.

    ``modelscope`` / ``ms`` / ``hf`` / ``huggingface`` may be given bare (or with
    an empty repo) to use the default repository of that hub; a bare
    ``owner/repo`` means ModelScope, and an ``http(s)://`` / ``file://`` URL (or
    an existing local directory) is used verbatim.
    """
    base = (value or DEFAULT_BASE_URL).strip()
    head = base.split("://", 1)[0].lower()
    if "://" in base and head in ("http", "https", "file"):
        return base.rstrip("/")
    if not base:
        raise ValueError("empty --baseurl")
    # a local directory works as an offline zoo (addresses are relative)
    if base.startswith(("/", "./", "../", "~")) or os.path.isdir(base):
        return Path(base).expanduser().resolve().as_uri()

    scheme = None
    if ":" in base.split("/", 1)[0]:
        scheme, _, base = base.partition(":")
        base = base.lstrip("/")  # tolerate 'modelscope://owner/repo'
    head = base.split("@", 1)[0].strip("/").lower()
    if head in DEFAULT_REPO_BY_SCHEME:  # bare 'hf' / 'ms' / 'huggingface'
        scheme, base = head, ""
    scheme = (scheme or "modelscope").lower()
    revision = None
    if "@" in base:
        base, _, revision = base.rpartition("@")
    base = base.strip("/") or DEFAULT_REPO_BY_SCHEME.get(scheme, "")
    if not base:
        raise ValueError(f"empty repository in --baseurl '{value}'")
    if scheme in ("modelscope", "ms"):
        rev = revision or DEFAULT_REVISION_BY_SCHEME[scheme]
        return f"{MS_ENDPOINT.rstrip('/')}/models/{base}/resolve/{rev}"
    if scheme in ("hf", "huggingface"):
        rev = revision or DEFAULT_REVISION_BY_SCHEME[scheme]
        return f"{HF_ENDPOINT.rstrip('/')}/{base}/resolve/{rev}"
    raise ValueError(f"unknown model-zoo source '{scheme}' "
                     f"(use modelscope/ms/hf/huggingface, a local dir or an https URL)")


def is_absolute_address(value: str) -> bool:
    """True when a metadata address is already a full URL (scheme://…)."""
    return bool(urllib.parse.urlsplit(str(value or "")).scheme)


def resolve_entry_url(value: str, base_url: str) -> str:
    """Absolute download URL for a metadata address.

    ``metadata.jsonl`` stores repo-relative paths (``LLM/A311Y3/<file>``) so the
    same file works on ModelScope, HuggingFace or a local mirror; addresses that
    already are full URLs (older metadata) are used as they are.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    if is_absolute_address(raw):
        return raw
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", raw.lstrip("/"))


def resolve_entry_addresses(entry: dict, base_url: str) -> dict:
    """Copy of *entry* whose url/config/files[].url are absolute for *base_url*."""
    resolved = dict(entry)
    resolved["url"] = resolve_entry_url(entry.get("url", ""), base_url)
    if entry.get("config"):
        resolved["config"] = resolve_entry_url(entry["config"], base_url)
    files = entry.get("files")
    if isinstance(files, list):
        resolved["files"] = [
            {**item, "url": resolve_entry_url(item.get("url", ""), base_url)}
            if isinstance(item, dict) else item
            for item in files
        ]
    return resolved


def is_asr_entry(entry: dict) -> bool:
    return (entry.get("model_type") or "").strip().lower() in ASR_MODEL_TYPES


def _request(url: str) -> urllib.request.Request:
    """Create a Request with a common User-Agent (+ optional bearer token)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AMLLLM-Bootstrap/1.0)"}
    token = (os.environ.get("AMLLLM_ZOO_TOKEN")
             or os.environ.get("MODELSCOPE_API_KEY")
             or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def fetch_metadata(base_url: str) -> list[dict]:
    """Fetch metadata.jsonl from the remote and return parsed entries."""
    url = f"{base_url.rstrip('/')}/metadata.jsonl"
    print(f"Fetching metadata from {url}")
    try:
        with urllib.request.urlopen(_request(url), timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as e:
        print(f"Error fetching metadata: {e}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for line in data.strip().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def normalize_name(name: str) -> str:
    """Lower-case *name* and strip ``-`` / ``_`` characters."""
    return re.sub(r"[-_]", "", name).lower()


def filename_from_url(url: str) -> str:
    """Extract the file name from a download URL, stripping query / fragment."""
    clean = url.split("?")[0].split("#")[0]
    return clean.rstrip("/").rsplit("/", 1)[-1]


def model_subdir(entry: dict) -> str:
    """Directory under models/ holding one model (``dir`` from the metadata).

    ``gen_metadata.py`` gives every entry its own ``dir`` (the model name), so
    each model keeps its own ``model.json`` and side-cars; legacy entries
    without ``dir`` fall back to ``default``.
    """
    return str(entry.get("dir") or "default")


def sidecar_relpath(item: dict) -> str:
    """Destination path relative to the model directory."""
    if item.get("path"):
        return str(item["path"]).replace("\\", "/").lstrip("/")
    return filename_from_url(item["url"])


def sidecar_dest(item: dict) -> str:
    """Where a side-car is stored, honouring its ``group`` directory.

    A directory side-car (the Whisper ``data_bin/`` tokenizer) is published with
    ``group`` set and ``path`` relative to the model dir; metadata written by an
    older generator has ``path`` flattened to the bare file name, which would
    drop the file next to the weights while ``model.json`` points at the
    directory. ``group`` is authoritative, so the file always lands where the
    runtime looks for it.
    """
    rel = sidecar_relpath(item)
    group = str(item.get("group") or "").strip("/")
    if group and not rel.startswith(f"{group}/"):
        return f"{group}/{Path(rel).name}"
    return rel


def incomplete_download(entry: dict, dest_root: Path, model_filename: str) -> list[str]:
    """Downloaded files of *entry* that are not on disk under *dest_root*.

    ``model.json`` is not checked: bootstrap writes it itself right after the
    downloads (a missing config URL earlier is a separate error).
    """
    missing = []
    if not (dest_root / model_filename).is_file():
        missing.append(model_filename)
    for item in extra_files(entry):
        rel = sidecar_dest(item)
        if not (dest_root / rel).is_file():
            missing.append(rel)
    return missing


def extra_files(entry: dict) -> list[dict]:
    files = entry.get("files") or []
    return [f for f in files if isinstance(f, dict) and f.get("url")]


def sidecar_config_value(role: str, rel: str, item: dict | None = None) -> str:
    """Value a side-car takes in model.json, from its path in the model dir.

    ``decoder`` / ``mmproj`` are files (use the path as downloaded); a
    ``tokenizer`` split over a directory (Whisper ``data_bin/data.bin`` +
    ``data_bin/tokenizer_info.bin``) is configured as that directory, exactly
    like ``examples/asr/whisper-large-v3-turbo/model.json``.  The metadata
    ``group`` key names that directory; older metadata falls back to the first
    path segment.
    """
    if role == "tokenizer":
        group = (item or {}).get("group")
        if group:
            return str(group)
        parts = Path(rel).parts
        if len(parts) > 1:
            return parts[0]
    return rel


def patch_downloaded_config(config_data: dict, weights_name: str, files: list[dict]) -> dict:
    """Fill weights/decoder/tokenizer/mmproj from the files actually downloaded."""
    config_data["weights"] = weights_name
    for item in files:
        role = item.get("role")
        if role in ("decoder", "tokenizer", "mmproj"):
            config_data[role] = sidecar_config_value(role, sidecar_relpath(item), item)
    return config_data


ID_CHARS = re.compile(r"[^a-z0-9]+")


def slugify_model_id(value: object) -> str:
    """Lowercase model id: only ``a-z``, ``0-9`` and ``-`` (never empty).

    ``ProxyRuntime`` keys models by this id, so it is the ``model=`` value
    clients send: ``Qwen3-4B-Instruct-2507`` -> ``qwen3-4b-instruct-2507``.
    """
    slug = ID_CHARS.sub("-", str(value or "").lower()).strip("-")
    return slug or "model"


def collect_model_ids(models_root: Path, exclude_dir: Path | None = None) -> set[str]:
    """Ids already used by ``<models_root>/*/model.json`` (canonical form).

    *exclude_dir* skips the directory being (re)generated, so re-running a
    download keeps the id it wrote before.
    """
    ids: set[str] = set()
    excluded = exclude_dir.resolve() if exclude_dir else None
    for cfg_file in sorted(models_root.glob("*/model.json")):
        try:
            if excluded and cfg_file.parent.resolve() == excluded:
                continue
            with open(cfg_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue  # unreadable / broken configs cannot claim an id
        if isinstance(data, dict):
            value = data.get("id") or data.get("name")
            if value:
                ids.add(slugify_model_id(value))
    return ids


def unique_model_id(base: str, taken: set[str]) -> str:
    """*base*, or ``base-2`` / ``base-3`` … when another model already has it."""
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def finalize_model_id(config_data: dict, models_root: Path, dest_dir: Path,
                      previous: dict | None = None) -> str:
    """Force ``config_data['id']`` to a unique lowercase id (``a-z0-9-``).

    The candidate comes from the id this directory already had (so repeated
    runs and ``--force`` are stable), else from the config / name;
    ``unique_model_id`` disambiguates against every other downloaded model.
    """
    previous_id = str((previous or {}).get("id") or "").strip()
    source = previous_id or config_data.get("id") or config_data.get("name") \
        or dest_dir.name
    model_id = unique_model_id(slugify_model_id(source),
                               collect_model_ids(models_root, exclude_dir=dest_dir))
    original = str(config_data.get("id") or "")
    config_data["id"] = model_id
    if model_id == original:
        print(f"  Model id: {model_id!r}")
    else:
        note = "renamed" if slugify_model_id(original) == model_id else "deduplicated"
        print(f"  Model id: {model_id!r} ({note} from {original!r})")
    return model_id


def mmproj_relpath(files: list[dict]) -> str:
    """Destination path of the ``mmproj`` side-car ('' when there is none)."""
    for item in files:
        if item.get("role") == "mmproj":
            return sidecar_relpath(item)
    return ""


def llm_config_from_entry(entry: dict, weights_name: str, files: list[dict]) -> dict:
    """Minimal LLM/VLM model.json (``weights``, plus ``mmproj`` for VLM).

    Only ``weights`` is required to work (see ``examples/llm/qwen3.json``);
    multimodal entries (``files`` role ``mmproj``) also get the vision
    projector so the server can accept image inputs.
    """
    name = entry.get("name") or Path(weights_name).stem
    metadata = {"family": entry.get("family") or name, "device": "amlogic"}
    mmproj = mmproj_relpath(files)
    if entry.get("multimodal") or mmproj:
        metadata["multimodal"] = True
    config: dict = {
        "id": name,
        "weights": weights_name,
        "sampling_mode": "chain",
        "sampler_params": {"temp": 0.7, "top_p": 0.85, "top_k": 40,
                           "penalty_repeat": 1.15},
        "retain_history": True,
        "loglevel": "ERROR",
        "metadata": metadata,
    }
    if mmproj:
        config["mmproj"] = mmproj
    if entry.get("context"):
        config["context_size"] = int(entry["context"])
    return config


def asr_config_from_entry(entry: dict, weights_name: str, files: list[dict]) -> dict:
    """ASR model.json synthesized from metadata roles (see examples/asr/*/model.json)."""
    model_type = (entry.get("model_type") or "").strip().lower()
    if model_type in ("", "asr"):
        model_type = "whisper" if any(f.get("role") == "decoder" for f in files) \
            else "sensevoice"
    cfg: dict = {
        "id": entry.get("name") or Path(weights_name).stem,
        "model_type": model_type,
        "backend": str(entry.get("backend") or "adla"),
        "weights": weights_name,
    }
    for item in files:
        role = item.get("role")
        if role in ("decoder", "tokenizer"):
            # a tokenizer spread over data_bin/ is configured as that directory
            cfg[role] = sidecar_config_value(role, sidecar_relpath(item), item)
    if "tokenizer" not in cfg:
        raise ValueError(
            "ASR metadata entry needs a tokenizer file (files role=tokenizer)")
    if model_type == "whisper" and "decoder" not in cfg:
        raise ValueError(
            "whisper metadata entry needs a decoder file (files role=decoder)")
    cfg["language"] = str(entry.get("language") or "auto")
    cfg["metadata"] = {"family": model_type, "task": "asr"}
    return cfg


def resolve_entry(entries: list[dict], model_arg: str) -> dict:
    """Find the entry named (or identified) by *model_arg*.

    Exact names win, then case-insensitive names, then the model id clients
    send (see ``slugify_model_id``); only then the separator-insensitive match,
    which refuses to guess when several models collapse to the same key.
    """
    try:
        idx = int(model_arg)
        if idx < 0 or idx >= len(entries):
            print(f"Error: line number {idx} out of range "
                  f"(valid: 0–{len(entries) - 1})", file=sys.stderr)
            sys.exit(1)
        return entries[idx]
    except ValueError:
        pass

    for e in entries:
        if str(e.get("name", "")) == model_arg:
            return e
    lowered = model_arg.lower()
    for e in entries:
        if str(e.get("name", "")).lower() == lowered:
            return e
    wanted_id = slugify_model_id(model_arg)
    for e in entries:
        if slugify_model_id(e.get("name", "")) == wanted_id:
            return e

    norm_query = normalize_name(model_arg)
    matches = [e for e in entries
               if normalize_name(str(e.get("name", ""))) == norm_query]
    if len(matches) > 1:
        names = ", ".join(repr(m.get("name")) for m in matches)
        print(f"Error: '{model_arg}' matches several models ({names}); "
              f"use the line number from `list`", file=sys.stderr)
        sys.exit(1)
    if not matches:
        print(f"Error: no model matching '{model_arg}' found", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def sha1_of_file(path: Path) -> str:
    """Return the hex SHA1 digest of *path*."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Download & verification
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, force: bool = False,
                  expected_size: int | None = None,
                  expected_sha1: str | None = None) -> bool:
    """Download *url* to *dest*.

    Returns ``True`` when the file was actually downloaded, ``False`` when
    skipped because a matching file already exists.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if expected_size is not None and expected_sha1 is not None:
            actual_size = dest.stat().st_size
            actual_sha1 = sha1_of_file(dest)
            if actual_size == expected_size and actual_sha1 == expected_sha1:
                print(f"  File exists and matches: {dest.name}")
                return False
            if actual_size < expected_size:
                # a previous run was interrupted mid-download: the file is
                # truncated, so skipping it would keep a broken model
                print(f"  File is incomplete ({actual_size}/{expected_size} bytes); "
                      f"re-downloading {dest.name}")
            else:
                print(f"  File exists but content differs: {dest.name}")
                print(f"    Expected  size={expected_size}  sha1={expected_sha1}")
                print(f"    Actual    size={actual_size}  sha1={actual_sha1}")
                print("  Skipping (use --force to overwrite)")
                return False
        else:
            print(f"  File already exists: {dest.name}, skipping")
            return False

    print(f"  Downloading {dest.name}")
    try:
        with urllib.request.urlopen(_request(url), timeout=300) as src, \
                open(dest, "wb") as dst:
            total = expected_size or 0
            downloaded = 0
            next_report = 0  # next threshold to print progress
            while True:
                chunk = src.read(65536)
                if not chunk:
                    break
                dst.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    # Show percentage — report every ~2 %
                    pct = downloaded * 100 // total
                    if pct >= next_report:
                        print(f"\r  Progress: {pct}% ({downloaded}/{total} bytes)",
                              end="", flush=True)
                        next_report = pct + 2
    except (urllib.error.URLError, OSError) as e:
        print(f"  Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        if total > 0:
            print()  # newline after progress
    return True


def verify_file(path: Path, expected_size: int | None = None,
                expected_sha1: str | None = None) -> bool:
    """Compare the file at *path* against expected size / SHA1.

    Prints warnings on mismatch and returns ``True`` when everything matches.
    Skips checks that are not provided in metadata.
    """
    if expected_size is None and not expected_sha1:
        return True

    ok = True
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        print(f"  WARNING: size mismatch for {path.name}: "
              f"expected {expected_size}, got {actual_size}")
        ok = False

    if expected_sha1:
        actual_sha1 = sha1_of_file(path)
        if actual_sha1 != expected_sha1:
            print(f"  WARNING: SHA1 mismatch for {path.name}: "
                  f"expected {expected_sha1}, got {actual_sha1}")
            ok = False

    if ok:
        print(f"  Verification passed: {path.name}")
    return ok


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """Print a table of all available models."""
    entries = fetch_metadata(resolve_base_url(args.baseurl))
    if not entries:
        print("No models found.")
        return

    headers = ["#", "Name", "Type", "Thinking", "Multimodal", "Context", "Tools", "Files"]
    col_widths = [len(h) for h in headers]

    rows: list[list[str]] = []
    for i, e in enumerate(entries):
        row = [
            str(i),
            e.get("name", ""),
            e.get("model_type", ""),
            "Yes" if e.get("thinking") else "No",
            "Yes" if e.get("multimodal") else "No",
            str(e.get("context", "")),
            "Yes" if e.get("tools") else "No",
            str(len(e.get("files") or [])),
        ]
        rows.append(row)
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    sep = "-+-".join("-" * w for w in col_widths)
    header = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True))
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, col_widths, strict=True)))


def write_model_config(config_file: Path, config_data: dict) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"  Config written to {config_file}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download a model + its config and write server.yaml."""
    base_url = resolve_base_url(args.baseurl)
    workdir = Path(args.target).resolve()
    model_dir = (Path(args.model_dir).resolve()
                 if args.model_dir else workdir / "models")
    config_dir = (Path(args.config_dir).resolve()
                  if args.config_dir else workdir / "config")
    force = args.force

    entries = fetch_metadata(base_url)
    # metadata addresses are repo-relative: they follow the base URL we were
    # pointed at (ModelScope / HuggingFace / mirror), not the hub it was made on
    entry = resolve_entry_addresses(resolve_entry(entries, args.model), base_url)

    model_name = entry["name"]
    model_url = entry["url"]
    config_url = entry.get("config") or ""
    expected_size = entry.get("size")
    expected_sha1 = entry.get("sha1")
    subdir = model_subdir(entry)
    sidecars = extra_files(entry)
    dest_root = model_dir / subdir
    asr = is_asr_entry(entry)

    print(f"Model:  {model_name}")
    print(f"Type:   {entry.get('model_type', '')}"
          f"{' (multimodal)' if entry.get('multimodal') else ''}")
    print(f"Dir:    {dest_root}")
    print(f"URL:    {model_url}")
    print(f"Config: {config_url or '(none — model.json will be generated)'}")
    expected_mmproj = mmproj_relpath(sidecars)
    if entry.get("multimodal") and not expected_mmproj:
        print("WARNING: entry is multimodal but lists no mmproj file; "
              "image inputs will be unavailable", file=sys.stderr)

    model_filename = filename_from_url(model_url)
    model_path = dest_root / model_filename

    print("\n--- Model download ---")
    download_file(model_url, model_path, force, expected_size, expected_sha1)

    if model_path.exists() and (expected_size is not None or expected_sha1):
        print("\n--- Verification ---")
        verify_file(model_path, expected_size, expected_sha1)

    if sidecars:
        print("\n--- Extra files ---")
        for item in sidecars:
            rel = sidecar_dest(item)  # honours the side-car's group directory
            dest = dest_root / rel
            print(f"  {rel}")
            download_file(item["url"], dest, force, item.get("size"), item.get("sha1"))
            if dest.exists() and (item.get("size") is not None or item.get("sha1")):
                verify_file(dest, item.get("size"), item.get("sha1"))

    config_file = dest_root / "model.json"
    print("\n--- Model configuration ---")
    dest_root.mkdir(parents=True, exist_ok=True)

    # every file the entry references must be on disk: a failed side-car
    # download used to be easy to miss and only surfaced later as a confusing
    # runtime error (e.g. whisper "cannot open tokenizer data_bin/…")
    incomplete = incomplete_download(entry, dest_root, model_filename)
    if incomplete:
        print("ERROR: download incomplete, missing file(s):", file=sys.stderr)
        for rel in incomplete:
            print(f"  {rel}", file=sys.stderr)
        print("  re-run this command to retry (existing files are kept).",
              file=sys.stderr)

    # read once: the id recorded before is reused, so re-runs stay stable
    previous_cfg: dict | None = None
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            previous_cfg = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            previous_cfg = None

    config_data: dict | None = None  # None => keep the existing model.json
    if config_url:
        # Original path: fetch the remote model.json (ASR side-cars, chat
        # formats, …) and patch it with the files actually downloaded.
        try:
            with urllib.request.urlopen(_request(config_url), timeout=30) as resp:
                config_data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as e:
            print(f"Error downloading config: {e}", file=sys.stderr)
            sys.exit(1)
        patch_downloaded_config(config_data, model_filename, sidecars)
    elif asr:
        # No remote config: synthesize one following examples/asr/*.
        try:
            config_data = asr_config_from_entry(entry, model_filename, sidecars)
        except ValueError as e:
            print(f"Error: {e} (ASR models require the tokenizer attachment; "
                  f"see scripts/gen_metadata.py)", file=sys.stderr)
            sys.exit(1)
    else:
        # LLM/VLM: only weights (plus mmproj for VLM) are needed to work,
        # see examples/llm/qwen3.json.
        if previous_cfg is not None and not force and \
                str(previous_cfg.get("weights")) == model_filename and \
                str(previous_cfg.get("mmproj") or "") == expected_mmproj:
            print(f"  model.json already exists for {model_filename}, "
                  f"keeping it (use --force to regenerate)")
        else:
            if previous_cfg is not None:
                print(f"  model.json is stale "
                      f"(weights={previous_cfg.get('weights')!r}/"
                      f"mmproj={previous_cfg.get('mmproj')!r} -> "
                      f"{model_filename!r}/{expected_mmproj!r}); regenerating")
            config_data = llm_config_from_entry(entry, model_filename, sidecars)

    if config_data is not None:
        # every generated model.json gets a unique, lowercase id (a-z, 0-9, -)
        finalize_model_id(config_data, model_dir, dest_root, previous_cfg)
        write_model_config(config_file, config_data)

    print("\n--- Server configuration ---")
    server_yaml = config_dir / "server.yaml"
    server_yaml.parent.mkdir(parents=True, exist_ok=True)

    if server_yaml.exists():
        with open(server_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    svr = cfg.setdefault("server", {})
    svr["host"] = args.host
    svr["port"] = args.port
    svr.setdefault("api_key", SERVER_YAML_TEMPLATE["server"]["api_key"])
    svr.setdefault("log_level", SERVER_YAML_TEMPLATE["server"]["log_level"])
    svr.setdefault("title", SERVER_YAML_TEMPLATE["server"]["title"])
    svr.setdefault("version", SERVER_YAML_TEMPLATE["server"]["version"])

    mdl = cfg.setdefault("models", {})
    try:
        rel = os.path.relpath(model_dir, config_dir)
    except ValueError:
        rel = str(model_dir)
    mdl["root_dir"] = rel
    # keep an explicit value in the file (existing edits win)
    mdl.setdefault("reject_when_busy",
                   SERVER_YAML_TEMPLATE["models"]["reject_when_busy"])
    enabled = mdl.setdefault("enabled", list(SERVER_YAML_TEMPLATE["models"]["enabled"]))
    if not isinstance(enabled, list):
        enabled = mdl["enabled"] = list(enabled or [])
    if subdir not in enabled and not incomplete:
        enabled.append(subdir)
    # keep the config loadable: only list directories that really exist
    missing = [d for d in enabled if not (model_dir / str(d) / "model.json").is_file()]
    for d in missing:
        print(f"  WARNING: models.enabled lists '{d}' but "
              f"{model_dir / str(d)}/model.json is missing; disabling it",
              file=sys.stderr)
        enabled.remove(d)

    with open(server_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)
    print(f"  Server config written to {server_yaml}")

    if incomplete:
        print(f"\nERROR: '{subdir}' was NOT enabled in server.yaml because "
              f"{len(incomplete)} file(s) are missing.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proxy server bootstrap — list, download and configure models.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--baseurl",
        help="Model-zoo source: modelscope:[owner/repo][@rev] | ms:… | "
             "hf:[org/repo][@rev] | huggingface:… | modelscope/ms/hf/… (bare, "
             "hub default repo) | owner/repo | https://… | /local/dir "
             f"(default: {DEFAULT_BASE_URL}; env AMLLLM_MODEL_ZOO also honored)")
    parser.add_argument(
        "--token", default=None,
        help="Bearer token for private zoo repos (env AMLLLM_ZOO_TOKEN / "
             "MODELSCOPE_API_KEY / HF_TOKEN also honored)")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- list ---
    p_list = sub.add_parser("list", help="List all available models")
    p_list.set_defaults(func=cmd_list)

    # --- download ---
    p_dl = sub.add_parser("download", help="Download a model and create config")
    p_dl.add_argument("model", help="Line number (int) or model name (str)")
    p_dl.add_argument("-t", "--target", default=".",
                      help="Working directory (default: current dir)")
    p_dl.add_argument("-m", "--model_dir",
                      help="Model storage directory (default: <target>/models)")
    p_dl.add_argument("-c", "--config_dir",
                      help="Config directory (default: <target>/config)")
    p_dl.add_argument("--host", default="0.0.0.0",
                      help="Server listen IP (default: 0.0.0.0)")
    p_dl.add_argument("--port", type=int, default=8000,
                      help="Server listen port (default: 8000)")
    p_dl.add_argument("-f", "--force", action="store_true",
                      help="Force overwrite of existing files")
    p_dl.set_defaults(func=cmd_download)

    args = parser.parse_args()
    if args.token:
        # `--token` feeds the download path; $MODELSCOPE_API_KEY is honored too
        os.environ["AMLLLM_ZOO_TOKEN"] = args.token
        os.environ.setdefault("MODELSCOPE_API_KEY", args.token)
    args.func(args)


if __name__ == "__main__":
    main()
