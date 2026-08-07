#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml>=6.0",
# ]
# ///

"""Bootstrap tool for proxy server — model listing, downloading, and configuration."""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

DEFAULT_BASE_URL = "https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/LLM"
                
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
        "enabled": ["default"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request(url: str) -> urllib.request.Request:
    """Create a Request with a common User-Agent header."""
    return urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AMLLLM-Bootstrap/1.0)"},
    )


def fetch_metadata(base_url: str) -> list[dict]:
    """Fetch metadata.jsonl from the remote and return parsed entries."""
    url = f"{base_url.rstrip('/')}/metadata.jsonl"
    print(f"Fetching metadata from {url}")
    try:
        with urllib.request.urlopen(_request(url), timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
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
            print(f"  File exists but content differs: {dest.name}")
            print(f"    Expected  size={expected_size}  sha1={expected_sha1}")
            print(f"    Actual    size={actual_size}  sha1={actual_sha1}")
            if not force:
                print(f"  Skipping (use --force to overwrite)")
                return False
            print(f"  Overwriting (--force set) …")
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
    except urllib.error.URLError as e:
        print(f"  Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        if total > 0:
            print()  # newline after progress
    return True


def verify_file(path: Path, expected_size: int, expected_sha1: str) -> bool:
    """Compare the file at *path* against expected size / SHA1.

    Prints warnings on mismatch and returns ``True`` when everything matches.
    """
    ok = True
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        print(f"  WARNING: size mismatch for {path.name}: "
              f"expected {expected_size}, got {actual_size}")
        ok = False

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
    entries = fetch_metadata(args.baseurl or DEFAULT_BASE_URL)
    if not entries:
        print("No models found.")
        return

    headers = ["#", "Name", "Type", "Thinking", "Multimodal", "Context", "Tools"]
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
        ]
        rows.append(row)
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    sep = "-+-".join("-" * w for w in col_widths)
    header = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, col_widths)))


def cmd_download(args: argparse.Namespace) -> None:
    """Download a model + its config and write server.yaml."""
    base_url = args.baseurl or DEFAULT_BASE_URL
    workdir = Path(args.target).resolve()
    model_dir = (Path(args.model_dir).resolve()
                 if args.model_dir else workdir / "models")
    config_dir = (Path(args.config_dir).resolve()
                  if args.config_dir else workdir / "config")
    force = args.force

    entries = fetch_metadata(base_url)

    # --- resolve model entry -------------------------------------------------
    model_arg: str = args.model
    entry: dict | None = None

    try:
        idx = int(model_arg)
        if idx < 0 or idx >= len(entries):
            print(f"Error: line number {idx} out of range "
                  f"(valid: 0–{len(entries) - 1})", file=sys.stderr)
            sys.exit(1)
        entry = entries[idx]
    except ValueError:
        norm_query = normalize_name(model_arg)
        for e in entries:
            if normalize_name(e.get("name", "")) == norm_query:
                entry = e  # keep overwriting — last match survives
        if entry is None:
            print(f"Error: no model matching '{model_arg}' found",
                  file=sys.stderr)
            sys.exit(1)

    model_name = entry["name"]
    model_url = entry["url"]
    config_url = entry["config"]
    expected_size = entry.get("size")
    expected_sha1 = entry.get("sha1")

    print(f"Model:  {model_name}")
    print(f"URL:    {model_url}")
    print(f"Config: {config_url}")

    # --- download model file -------------------------------------------------
    model_filename = filename_from_url(model_url)
    model_path = model_dir / "default" / model_filename

    print("\n--- Model download ---")
    downloaded = download_file(model_url, model_path, force,
                               expected_size, expected_sha1)

    # Verify after download (or if file already existed)
    if model_path.exists():
        print("\n--- Verification ---")
        verify_file(model_path, expected_size, expected_sha1)

    # --- download & modify config JSON ---------------------------------------
    print("\n--- Config download ---")
    model_dir_default = model_dir / "default"
    model_dir_default.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(_request(config_url), timeout=30) as resp:
            config_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error downloading config: {e}", file=sys.stderr)
        sys.exit(1)

    config_data["weights"] = model_filename

    config_file = model_dir_default / "model.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"  Config written to {config_file}")

    # --- server.yaml ---------------------------------------------------------
    print("\n--- Server configuration ---")
    server_yaml = config_dir / "server.yaml"
    server_yaml.parent.mkdir(parents=True, exist_ok=True)

    if server_yaml.exists():
        with open(server_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # Merge server section
    svr = cfg.setdefault("server", {})
    svr["host"] = args.host
    svr["port"] = args.port
    svr.setdefault("api_key", SERVER_YAML_TEMPLATE["server"]["api_key"])
    svr.setdefault("log_level", SERVER_YAML_TEMPLATE["server"]["log_level"])
    svr.setdefault("title", SERVER_YAML_TEMPLATE["server"]["title"])
    svr.setdefault("version", SERVER_YAML_TEMPLATE["server"]["version"])

    # Merge models section — always compute relative root_dir
    mdl = cfg.setdefault("models", {})
    try:
        rel = os.path.relpath(model_dir, config_dir)
    except ValueError:
        rel = str(model_dir)
    mdl["root_dir"] = rel
    mdl.setdefault("enabled", SERVER_YAML_TEMPLATE["models"]["enabled"])

    with open(server_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)
    print(f"  Server config written to {server_yaml}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proxy server bootstrap — list, download and configure models.")
    parser.add_argument(
        "--baseurl",
        help=f"Base URL for model zoo (default: {DEFAULT_BASE_URL})",
    )

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
    args.func(args)


if __name__ == "__main__":
    main()
