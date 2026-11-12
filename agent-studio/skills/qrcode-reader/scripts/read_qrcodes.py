#!/usr/bin/env python3
"""Run the bundled native QR code demo and emit portable JSON results."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="image file or directory")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("QRCODE_BIN", ROOT / "assets/bin/qr_demo")))
    p.add_argument("--model", type=Path, default=Path(os.environ.get("QRCODE_MODEL", ROOT / "assets/models/qrcode_int8_A311Y3.adla")))
    p.add_argument("--conf", type=float, default=0.8)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--pad", type=int, default=60)
    p.add_argument("--timeout", type=float, default=180, help="native process timeout in seconds")
    p.add_argument("--recursive", action="store_true")
    return p


def run(args):
    source = args.input.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    binary, model = args.binary.expanduser().resolve(), args.model.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Inference executable is missing or not executable: {binary}")
    if not model.is_file():
        raise ValueError(f"Model does not exist: {model}")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in [args.conf, args.nms]):
        raise ValueError("--conf and --nms must be between 0 and 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be a finite positive number")
    if args.pad < 0:
        raise ValueError("--pad must not be negative")
    output = args.output_dir.expanduser().resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="qrcode-results-", dir=Path.cwd()))
    output.mkdir(parents=True, exist_ok=True)
    # Each invocation owns a fresh result path; never accept a stale result.json.
    with tempfile.TemporaryDirectory(prefix=".qrcode-", dir=output) as temp:
        report = Path(temp) / "result.json"
        command = [str(binary), "--model", str(model), "--input", str(source), "--output-dir", str(output), "--json-output", str(report), "--conf", str(args.conf), "--nms", str(args.nms)]
        command += ["--pad", str(args.pad)]
        if args.recursive:
            command.append("--recursive")
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"Inference timed out after {args.timeout:g} seconds; completed images remain in {output}") from error
        log = output / "backend.log"
        log.write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
        if not report.is_file():
            raise ValueError(f"Inference exited {process.returncode} without JSON results; check {log} and the system NPU driver/SDK")
        payload = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError("Inference did not return a files list")
    failed = any(f.get("status") != "ok" for f in payload["files"])
    success = any(f.get("status") == "ok" for f in payload["files"])
    ok = process.returncode == 0 and not failed and success and not payload.get("error")
    payload["status"] = "ok" if ok else "partial" if success else "error"
    payload["backend_exit_code"] = process.returncode

    artifacts = [{"path": str(log)}]
    for item in payload["files"]:
        if item.get("status") == "ok" and item.get("result_image"):
            image = Path(item["result_image"])
            if not image.is_file():
                raise ValueError(f"Backend returned a missing result image: {image}")
            artifacts.append({"path": str(image)})
    final = output / "result.json"
    payload["artifacts"] = artifacts + [{"path": str(final)}]
    final.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, 0 if ok else process.returncode or 1


def main():
    args = parser().parse_args()
    try:
        result, code = run(args)
    except (OSError, ValueError) as error:
        result, code = {"status": "error", "error": str(error), "files": [], "artifacts": []}, 2
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
