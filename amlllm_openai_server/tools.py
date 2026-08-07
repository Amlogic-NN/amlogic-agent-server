# -*- coding: utf-8 -*-
"""CLI entry point for the tool-calling evaluation framework.

Usage::

    python -m tools.cli \\
        --test-data tests/fixtures/sample_eval.json \\
        --config config/server.yaml \\
        --n-repeat 3 \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schema import TestCaseEntry, CloudApiConfig, RunRecord  # noqa: E402
from .evaluator import Evaluator, create_local_runner_from_yaml  # noqa: E402
from .runners import CloudModelRunner  # noqa: E402
from .report import generate_report  # noqa: E402
from .metrics import compile_all_stats  # noqa: E402


logger = logging.getLogger("tools.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate tool-calling accuracy of local vs cloud LLMs."
    )

    sub = parser.add_subparsers(dest="mode", help="Operation mode")

    # ---- evaluate mode (default) ----
    eval_parser = sub.add_parser("run", help="Run inference and evaluate")
    eval_parser.add_argument(
        "--test-data", required=True,
        help="Path to the JSON test cases file.",
    )
    eval_parser.add_argument(
        "--config", default=None,
        help="Path to the server YAML config for the local model (required unless --skip-local).",
    )
    eval_parser.add_argument(
        "--cloud-config",
        default="test_data/test_config.yaml",
        help="Path to the cloud API YAML config (default: test_data/test_config.yaml).",
    )
    eval_parser.add_argument(
        "--n-repeat", type=int, default=3,
        help="Number of times to repeat each conversation test (default: 3).",
    )
    eval_parser.add_argument(
        "--output-dir", default="results",
        help="Directory for output files (default: results/).",
    )
    eval_parser.add_argument(
        "--skip-local", action="store_true",
        help="Skip local model evaluation.",
    )
    eval_parser.add_argument(
        "--skip-cloud", action="store_true",
        help="Skip cloud model evaluation.",
    )
    eval_parser.add_argument("--cloud-base-url", default=None, help="Override cloud base URL.")
    eval_parser.add_argument("--cloud-api-key", default=None, help="Override cloud API key.")
    eval_parser.add_argument("--cloud-model", default=None, help="Override cloud model name.")
    eval_parser.add_argument(
        "--exclude-latency", default=None, nargs="*",
        help="Model labels to exclude from the latency comparison chart.",
    )

    # ---- compare mode ----
    cmp_parser = sub.add_parser("compare", help="Compare two or more existing JSONL result files")
    cmp_parser.add_argument(
        "--jsonl", required=True, nargs="+",
        help="Two or more JSONL result file paths.",
    )
    cmp_parser.add_argument(
        "--label", default=None, nargs="*",
        help="Labels for each JSONL (same number as --jsonl). If omitted, use model_type from file.",
    )
    cmp_parser.add_argument(
        "--output-dir", default="results/compare",
        help="Directory for output files (default: results/compare).",
    )
    cmp_parser.add_argument(
        "--exclude-latency", default=None, nargs="*",
        help="Model labels to exclude from the latency comparison chart.",
    )

    # ---- report mode ----
    rpt_parser = sub.add_parser("report", help="Score and generate report from existing JSONL (no model inference)")
    rpt_parser.add_argument(
        "--jsonl", required=True,
        help="Path to the existing JSONL result file.",
    )
    rpt_parser.add_argument(
        "--output-dir", default="results/report",
        help="Directory for output files (default: results/report).",
    )
    rpt_parser.add_argument(
        "--exclude-latency", default=None, nargs="*",
        help="Model labels to exclude from the latency comparison chart.",
    )

    # Shared
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    return parser


def main():
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "compare":
        _cmd_compare(args)
    elif args.mode == "report":
        _cmd_report(args)
    else:
        _cmd_run(args)


def _cmd_run(args):
    # --- Load test data ---
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.error("Test data file not found: %s", test_data_path)
        sys.exit(1)

    test_cases_raw = json.loads(test_data_path.read_text(encoding="utf-8"))
    if isinstance(test_cases_raw, dict):
        test_cases_raw = [test_cases_raw]
    if not isinstance(test_cases_raw, list):
        logger.error("Test data must be a JSON array or object.")
        sys.exit(1)

    test_cases: list[TestCaseEntry] = []
    for i, entry in enumerate(test_cases_raw):
        try:
            test_cases.append(TestCaseEntry.model_validate(entry))
        except Exception as exc:
            logger.error("Invalid test case at index %d: %s", i, exc)
            sys.exit(1)
    logger.info("Loaded %d test case entries.", len(test_cases))

    # Count total conversations
    total_convs = sum(len(e.messages) for e in test_cases)
    logger.info("Total conversations across all entries: %d", total_convs)
    logger.info("Repeats per conversation: %d", args.n_repeat)
    logger.info(
        "Estimated total runs: local=%d, cloud=%d",
        0 if args.skip_local else total_convs * args.n_repeat,
        0 if args.skip_cloud else total_convs * args.n_repeat,
    )

    # --- Create runners ---
    local_runner = None
    if not args.skip_local:
        if not args.config:
            logger.error("--config is required when not using --skip-local.")
            sys.exit(1)
        local_runner = create_local_runner_from_yaml(args.config)

    cloud_runner = None
    if not args.skip_cloud:
        cloud_cfg_path = Path(args.cloud_config)
        if not cloud_cfg_path.exists():
            logger.error("Cloud config file not found: %s", cloud_cfg_path)
            sys.exit(1)
        cloud_config = CloudApiConfig.from_yaml(str(cloud_cfg_path))
        # Apply CLI overrides
        if args.cloud_base_url:
            cloud_config.base_url = args.cloud_base_url.rstrip("/")
        if args.cloud_api_key:
            cloud_config.api_key = args.cloud_api_key
        if args.cloud_model:
            cloud_config.model = args.cloud_model
        logger.info(
            "Cloud config: base_url=%s model=%s thinking=%s",
            cloud_config.base_url, cloud_config.model, cloud_config.thinking,
        )
        cloud_runner = CloudModelRunner(cloud_config)

    if local_runner is None and cloud_runner is None:
        logger.error("Both --skip-local and --skip-cloud specified; nothing to do.")
        sys.exit(1)

    # --- Run evaluation ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "eval_results.jsonl"

    evaluator = Evaluator(
        local_runner=local_runner,
        cloud_runner=cloud_runner,
        n_repeat=args.n_repeat,
    )
    evaluator.evaluate(test_cases, jsonl_path)
    logger.info("Results written to: %s", jsonl_path)

    # --- Compute stats ---
    stats = evaluator.compute_stats_from_jsonl(jsonl_path)
    for mt, s in stats.items():
        d = s.to_dict()
        logger.info(
            "%s: TP=%d FP=%d FN=%d TN=%d Recall=%.4f Precision=%.4f Accuracy=%.4f F1=%.4f",
            mt, d["TP"], d["FP"], d["FN"], d["TN"],
            d["Recall"], d["Precision"], d["Accuracy"], d["F1"],
        )

    # --- Generate report ---
    report_path = generate_report(jsonl_path, output_dir, exclude_latency=args.exclude_latency)
    logger.info("Report written to: %s", report_path)
    logger.info("Charts written to: %s", output_dir / "charts")

    # --- Print summary to stdout ---
    print("\n" + "=" * 60)
    print("  TOOL-CALLING EVALUATION SUMMARY")
    print("=" * 60)
    for mt, s in stats.items():
        d = s.to_dict()
        print(f"\n  [{mt}]")
        print(f"    Total runs:  {d['total_runs']}")
        print(f"    Labeled:     {d['total_labeled']}")
        print(f"    TP={d['TP']}  FP={d['FP']}  FN={d['FN']}  TN={d['TN']}")
        print(f"    Recall:      {d['Recall']:.4f}")
        print(f"    Precision:   {d['Precision']:.4f}")
        print(f"    Accuracy:    {d['Accuracy']:.4f}")
        print(f"    F1:          {d['F1']:.4f}")
        print(f"    Errors:      {d['error_count']}")
        print(f"    Avg Latency: {d['avg_latency_ms']:.1f} ms")
    print("\n" + "=" * 60)


def _cmd_compare(args):
    """Compare mode: load N JSONL files, merge, compute stats, generate report."""
    jsonl_paths = [Path(p) for p in args.jsonl]
    if len(jsonl_paths) < 2:
        logger.error("--jsonl requires at least 2 files, got %d.", len(jsonl_paths))
        sys.exit(1)

    labels = args.label if args.label else []
    if labels and len(labels) != len(jsonl_paths):
        logger.error(
            "--label count (%d) must equal --jsonl count (%d), or omit --label entirely.",
            len(labels), len(jsonl_paths),
        )
        sys.exit(1)

    all_records: list[RunRecord] = []
    for i, jp in enumerate(jsonl_paths):
        if not jp.exists():
            logger.error("JSONL file not found: %s", jp)
            sys.exit(1)
        label = labels[i] if i < len(labels) else None
        records = _load_jsonl(jp, label)
        logger.info("Loaded %d records from %s (label=%s)", len(records), jp, label or "<from file>")
        all_records.extend(records)

    logger.info("Total merged records: %d", len(all_records))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_jsonl = output_dir / "eval_results.jsonl"
    with merged_jsonl.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(r.model_dump_json() + "\n")
    logger.info("Merged JSONL written to: %s", merged_jsonl)

    stats = compile_all_stats(all_records)
    for mt, s in stats.items():
        d = s.to_dict()
        logger.info(
            "%s: TP=%d FP=%d FN=%d TN=%d Recall=%.4f Precision=%.4f Accuracy=%.4f F1=%.4f",
            mt, d["TP"], d["FP"], d["FN"], d["TN"],
            d["Recall"], d["Precision"], d["Accuracy"], d["F1"],
        )

    report_path = generate_report(merged_jsonl, output_dir, exclude_latency=args.exclude_latency)
    logger.info("Report written to: %s", report_path)
    logger.info("Charts written to: %s", output_dir / "charts")

    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    for mt, s in stats.items():
        d = s.to_dict()
        print(f"\n  [{mt}]")
        print(f"    Total runs:  {d['total_runs']}")
        print(f"    Labeled:     {d['total_labeled']}")
        print(f"    TP={d['TP']}  FP={d['FP']}  FN={d['FN']}  TN={d['TN']}")
        print(f"    Recall:      {d['Recall']:.4f}")
        print(f"    Precision:   {d['Precision']:.4f}")
        print(f"    Accuracy:    {d['Accuracy']:.4f}")
        print(f"    F1:          {d['F1']:.4f}")
        print(f"    NameAcc:     {d['NameAccuracy']:.4f}")
        print(f"    VerifPass:   {d['VerifierPassRate']:.4f}")
        print(f"    Errors:      {d['error_count']}")
        print(f"    Avg Latency: {d['avg_latency_ms']:.1f} ms")
    print("\n" + "=" * 60)


def _cmd_report(args):
    """Report mode: load an existing JSONL, compute stats, generate report (no model inference)."""
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        logger.error("JSONL file not found: %s", jsonl_path)
        sys.exit(1)

    records = _load_jsonl(jsonl_path, label_override=None)
    logger.info("Loaded %d records from %s", len(records), jsonl_path)

    if not records:
        logger.error("No records found in JSONL file.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = compile_all_stats(records)
    for mt, s in stats.items():
        d = s.to_dict()
        logger.info(
            "%s: TP=%d FP=%d FN=%d TN=%d Recall=%.4f Precision=%.4f Accuracy=%.4f F1=%.4f",
            mt, d["TP"], d["FP"], d["FN"], d["TN"],
            d["Recall"], d["Precision"], d["Accuracy"], d["F1"],
        )

    report_path = generate_report(jsonl_path, output_dir, exclude_latency=args.exclude_latency)
    logger.info("Report written to: %s", report_path)
    logger.info("Charts written to: %s", output_dir / "charts")

    print("\n" + "=" * 60)
    print("  REPORT SUMMARY")
    print("=" * 60)
    for mt, s in stats.items():
        d = s.to_dict()
        print(f"\n  [{mt}]")
        print(f"    Total runs:  {d['total_runs']}")
        print(f"    Labeled:     {d['total_labeled']}")
        print(f"    TP={d['TP']}  FP={d['FP']}  FN={d['FN']}  TN={d['TN']}")
        print(f"    Recall:      {d['Recall']:.4f}")
        print(f"    Precision:   {d['Precision']:.4f}")
        print(f"    Accuracy:    {d['Accuracy']:.4f}")
        print(f"    F1:          {d['F1']:.4f}")
        print(f"    NameAcc:     {d['NameAccuracy']:.4f}")
        print(f"    VerifPass:   {d['VerifierPassRate']:.4f}")
        print(f"    Errors:      {d['error_count']}")
        print(f"    Avg Latency: {d['avg_latency_ms']:.1f} ms")
    print("\n" + "=" * 60)


def _load_jsonl(path: Path, label_override: Optional[str]) -> list[RunRecord]:
    """Load records from a JSONL file, optionally overriding model_type."""
    records: list[RunRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = RunRecord(**json.loads(line))
            if label_override:
                r.model_type = label_override
            records.append(r)
    return records


if __name__ == "__main__":
    main()
