# -*- coding: utf-8 -*-
"""CLI entry point for the agent capability benchmark framework.

Usage::

    # Download datasets
    python -m benchmark_tool.cli download

    # Create fixed 500-item samples
    python -m benchmark_tool.cli sample

    # Run benchmark (data collection)
    python -m benchmark_tool.cli run --dataset mmlu_pro --summary --save-responses

    # Score collected JSONL (offline)
    python -m benchmark_tool.cli score --jsonl results/eval_results.jsonl

    # Generate report from scored JSONL
    python -m benchmark_tool.cli report --jsonl results/eval_results_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional


from .datasets import get_dataset  # noqa: E402
from .sampler import SampleManager  # noqa: E402
from .runners import (  # noqa: E402
    BenchmarkLocalRunner, BenchmarkCloudRunner,
    create_benchmark_local_runner, create_benchmark_cloud_runner,
)
from .evaluator import BenchmarkEvaluator  # noqa: E402
from .scorer import BenchmarkScorer  # noqa: E402
from .report import generate_benchmark_report  # noqa: E402
from ..schema import CloudApiConfig  # noqa: E402

logger = logging.getLogger("benchmark.cli")

ALL_DATASETS = ["mmlu", "mmlu_pro", "tau2_bench", "bfcl", "mmstar", "ocrbench"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent Capability Benchmark — evaluate model Agent abilities."
    )
    sub = parser.add_subparsers(dest="mode", help="Operation mode")

    # ---- download ----
    d_parser = sub.add_parser("download", help="Download all benchmark datasets")
    d_parser.add_argument("--dataset", nargs="*", default=None,
                          help="Specific datasets to download (default: all).")
    d_parser.add_argument("--cache-dir", default="dataset",
                          help="Dataset cache directory (default: dataset/).")

    # ---- sample ----
    s_parser = sub.add_parser("sample", help="Create fixed random samples for summary tests")
    s_parser.add_argument("--dataset", nargs="*", default=None,
                          help="Specific datasets to sample (default: all).")
    s_parser.add_argument("--n", type=int, default=500,
                          help="Sample size per dataset (default: 500).")
    s_parser.add_argument("--seed", type=int, default=42,
                          help="Random seed for reproducibility (default: 42).")
    s_parser.add_argument("--cache-dir", default="dataset",
                          help="Dataset cache directory.")

    # ---- run ----
    r_parser = sub.add_parser("run", help="Run benchmark evaluation (data collection)")
    r_parser.add_argument("--dataset", required=True, nargs="+",
                          help="Dataset(s) to evaluate: mmlu_pro, tau2_bench, bfcl, or 'all'.")
    r_parser.add_argument("--config", default=None,
                          help="Path to server YAML config for local model.")
    r_parser.add_argument("--cloud-config", default="test_data/test_config.yaml",
                          help="Path to cloud API YAML config.")
    r_parser.add_argument("--output-dir", default="results",
                          help="Directory for output files (default: results/).")
    r_parser.add_argument("--n-repeat", type=int, default=1,
                          help="Number of repeats per item (default: 1).")
    r_parser.add_argument("--summary", action="store_true",
                          help="Use fixed 500-item sample instead of full dataset.")
    r_parser.add_argument("--save-responses", action="store_true", default=True,
                          help="Record full response text in JSONL (default: on).")
    r_parser.add_argument("--no-save-responses", dest="save_responses", action="store_false",
                          help="Disable recording full response text.")
    r_parser.add_argument("--skip-local", action="store_true",
                          help="Skip local model evaluation.")
    r_parser.add_argument("--skip-cloud", action="store_true",
                          help="Skip cloud model evaluation.")
    r_parser.add_argument("--max-tokens", type=int, default=None,
                          help="Maximum tokens to generate per inference (default: model default).")
    r_parser.add_argument("--mmstar-resize", type=int, nargs=2, metavar=("W", "H"),
                          default=None,
                          help="Resize MMStar images to W x H pixels before evaluation. "
                               "If not set, original image sizes are used.")
    r_parser.add_argument("--ocrbench-resize", type=int, nargs=2, metavar=("W", "H"),
                          default=None,
                          help="Resize OCRBench images to W x H pixels before evaluation. "
                               "If not set, original image sizes are used.")

    # ---- score ----
    sc_parser = sub.add_parser("score", help="Score collected JSONL (offline)")
    sc_parser.add_argument("--jsonl", required=True, nargs="+",
                           help="Path(s) to collected JSONL file(s).")
    sc_parser.add_argument("--output-dir", default=None,
                           help="Directory for scored output (default: same as JSONL).")
    sc_parser.add_argument("--labels", nargs="*", default=None,
                           help="Display labels for each JSONL (same order).")

    # ---- report ----
    rep_parser = sub.add_parser("report", help="Generate report from scored JSONL(s)")
    rep_parser.add_argument("--jsonl", required=True, nargs="+",
                            help="Path(s) to scored JSONL file(s).")
    rep_parser.add_argument("--output-dir", default=None,
                            help="Directory for report output.")
    rep_parser.add_argument("--labels", nargs="*", default=None,
                            help="Display labels for each JSONL (same order).")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    log_level = getattr(args, "log_level", "INFO") or "INFO"
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.mode:
        parser.print_help()
        sys.exit(1)
    if args.mode == "download":
        _cmd_download(args)
    elif args.mode == "sample":
        _cmd_sample(args)
    elif args.mode == "run":
        _cmd_run(args)
    elif args.mode == "score":
        _cmd_score(args)
    elif args.mode == "report":
        _cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_download(args):
    dataset_names = args.dataset or ALL_DATASETS
    for name in dataset_names:
        try:
            ds = get_dataset(name)
            ds.download()
            logger.info("Downloaded: %s", name)
        except Exception as exc:
            logger.error("Failed to download %s: %s", name, exc)


def _cmd_sample(args):
    mgr = SampleManager(cache_dir=args.cache_dir)
    dataset_names = args.dataset or ALL_DATASETS
    for name in dataset_names:
        try:
            items = mgr.create_sample(name, n=args.n, seed=args.seed)
            logger.info("Sampled %s: %d items", name, len(items))
        except Exception as exc:
            logger.error("Failed to sample %s: %s", name, exc)
    print(f"\nSamples created ({args.n} items each, seed={args.seed}).")


def _cmd_run(args):
    # Resolve datasets
    if "all" in (d.lower() for d in args.dataset):
        dataset_names = ALL_DATASETS
    else:
        dataset_names = [d.lower().strip() for d in args.dataset]

    # Create runners
    local_runner = None
    if not args.skip_local:
        if not args.config:
            logger.error("--config is required unless --skip-local is set")
            sys.exit(1)
        local_runner = create_benchmark_local_runner(args.config)

    cloud_runner = None
    if not args.skip_cloud:
        cloud_cfg_path = Path(args.cloud_config)
        if not cloud_cfg_path.exists():
            logger.error("Cloud config not found: %s", cloud_cfg_path)
            sys.exit(1)
        cloud_config = CloudApiConfig.from_yaml(str(cloud_cfg_path))
        logger.info("Cloud: base_url=%s model=%s", cloud_config.base_url, cloud_config.model)
        cloud_runner = create_benchmark_cloud_runner(cloud_config)

    if local_runner is None and cloud_runner is None:
        logger.error("No runners configured.")
        sys.exit(1)

    # Import locally to avoid circular imports
    from .evaluator import BenchmarkEvaluator as _Evaluator  # noqa: E402

    evaluator = _Evaluator(
        local_runner=local_runner,
        cloud_runner=cloud_runner,
        n_repeat=args.n_repeat,
        save_responses=args.save_responses,
        max_tokens=args.max_tokens,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in dataset_names:
        ds = get_dataset(ds_name)
        # Apply MMStar image resize if configured
        if getattr(args, "mmstar_resize", None) is not None:
            if hasattr(ds, "resize"):
                ds.resize = tuple(args.mmstar_resize)
                logger.info("%s image resize set to %dx%d", ds_name, *ds.resize)
        # Apply OCRBench image resize if configured
        if getattr(args, "ocrbench_resize", None) is not None:
            if hasattr(ds, "resize"):
                ds.resize = tuple(args.ocrbench_resize)
                logger.info("%s image resize set to %dx%d", ds_name, *ds.resize)
        if args.summary:
            items = ds.sample(n=ds.default_sample_size)
            logger.info("Using summary sample: %s (%d items)", ds_name, len(items))
        else:
            items = ds.load()
            logger.info("Using full dataset: %s (%d items)", ds_name, len(items))

        jsonl_path = output_dir / f"{ds_name}_eval.jsonl"
        total = evaluator.evaluate(ds_name, items, jsonl_path, is_summary=args.summary)
        logger.info("Dataset %s: %d runs -> %s", ds_name, total, jsonl_path)

    logger.info("All datasets complete.")


def _cmd_score(args):
    jsonl_paths = [Path(p) for p in args.jsonl]
    labels = args.labels or []

    for i, jsonl_path in enumerate(jsonl_paths):
        if not jsonl_path.exists():
            logger.error("JSONL not found: %s", jsonl_path)
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else jsonl_path.parent
        output_path = output_dir / f"{jsonl_path.stem}_scored.jsonl"

        scorer = BenchmarkScorer()
        report = scorer.score_jsonl(jsonl_path, output_path)

        label = labels[i] if i < len(labels) else jsonl_path.stem
        logger.info("Scored: %s -> %s", label, output_path)

        # Only print summary for the last file (or all files if single)
        if len(jsonl_paths) == 1 or i == len(jsonl_paths) - 1:
            _print_score_summary(report)


def _print_score_summary(report):
    print("\n" + "=" * 60)
    print("  SCORING COMPLETE")
    print("=" * 60)
    for s in report.scores:
        print(f"\n  [{s.dataset}] {s.model_type}")
        print(f"    Items: {s.total_items}  Correct: {s.correct_items}  Accuracy: {s.accuracy:.4f}")
        print(f"    Errors: {s.error_count}  Avg Latency: {s.avg_latency_ms:.1f} ms")
        if s.avg_ttft_ms:
            print(f"    Avg TTFT: {s.avg_ttft_ms:.1f} ms  Avg TPS: {s.avg_tps or 0:.1f}")
        if s.breakdown:
            for cat, bd in sorted(s.breakdown.items()):
                print(f"      {cat}: {bd['accuracy']:.4f}")


def _cmd_report(args):
    jsonl_paths = [Path(p) for p in args.jsonl]
    labels = args.labels or []

    for jp in jsonl_paths:
        if not jp.exists():
            logger.error("JSONL not found: %s", jp)
            sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif len(jsonl_paths) == 1:
        output_dir = jsonl_paths[0].parent / "report"
    else:
        output_dir = jsonl_paths[0].parent / "report"

    report_path = generate_benchmark_report(jsonl_paths, output_dir, labels=labels or None)
    logger.info("Report generated at %s", report_path)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
