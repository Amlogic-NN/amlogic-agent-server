# -*- coding: utf-8 -*-
"""Benchmark Report — generates Markdown report with charts from scored JSONL.

Reuses chart infrastructure patterns from tools/report.py.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from .schema import BenchmarkRunRecord, BenchmarkScoreReport
from .scorer import BenchmarkScorer

logger = __import__("logging").getLogger("benchmark.report")


# ---------------------------------------------------------------------------
# Font setup (shared with tools/report.py)
# ---------------------------------------------------------------------------

def _find_cjk_font() -> Optional[str]:
    candidates = []
    for f in fm.fontManager.ttflist:
        name = f.name.lower()
        if any(k in name for k in ("cjk", "noto sans", "wenquanyi", "simhei", "simsun",
                                     "microsoft yahei", "pingfang", "hiragino", "source han",
                                     "droid sans fallback", "wqy")):
            candidates.append(f.fname)
    if candidates:
        return candidates[0]
    common = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for p in common:
        if Path(p).exists():
            return p
    return None


_CJK_FONT_PROP = None


def _get_cjk_prop():
    global _CJK_FONT_PROP
    if _CJK_FONT_PROP is None:
        font_path = _find_cjk_font()
        _CJK_FONT_PROP = fm.FontProperties(fname=font_path) if font_path else fm.FontProperties()
    return _CJK_FONT_PROP


def _set_style():
    avail = plt.style.available
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in avail else "ggplot")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_benchmark_report(
    jsonl_paths: list[Path],
    output_dir: Path,
    report_name: str = "benchmark_report.md",
    labels: Optional[list[str]] = None,
) -> Path:
    """Generate a full benchmark report from one or more scored JSONL files.

    When multiple JSONL files are provided (each representing the same dataset
    evaluated with a different model/platform), the report produces comparison
    charts and a side-by-side accuracy table.

    Args:
        jsonl_paths: One or more paths to scored JSONL files.
        output_dir: Directory for report + charts.
        report_name: Markdown report filename.
        labels: Optional human-readable labels for each JSONL file.
                Defaults to the model_type from the scored JSONL.

    Returns:
        Path to the generated Markdown report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Build combined report from all JSONL files
    combined, label_map = _build_combined_report(jsonl_paths, labels)

    _set_style()
    font_prop = _get_cjk_prop()

    # Charts (use combined report + list of individual records)
    chart_paths: dict[str, str] = {}
    chart_paths["accuracy_comparison"] = _chart_accuracy_comparison(combined, charts_dir, font_prop)
    chart_paths["per_dataset_breakdown"] = _chart_per_dataset_breakdown(combined, charts_dir, font_prop)
    chart_paths["side_by_side"] = _chart_side_by_side_accuracy(combined, charts_dir, font_prop, label_map)
    # Latency / TTFT-TPS can use all raw records from all files
    all_records = _load_all_records(jsonl_paths)
    chart_paths["latency_distribution"] = _chart_latency_distribution(all_records, charts_dir, font_prop, label_map)
    chart_paths["ttft_tps"] = _chart_ttft_tps(all_records, charts_dir, font_prop, label_map)

    # BFCL failure analysis (stacked bar per model showing failure reasons)
    bfcl_chart = _chart_bfcl_failure_analysis(jsonl_paths, charts_dir, font_prop, labels, label_map)
    if bfcl_chart:
        chart_paths["bfcl_failure"] = bfcl_chart

    # Markdown
    report_path = output_dir / report_name
    with report_path.open("w", encoding="utf-8") as f:
        f.write(_build_markdown(combined, chart_paths, output_dir, label_map, jsonl_paths))
    plt.close("all")
    return report_path


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _chart_accuracy_comparison(report: BenchmarkScoreReport, charts_dir: Path, font_prop) -> str:
    """Bar chart: accuracy per dataset per model."""
    datasets = report.datasets
    models = report.model_types
    if not datasets or not models:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)
    colors = plt.cm.Set2.colors[:max(len(models), 2)]

    for i, model in enumerate(models):
        accs = []
        for ds in datasets:
            s = report.get_score(ds, model)
            accs.append(s.accuracy if s else 0.0)
        display = _label_for(model, None)  # Use cleaned model name
        bars = ax.bar(x + i * width, accs, width, label=display, color=colors[i % len(colors)])
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontproperties=font_prop)

    ax.set_ylabel("Accuracy", fontproperties=font_prop)
    ax.set_title("Benchmark Accuracy Comparison", fontproperties=font_prop, fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, fontproperties=font_prop)
    ax.legend(prop=font_prop)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    path = str(charts_dir / "accuracy_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _chart_per_dataset_breakdown(report: BenchmarkScoreReport, charts_dir: Path, font_prop) -> str:
    """Grouped bar chart: per-category/subject accuracy for each dataset."""
    datasets = report.datasets
    if not datasets:
        return ""

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        # Get first model's breakdown
        first_model = report.model_types[0] if report.model_types else None
        if first_model is None:
            continue
        s = report.get_score(ds_name, first_model)
        if s is None or not s.breakdown:
            ax.set_title(f"{ds_name} (no data)", fontproperties=font_prop)
            continue

        cats = sorted(s.breakdown.keys())
        accs = [s.breakdown[c]["accuracy"] for c in cats]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cats)))
        bars = ax.bar(range(len(cats)), accs, color=colors)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{ds_name} Breakdown", fontproperties=font_prop, fontsize=12)
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = str(charts_dir / "per_dataset_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _chart_latency_distribution(records: list[BenchmarkRunRecord], charts_dir: Path,
                               font_prop, label_map: Optional[dict[str, str]] = None) -> str:
    """Boxplot: latency distribution per model."""
    if not records:
        return ""

    by_model = defaultdict(list)
    for r in records:
        if r.latency_ms > 0:
            by_model[r.model_type].append(r.latency_ms)

    if not by_model:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    models = sorted(by_model.keys())
    display_names = [_label_for(model, label_map) for model in models]
    data = [by_model[m] for m in models]
    bp = ax.boxplot(data, tick_labels=display_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.Set2.colors[:len(models)]):
        patch.set_facecolor(color)
    ax.set_ylabel("Latency (ms)", fontproperties=font_prop)
    ax.set_title("Latency Distribution per Model", fontproperties=font_prop, fontsize=14)
    fig.tight_layout()
    path = str(charts_dir / "latency_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _chart_ttft_tps(records: list[BenchmarkRunRecord], charts_dir: Path,
                    font_prop, label_map: Optional[dict[str, str]] = None) -> str:
    """Scatter plot: TTFT vs TPS for ADLA models (where available)."""
    ttft_tps = [(r.ttft_ms, r.tps, r.model_type) for r in records
                if r.ttft_ms is not None and r.tps is not None and r.ttft_ms > 0 and r.tps > 0]
    if not ttft_tps:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    by_model = defaultdict(lambda: {"ttft": [], "tps": []})
    for ttft, tps, model in ttft_tps:
        by_model[model]["ttft"].append(ttft)
        by_model[model]["tps"].append(tps)

    colors = plt.cm.Set2.colors[:max(len(by_model), 2)]
    for (model, data), color in zip(by_model.items(), colors):
        label = _label_for(model, label_map)
        ax.scatter(data["ttft"], data["tps"], label=label, color=color, alpha=0.6, s=30)

    ax.set_xlabel("TTFT (ms)", fontproperties=font_prop)
    ax.set_ylabel("TPS (tokens/sec)", fontproperties=font_prop)
    ax.set_title("ADLA Performance: TTFT vs TPS", fontproperties=font_prop, fontsize=14)
    ax.legend(prop=font_prop)
    fig.tight_layout()
    path = str(charts_dir / "ttft_tps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------

def _build_markdown(report: BenchmarkScoreReport, chart_paths: dict[str, str],
                    base_dir: Path, label_map: Optional[dict[str, str]] = None,
                    jsonl_paths: Optional[list[Path]] = None) -> str:
    lines = [
        "# Agent Capability Benchmark Report",
        "",
        "---",
        "",
        "## 1. Overall Accuracy",
        "",
        "| Dataset | Model | Total | Correct | Accuracy | Avg Lat (ms) | Avg TTFT (ms) | Avg TPS |",
        "|---------|-------|-------|---------|----------|-------------|---------------|---------|",
    ]
    for s in report.scores:
        display = _label_for(s.model_type, label_map)
        lines.append(
            f"| {s.dataset} | {display} | {s.total_items} | {s.correct_items} "
            f"| {s.accuracy:.4f} | {s.avg_latency_ms:.1f} "
            f"| {s.avg_ttft_ms or 'N/A'} | {s.avg_tps or 'N/A'} |"
        )
    lines.append("")

    lines.extend([
        "---", "",
        "## 2. Accuracy Comparison", "",
        f"![Accuracy Comparison](charts/accuracy_comparison.png)", "",
        "---", "",
        "## 3. Per-Dataset Breakdown", "",
        f"![Per-Dataset Breakdown](charts/per_dataset_breakdown.png)", "",
        "---", "",
        "## 4. Latency Distribution", "",
        f"![Latency Distribution](charts/latency_distribution.png)", "",
    ])

    if chart_paths.get("ttft_tps"):
        lines.extend([
            "---", "",
            "## 5. ADLA Performance (TTFT vs TPS)", "",
            f"![TTFT vs TPS](charts/ttft_tps.png)", "",
        ])

    lines.extend([
        "---", "",
        "## 6. Side-by-Side Accuracy Comparison by Dataset", "",
        f"![Side-by-Side Accuracy](charts/side_by_side.png)", "",
    ])

    # BFCL failure analysis
    if chart_paths.get("bfcl_failure") and jsonl_paths:
        lines.extend([
            "---", "",
            "## 7. BFCL Failure Analysis", "",
            "The chart below shows, for each model's BFCL dataset, the breakdown by category "
            "of correct calls, wrong tool name, wrong arguments, and inference errors.", "",
            f"![BFCL Failure Analysis](charts/bfcl_failure_analysis.png)", "",
        ])
        lines.extend(_bfcl_failure_tables(jsonl_paths, label_map))
    lines.extend([
        "---", "",
        "## 8. Detailed Breakdowns", "",
    ])
    for s in report.scores:
        if not s.breakdown:
            continue
        display = _label_for(s.model_type, label_map)
        lines.extend([
            f"### {s.dataset} — {display}", "",
            "| Category | Total | Correct | Accuracy |",
            "|----------|-------|---------|----------|",
        ])
        for cat, bd in sorted(s.breakdown.items()):
            lines.append(f"| {cat} | {bd['total']} | {bd['correct']} | {bd['accuracy']:.4f} |")
        lines.append("")

    return "\n".join(lines)


def _chart_side_by_side_accuracy(report: BenchmarkScoreReport, charts_dir: Path,
                                 font_prop, label_map: Optional[dict[str, str]] = None) -> str:
    """Grouped bar chart: accuracy per model per dataset, side by side."""
    datasets = report.datasets
    models = report.model_types
    if not datasets or not models:
        return ""

    n_datasets = len(datasets)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(8, n_datasets * 2.5), 6))
    x = np.arange(n_datasets)
    width = 0.8 / max(n_models, 1)
    colors = plt.cm.tab10.colors[:max(n_models, 10)]

    for i, model in enumerate(models):
        accs = []
        for ds in datasets:
            s = report.get_score(ds, model)
            accs.append(s.accuracy if s else 0.0)
        display = _label_for(model, label_map)
        bars = ax.bar(x + i * width, accs, width, label=display, color=colors[i % len(colors)])
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontproperties=font_prop)

    ax.set_ylabel("Accuracy", fontproperties=font_prop)
    ax.set_title("Side-by-Side Accuracy Comparison", fontproperties=font_prop, fontsize=14)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets, fontproperties=font_prop, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(prop=font_prop, fontsize=9)
    fig.tight_layout()
    path = str(charts_dir / "side_by_side.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _chart_bfcl_failure_analysis(
    jsonl_paths: list[Path],
    charts_dir: Path,
    font_prop,
    cli_labels: Optional[list[str]] = None,
    label_map: Optional[dict[str, str]] = None,
) -> str:
    """Stacked bar chart: BFCL per-category failure breakdown per model.

    Each model gets a group of bars (one per BFCL category) stacked into:
    correct (green), wrong args (orange), wrong tool (red), error (grey).
    Uses CLI labels when available, falling back to label_map then model_type.
    """
    all_data: list[tuple[str, str, dict]] = []  # (display_label, cat, counts)

    for i, jp in enumerate(jsonl_paths):
        records = _load_records(jp)
        bfcl_recs = [r for r in records
                     if r.dataset.lower().strip() in ("bfcl",) and r.score_detail]
        if not bfcl_recs:
            continue

        # Determine display label: CLI labels > label_map > raw model_type
        if cli_labels and i < len(cli_labels) and cli_labels[i]:
            display = cli_labels[i]
        else:
            scores = BenchmarkScorer()._build_report(bfcl_recs)
            raw_model = scores.scores[0].model_type if scores.scores else f"file_{i}"
            display = _label_for(raw_model, label_map)

        # Categorize by category
        by_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0,
            "wrong_tool": 0, "wrong_args": 0, "error": 0, "other": 0})
        for r in bfcl_recs:
            cat = (r.score_detail or {}).get("category", "unknown")
            by_cat[cat]["total"] += 1
            if r.is_correct is True:
                by_cat[cat]["correct"] += 1
            elif r.error:
                by_cat[cat]["error"] += 1
            elif not r.name_matched:
                by_cat[cat]["wrong_tool"] += 1
            elif r.is_correct is False:
                by_cat[cat]["wrong_args"] += 1
            else:
                by_cat[cat]["other"] += 1

        for cat, counts in sorted(by_cat.items()):
            all_data.append((display, cat, counts))

    if not all_data:
        return ""

    # Determine layout
    models = sorted(set(d[0] for d in all_data))
    cats_per_model = defaultdict(set)
    for m, c, _ in all_data:
        cats_per_model[m].add(c)
    all_cats = sorted(set(d[1] for d in all_data))

    n_models = len(models)
    n_cats = len(all_cats)
    if n_cats == 0:
        return ""

    fig, ax = plt.subplots(figsize=(max(8, n_models * n_cats * 2.5), 6))
    x_positions = np.arange(n_models)
    bar_width = 0.7 / max(n_cats, 1)

    # Colors: green=correct, orange=wrong_args, red=wrong_tool, grey=error
    stack_colors = {"correct": "#2ca02c", "wrong_args": "#ff7f0e", "wrong_tool": "#d62728", "error": "#7f7f7f", "other": "#bcbd22"}
    stack_keys = ["correct", "wrong_args", "wrong_tool", "error", "other"]
    stack_labels = {"correct": "Correct", "wrong_args": "Wrong Args", "wrong_tool": "Wrong Tool", "error": "Error", "other": "Other"}

    # Build data matrix: [model_idx][cat_idx] = {counts dict}
    data_matrix: dict[str, dict[str, dict]] = {}
    for m, c, counts in all_data:
        data_matrix.setdefault(m, {})[c] = counts

    for ci, cat in enumerate(all_cats):
        bottom_vals = np.zeros(n_models)
        for key in stack_keys:
            vals = []
            for mi, model in enumerate(models):
                counts = data_matrix.get(model, {}).get(cat, {})
                total = counts.get("total", 0)
                val = counts.get(key, 0)
                vals.append(val / total * 100 if total > 0 else 0)
            vals_arr = np.array(vals)
            label = stack_labels[key] if ci == 0 else None  # Only label first category in legend
            ax.bar(x_positions + ci * bar_width, vals_arr, bar_width * 0.8,
                   bottom=bottom_vals, color=stack_colors[key], label=label)
            bottom_vals += vals_arr

    # X-axis labels: model names with category labels
    tick_positions = x_positions + (n_cats - 1) * bar_width / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(models, fontproperties=font_prop, fontsize=9)
    ax.set_ylabel("% of Cases", fontproperties=font_prop)
    ax.set_title("BFCL Failure Analysis by Category", fontproperties=font_prop, fontsize=14)

    # Add category legend at top
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=stack_colors[k], label=stack_labels[k]) for k in stack_keys]
    ax.legend(handles=legend_patches, prop=font_prop, fontsize=8, loc="upper right")

    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = str(charts_dir / "bfcl_failure_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _bfcl_failure_tables(
    jsonl_paths: list[Path],
    label_map: Optional[dict[str, str]] = None,
) -> list[str]:
    """Generate markdown tables for BFCL failure breakdown per model."""
    lines: list[str] = []

    for i, jp in enumerate(jsonl_paths):
        records = _load_records(jp)
        bfcl_recs = [r for r in records
                     if r.dataset.lower().strip() in ("bfcl",) and r.score_detail]
        if not bfcl_recs:
            continue

        scores = BenchmarkScorer()._build_report(bfcl_recs)
        # Build unique key the same way _build_combined_report does
        raw_model = scores.scores[0].model_type if scores.scores else f"file_{i}"
        unique_key = f"{raw_model}__file_{i}"
        display = _label_for(unique_key, label_map)

        by_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0,
            "wrong_tool": 0, "wrong_args": 0, "error": 0})
        for r in bfcl_recs:
            cat = (r.score_detail or {}).get("category", "unknown")
            by_cat[cat]["total"] += 1
            if r.is_correct is True:
                by_cat[cat]["correct"] += 1
            elif r.error:
                by_cat[cat]["error"] += 1
            elif not r.name_matched:
                by_cat[cat]["wrong_tool"] += 1
            elif r.is_correct is False:
                by_cat[cat]["wrong_args"] += 1

        lines.append(f"### BFCL — {display}")
        lines.append("")
        lines.append("| Category | Total | Correct | Wrong Tool | Wrong Args | Error | Accuracy |")
        lines.append("|----------|-------|---------|------------|------------|-------|----------|")

        for cat in sorted(by_cat):
            d = by_cat[cat]
            total = d["total"]
            lines.append(
                f"| {cat} | {total} | {d['correct']} "
                f"| {d['wrong_tool']} ({_pct(d['wrong_tool'], total)}) "
                f"| {d['wrong_args']} ({_pct(d['wrong_args'], total)}) "
                f"| {d['error']} ({_pct(d['error'], total)}) "
                f"| {d['correct'] / total:.4f} |"
            )
        lines.append("")

    return lines


def _pct(count: int, total: int) -> str:
    """Format count as percentage string."""
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def _label_for(model_type: str, label_map: Optional[dict[str, str]] = None) -> str:
    """Return display label for a model type, falling back to model_type."""
    if label_map and model_type in label_map:
        return label_map[model_type]
    # Strip internal __file_N suffix for display when no label is set
    import re
    cleaned = re.sub(r'__file_\d+$', '', model_type)
    return cleaned

def _build_combined_report(
    jsonl_paths: list[Path],
    labels: Optional[list[str]] = None,
) -> tuple[BenchmarkScoreReport, dict[str, str]]:
    """Build a single BenchmarkScoreReport from multiple scored JSONL files.

    Each JSONL is scored independently; scores are merged so that
    different model_types appear side-by-side in the same report.

    Returns:
        (combined_report, label_map) where label_map is {model_type: display_label}
    """
    scorer = BenchmarkScorer()
    all_scores: list[DatasetScore] = []
    label_map: dict[str, str] = {}

    for i, jp in enumerate(jsonl_paths):
        if not jp.exists():
            logger.warning("JSONL not found, skipping: %s", jp)
            continue

        # Load records directly and build report (avoid side-effect output from score_jsonl)
        records = scorer._load_records(jp)
        if not records:
            continue

        # Score records if not already scored
        for rec in records:
            if rec.is_correct is None:
                scorer._score_one(rec)

        report = scorer._build_report(records)

        # Determine label: use explicit label, otherwise leave as None (use original model_type)
        has_explicit_label = labels and i < len(labels) and bool(labels[i])
        file_label = labels[i] if has_explicit_label else None

        for s in report.scores:
            # Create a unique model_type key per file to avoid label overwriting
            unique_key = f"{s.model_type}__file_{i}"
            if file_label:
                # Only store in label_map when there's an explicit label;
                # otherwise _label_for will fall back to the original model_type.
                label_map[unique_key] = file_label
            # Override model_type with unique key for correct display
            s.model_type = unique_key
            all_scores.append(s)

    combined = BenchmarkScoreReport(scores=all_scores)
    return combined, label_map


def _load_all_records(jsonl_paths: list[Path]) -> list[BenchmarkRunRecord]:
    """Load records from multiple JSONL files."""
    all_records = []
    for jp in jsonl_paths:
        all_records.extend(_load_records(jp))
    return all_records


def _load_records(jsonl_path: Path) -> list[BenchmarkRunRecord]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(BenchmarkRunRecord(**json.loads(line)))
            except Exception:
                pass
    return records
