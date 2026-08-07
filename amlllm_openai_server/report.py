# -*- coding: utf-8 -*-
"""Report generation — produces a Markdown report with embedded matplotlib charts
from tool-calling evaluation results.
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

from .schema import RunRecord, StatsResult
from .metrics import compile_all_stats


# ---------------------------------------------------------------------------
# Font setup
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
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
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
        if font_path:
            _CJK_FONT_PROP = fm.FontProperties(fname=font_path)
        else:
            _CJK_FONT_PROP = fm.FontProperties()
    return _CJK_FONT_PROP

def _set_style():
    avail = plt.style.available
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in avail else "ggplot")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    jsonl_path: Path,
    output_dir: Path,
    report_name: str = "report.md",
    exclude_latency: Optional[list[str]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(RunRecord(**json.loads(line)))
    if not records:
        raise ValueError("No records found in JSONL file")

    stats = compile_all_stats(records)
    model_types = sorted(stats.keys())
    _set_style()
    font_prop = _get_cjk_prop()

    chart_paths: dict[str, str] = {}
    chart_paths["metrics_comparison"] = _chart_metrics_comparison(stats, charts_dir, font_prop)
    for mt in model_types:
        chart_paths[f"confusion_{mt}"] = _chart_confusion_matrix(stats[mt], charts_dir, font_prop)
    chart_paths["name_accuracy"] = _chart_name_accuracy(stats, charts_dir, font_prop)
    chart_paths["entry_success_rate"] = _chart_entry_success_rate(records, charts_dir, font_prop)
    chart_paths["latency_boxplot"] = _chart_latency_boxplot(records, charts_dir, font_prop)
    chart_paths["error_breakdown"] = _chart_error_breakdown(records, charts_dir, font_prop)
    if len(model_types) > 1:
        chart_paths["latency_comparison"] = _chart_latency_comparison(
            records, model_types, charts_dir, font_prop, exclude_latency=exclude_latency,
        )
        chart_paths["tool_call_count"] = _chart_tool_call_count(records, model_types, charts_dir, font_prop)

    report_path = output_dir / report_name
    with report_path.open("w", encoding="utf-8") as f:
        f.write(_build_markdown(stats, chart_paths))
    plt.close("all")
    return report_path


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------

def _rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base))

def _build_markdown(stats: dict[str, StatsResult], chart_paths: dict[str, str]) -> str:
    report_dir = Path(chart_paths.get("metrics_comparison", ".")).parent.parent
    lines = [
        "# Tool-Calling Evaluation Report",
        "",
        "---",
        "",
        "## 1. Summary Statistics",
        "",
        "| Model | Total | Labeled | TP | FP | FN | TN | Recall | Prec | Acc | F1 | NameAcc | VerifPass | Errors | Avg Lat(ms) |",
        "|-------|-------|---------|----|----|----|----|--------|------|-----|----|--------|-----------|--------|-------------|",
    ]
    for mt, s in stats.items():
        d = s.to_dict()
        lines.append(
            f"| {mt} | {d['total_runs']} | {d['total_labeled']} "
            f"| {d['TP']} | {d['FP']} | {d['FN']} | {d['TN']} "
            f"| {d['Recall']:.4f} | {d['Precision']:.4f} | {d['Accuracy']:.4f} "
            f"| {d['F1']:.4f} | {d['NameAccuracy']:.4f} | {d['VerifierPassRate']:.4f} "
            f"| {d['error_count']} | {d['avg_latency_ms']} |"
        )
    lines.append("")

    lines.extend([
        "---", "",
        "## 2. Metrics Comparison", "",
        f"![Metrics Comparison]({_rel(Path(chart_paths['metrics_comparison']), report_dir)})", "",
        "---", "",
        "## 3. Confusion Matrices", "",
    ])
    for mt in stats:
        lines.extend([
            f"### {mt}", "",
            f"![Confusion Matrix ({mt})]({_rel(Path(chart_paths[f'confusion_{mt}']), report_dir)})", "",
        ])

    lines.extend([
        "---", "",
        "## 4. Tool Name Accuracy", "",
        f"![Name Accuracy]({_rel(Path(chart_paths['name_accuracy']), report_dir)})", "",
        "---", "",
        "## 5. Per-Entry Success Rate", "",
        f"![Entry Success Rate]({_rel(Path(chart_paths['entry_success_rate']), report_dir)})", "",
        "---", "",
        "## 6. Latency Distribution", "",
        f"![Latency Boxplot]({_rel(Path(chart_paths['latency_boxplot']), report_dir)})", "",
    ])
    if "latency_comparison" in chart_paths:
        lines.extend([
            "", "### Latency Comparison Across Models", "",
            f"![Latency Comparison]({_rel(Path(chart_paths['latency_comparison']), report_dir)})", "",
        ])
    lines.extend([
        "---", "",
        "## 7. Error Breakdown", "",
        f"![Error Breakdown]({_rel(Path(chart_paths['error_breakdown']), report_dir)})", "",
    ])
    if "tool_call_count" in chart_paths:
        lines.extend([
            "", "### Tool Call Count Distribution", "",
            f"![Tool Call Count]({_rel(Path(chart_paths['tool_call_count']), report_dir)})", "",
        ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _chart_metrics_comparison(stats, charts_dir, font_prop) -> str:
    model_types = sorted(stats.keys())
    metric_names = ["Recall", "Precision", "Accuracy", "F1", "NameAcc", "VerifPass"]
    x = list(range(len(metric_names)))
    n_models = len(model_types)
    bar_width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2.colors[:max(n_models, 2)]
    for i, mt in enumerate(model_types):
        s = stats[mt]
        values = [s.recall, s.precision, s.accuracy, s.f1, s.name_accuracy, s.verifier_pass_rate]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar([v + offset for v in x], values, bar_width, label=mt, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontproperties=font_prop)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontproperties=font_prop)
    ax.set_title("Tool-Calling Metrics Comparison", fontproperties=font_prop, fontsize=14)
    ax.legend(prop=font_prop)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = charts_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_confusion_matrix(stat, charts_dir, font_prop) -> str:
    matrix = [[stat.tn, stat.fp], [stat.fn, stat.tp]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = max(1, max(max(r) for r in matrix))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: No Tool", "Pred: Has Tool"], fontproperties=font_prop)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual: No Tool", "Actual: Has Tool"], fontproperties=font_prop)
    ax.set_title(f"Confusion Matrix — {stat.model_type}", fontproperties=font_prop, fontsize=13)
    for i in range(2):
        for j in range(2):
            val = matrix[i][j]
            color = "white" if val > vmax / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{val}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    path = charts_dir / f"confusion_{stat.model_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_name_accuracy(stats, charts_dir, font_prop) -> str:
    model_types = sorted(stats.keys())
    categories = ["Name Match", "Name Mismatch"]
    x = list(range(len(model_types)))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(model_types) * 2), 5))
    for i, cat in enumerate(categories):
        values = []
        for mt in model_types:
            s = stats[mt]
            values.append(s.name_match_count if cat == "Name Match" else s.name_mismatch_count)
        offset = (i - 0.5) * bar_width
        ax.bar([v + offset for v in x], values, bar_width, label=cat,
               color=plt.cm.Set2.colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_types, fontproperties=font_prop)
    ax.set_ylabel("Count", fontproperties=font_prop)
    ax.set_title("Tool Name Match vs Mismatch", fontproperties=font_prop, fontsize=14)
    ax.legend(prop=font_prop)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = charts_dir / "name_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_entry_success_rate(records, charts_dir, font_prop) -> str:
    model_entries = defaultdict(lambda: defaultdict(list))
    for r in records:
        model_entries[r.model_type][(r.entry_index, r.conv_index)].append(r)
    all_keys = set()
    for entries in model_entries.values():
        all_keys.update(entries.keys())
    sorted_keys = sorted(all_keys)
    model_types = sorted(model_entries.keys())
    n_keys = len(sorted_keys)
    n_models = len(model_types)
    bar_width = 0.8 / max(n_models, 1)
    x = list(range(n_keys))
    fig, ax = plt.subplots(figsize=(max(8, n_keys * 1.2), 5))
    colors = plt.cm.Set2.colors[:max(n_models, 2)]
    for i, mt in enumerate(model_types):
        entries = model_entries[mt]
        rates = []
        for key in sorted_keys:
            recs = entries.get(key, [])
            if not recs:
                rates.append(0.0)
            else:
                success = sum(1 for r in recs
                              if r.name_matched or (r.expected_tool_name is None and not r.actual_has_tool_calls))
                rates.append(success / len(recs))
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar([v + offset for v in x], rates, bar_width, label=mt, color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{k[0]}.C{k[1]}" for k in sorted_keys], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Success Rate", fontproperties=font_prop)
    ax.set_title("Per-Entry Tool Call Success Rate", fontproperties=font_prop, fontsize=14)
    ax.legend(prop=font_prop)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = charts_dir / "entry_success_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_latency_boxplot(records, charts_dir, font_prop) -> str:
    groups = defaultdict(list)
    for r in records:
        label = f"{r.model_type}\nhas_tool={r.actual_has_tool_calls}"
        groups[label].append(r.latency_ms)
    labels = sorted(groups.keys())
    data = [groups[l] for l in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.5), 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True, meanline=True)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(plt.cm.Set2.colors[0] if "local" in label else plt.cm.Set2.colors[1])
    ax.set_ylabel("Latency (ms)", fontproperties=font_prop)
    ax.set_title("Latency Distribution by Model & Tool Call Presence", fontproperties=font_prop, fontsize=14)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(font_prop)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = charts_dir / "latency_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_error_breakdown(records, charts_dir, font_prop) -> str:
    from collections import Counter
    model_errors: dict[str, Counter] = {}
    for r in records:
        if r.error:
            model_errors.setdefault(r.model_type, Counter())
            short = r.error[:60] + ("..." if len(r.error) > 60 else "")
            model_errors[r.model_type][short] += 1
    if not model_errors:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No errors recorded", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Error Breakdown", fontproperties=font_prop)
        fig.tight_layout()
        path = charts_dir / "error_breakdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)
    model_types = sorted(model_errors.keys())
    all_error_types: set[str] = set()
    for c in model_errors.values():
        all_error_types.update(c.keys())
    sorted_errors = sorted(all_error_types)
    fig, ax = plt.subplots(figsize=(max(8, len(sorted_errors) * 1.5), 5))
    colors = plt.cm.Set2.colors[:max(len(model_types), 2)]
    bar_width = 0.8 / max(len(model_types), 1)
    x = list(range(len(sorted_errors)))
    for i, mt in enumerate(model_types):
        counts = [model_errors[mt].get(e, 0) for e in sorted_errors]
        offset = (i - (len(model_types) - 1) / 2) * bar_width
        ax.bar([v + offset for v in x], counts, bar_width, label=mt, color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_errors, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Count", fontproperties=font_prop)
    ax.set_title("Error Breakdown by Model", fontproperties=font_prop, fontsize=14)
    ax.legend(prop=font_prop)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = charts_dir / "error_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_latency_comparison(
    records: list[RunRecord],
    model_types: list[str],
    charts_dir: Path,
    font_prop,
    exclude_latency: Optional[list[str]] = None,
) -> str:
    """Stacked bar chart: prefill_ms (bottom) + decode_ms (top) per model per conversation.

    Models listed in *exclude_latency* are omitted (e.g. GGUF which lacks prefill/decode data).
    """
    import numpy as np

    exclude_set = set(exclude_latency or [])
    visible_models = [mt for mt in model_types if mt not in exclude_set]
    if not visible_models:
        # No data to plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "All models excluded from latency comparison.", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Latency Comparison (Prefill + Decode)", fontproperties=font_prop)
        fig.tight_layout()
        path = charts_dir / "latency_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # Group by (model_type, conv_index)
    prefill_data = defaultdict(lambda: defaultdict(list))
    decode_data = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.model_type in exclude_set:
            continue
        if r.prefill_ms is not None and r.decode_ms is not None:
            prefill_data[r.model_type][r.conv_index].append(r.prefill_ms)
            decode_data[r.model_type][r.conv_index].append(r.decode_ms)

    conv_indices = sorted({r.conv_index for r in records if r.model_type not in exclude_set})
    case_labels = [f"Case {ci + 1}" for ci in conv_indices]

    x = np.arange(len(conv_indices))
    n_models = len(visible_models)
    bar_width = 0.7 / n_models
    fig, ax = plt.subplots(figsize=(max(10, len(conv_indices) * 2.5), 6))

    prefill_color = "#3498DB"   # blue
    decode_color = "#E74C3C"    # red

    for i, mt in enumerate(visible_models):
        p_means = [np.mean(prefill_data[mt].get(ci, [0.0])) for ci in conv_indices]
        d_means = [np.mean(decode_data[mt].get(ci, [0.0])) for ci in conv_indices]
        offset = i * bar_width
        bars_p = ax.bar(x + offset, p_means, bar_width, label=f"{mt} (prefill)" if i == 0 else "",
                        color=prefill_color, alpha=0.85)
        bars_d = ax.bar(x + offset, d_means, bar_width, bottom=p_means,
                        label=f"{mt} (decode)" if i == 0 else "",
                        color=decode_color, alpha=0.85)
        # Prefill & Decode labels placed at 50% of bar height, side-by-side
        for ci_idx in range(len(conv_indices)):
            pval = p_means[ci_idx]
            dval = d_means[ci_idx]
            total = pval + dval
            if total > 0:
                mid_y = total / 2
                ax.text(x[ci_idx] + offset + bar_width / 2, mid_y,
                        f"P:{pval:.0f}  D:{dval:.0f}", ha="center", va="center",
                        fontsize=4.5, color="#2C3E50", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=0.75))
                ax.text(x[ci_idx] + offset + bar_width / 2,
                        total + max(p_means + d_means) * 0.01,
                        f"{total/1000:.1f}s", ha="center", va="bottom", fontsize=6,
                        fontweight="bold")

    # Add a legend entry for prefill/decode
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=prefill_color, label="Prefill"),
                       Patch(facecolor=decode_color, label="Decode")]
    ax.legend(handles=legend_elements, loc="upper right", prop=dict(size=9))

    # Model labels under each group
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(case_labels, rotation=30, ha="right", fontsize=9)
    # Add model name annotations below case labels
    for i, mt in enumerate(visible_models):
        for ci_idx in range(len(conv_indices)):
            ax.text(x[ci_idx] + i * bar_width + bar_width / 2, -max(p_means + d_means) * 0.03,
                    mt, ha="center", va="top", fontsize=5, rotation=90, color="grey", alpha=0.7)

    ax.set_ylabel("Time (ms)", fontproperties=font_prop)
    ax.set_title("Latency Comparison — Prefill + Decode", fontproperties=font_prop, fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = charts_dir / "latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_tool_call_count(
    records: list[RunRecord],
    model_types: list[str],
    charts_dir: Path,
    font_prop,
) -> str:
    """Two-part chart: (top) overall grouped bar with X=model, (bottom) per-model stacked bars by case."""

    categories = ["0 (no call)", "1 (normal)", ">1 (looping)"]
    cat_keys = ["0", "1", ">1"]
    cat_colors = {"0": "#E74C3C", "1": "#2ECC71", ">1": "#F39C12"}

    # Overall counts per model
    overall: dict[str, dict[str, int]] = {mt: {ck: 0 for ck in cat_keys} for mt in model_types}
    # Per-case counts per model
    conv_indices = sorted({r.conv_index for r in records})
    case_data: dict[str, dict[str, list[int]]] = {
        mt: {ck: [0] * len(conv_indices) for ck in cat_keys} for mt in model_types
    }
    for r in records:
        tcc = r.tool_call_count or 0
        tloop = r.error is not None and "Loop detected" in r.error
        ck = "0" if tcc == 0 else ("1" if tcc == 1 else ">1")
        if tloop:
            ck = ">1"
        overall[r.model_type][ck] += 1
        idx = conv_indices.index(r.conv_index)
        case_data[r.model_type][ck][idx] += 1

    import numpy as np
    colors = plt.cm.Set2.colors[:max(len(model_types), 2)]
    n_models = len(model_types)

    fig = plt.figure(figsize=(max(12, n_models * 5), 10))

    # ---- Top: overall grouped bar ----
    ax_top = fig.add_subplot(2, 1, 1)
    x = np.arange(len(model_types))
    bar_width = 0.25
    for i, (ck, cat_label) in enumerate(zip(cat_keys, categories)):
        values = [overall[mt][ck] for mt in model_types]
        offset = (i - 1) * bar_width
        bars = ax_top.bar(x + offset, values, bar_width, label=cat_label,
                          color=cat_colors[ck], alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            str(val), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(model_types, fontproperties=font_prop, rotation=20, ha="right")
    ax_top.set_ylabel("Count", fontproperties=font_prop)
    ax_top.set_title("Tool Call Count Distribution (Overall)", fontproperties=font_prop, fontsize=14)
    ax_top.legend(prop=font_prop, fontsize=9)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.grid(axis="y", alpha=0.3)

    # ---- Bottom: per-model stacked bars by case ----
    case_names = [f"Case {ci + 1}" for ci in conv_indices]
    stack_order = [">1", "1", "0"]
    for model_idx, mt in enumerate(model_types):
        ax = fig.add_subplot(2, n_models, n_models + model_idx + 1)
        x2 = np.arange(len(conv_indices))
        bottom = np.zeros(len(conv_indices))
        for ck in stack_order:
            vals = case_data[mt][ck]
            ax.bar(x2, vals, 0.55, bottom=bottom, label=f"{ck} call(s)",
                   color=cat_colors[ck], edgecolor="black", linewidth=0.5, alpha=0.85)
            for j, v in enumerate(vals):
                if v > 0:
                    ax.text(x2[j], bottom[j] + v / 2, str(v),
                            ha="center", va="center", fontsize=7, fontweight="bold")
            bottom += vals
        ax.set_title(mt, fontproperties=font_prop, fontsize=11, fontweight="bold",
                     color=colors[model_idx])
        ax.set_xticks(x2)
        ax.set_xticklabels(case_names, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, max(bottom) * 1.15 if any(bottom) else 22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if model_idx == 0:
            ax.set_ylabel("Count", fontproperties=font_prop)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[ck], alpha=0.85)
               for ck in stack_order]
    fig.legend(handles, [f"{ck} call(s)" for ck in stack_order],
               loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Tool Call Count Analysis", fontproperties=font_prop, fontsize=15, fontweight="bold",
                 y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = charts_dir / "tool_call_count.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)
