# -*- coding: utf-8 -*-
"""BFCL (Berkeley Function Calling Leaderboard) dataset loader.

Download from Hugging Face ``gorilla-llm/Berkeley-Function-Calling-Leaderboard``,
filter to Simple + Parallel categories for 8K context compatibility.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from .base import BenchmarkDataset

logger = logging.getLogger("benchmark.datasets.bfcl")

# All BFCL v4 categories for HuggingFace download
BFCL_CATEGORIES = ["simple", "parallel", "multiple", "parallel_multiple"]

# Default categories to load (all single-turn scoring categories).
# Excludes multi-turn (needs execution env), memory/web_search (needs state),
# and format_sensitivity (non-scoring).
BFCL_SIMPLE_CATEGORIES = ["simple", "parallel"]

# All single-turn categories available in BFCL v4 local clone.
# Map: data file prefix → category name
_BFCL_FILE_CATEGORY_MAP: dict[str, str] = {
    # ---- Simple ----
    "BFCL_v4_simple_python":     "simple",
    "BFCL_v4_simple_java":       "simple",
    "BFCL_v4_simple_javascript": "simple",
    "BFCL_v4_live_simple":       "simple",
    # ---- Parallel ----
    "BFCL_v4_parallel":              "parallel",
    "BFCL_v4_live_parallel":         "parallel",
    # ---- Multiple ----
    "BFCL_v4_multiple":              "multiple",
    "BFCL_v4_live_multiple":         "multiple",
    # ---- Parallel + Multiple ----
    "BFCL_v4_parallel_multiple":         "parallel_multiple",
    "BFCL_v4_live_parallel_multiple":    "parallel_multiple",
    # ---- Irrelevance / Relevance ----
    "BFCL_v4_irrelevance":          "irrelevance",
    "BFCL_v4_live_irrelevance":     "irrelevance",
    "BFCL_v4_live_relevance":       "relevance",
    # ---- Multi-turn (excluded by default — requires execution environment) ----
    "BFCL_v4_multi_turn_base":          "multi_turn_base",
    "BFCL_v4_multi_turn_long_context":  "multi_turn_long_context",
    "BFCL_v4_multi_turn_miss_func":     "multi_turn_miss_func",
    "BFCL_v4_multi_turn_miss_param":    "multi_turn_miss_param",
    # ---- Agentic (excluded by default — requires state / external APIs) ----
    "BFCL_v4_memory":      "memory",
    "BFCL_v4_web_search":  "web_search",
    # ---- Non-scoring ----
    "BFCL_v4_format_sensitivity": "format_sensitivity",
}

# Categories excluded by default (multi-turn, agentic, non-scoring)
_BFCL_EXCLUDED_CATEGORIES: set[str] = {
    "multi_turn_base", "multi_turn_long_context",
    "multi_turn_miss_func", "multi_turn_miss_param",
    "memory", "web_search", "format_sensitivity",
}

# All single-turn scoring categories (what we load by default)
BFCL_ALL_SINGLE_TURN = [
    "simple", "parallel", "multiple", "parallel_multiple",
    "irrelevance", "relevance",
]


class BFCLDataset(BenchmarkDataset):
    dataset_name = "bfcl"
    hf_repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    hf_config = "default"
    hf_split = "test"

    def __init__(self, cache_dir: str | Path = "dataset",
                 categories: list[str] | None = None):
        super().__init__(cache_dir)
        self._categories = categories if categories is not None else BFCL_ALL_SINGLE_TURN

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> None:
        if self.processed_path.exists():
            logger.info("BFCL already downloaded and processed at %s", self.processed_path)
            return

        all_items: list[dict] = []

        # Priority 1: Load from local GitHub clone (dataset_repos/gorilla)
        self._load_from_local_clone(all_items)

        if all_items:
            logger.info("BFCL: %d total items loaded from local clone.", len(all_items))
            self._save_processed(all_items)
            return

        # Priority 2: Fallback to HuggingFace
        logger.info("BFCL local clone unavailable. Downloading from %s...", self.hf_repo)
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "BFCL requires 'datasets' package for HuggingFace download. "
                "Install with: pip install datasets, or clone gorilla repo to dataset_repos/gorilla."
            )

        for category in BFCL_CATEGORIES:
            try:
                ds = load_dataset(self.hf_repo, category, split=self.hf_split)
                items = self._convert(ds, category)
                all_items.extend(items)
                logger.info("BFCL/%s: %d items loaded from HF.", category, len(items))
            except Exception as exc:
                logger.warning("BFCL/%s from HF failed: %s", category, exc)

        self._save_processed(all_items)
        logger.info("BFCL: %d total items processed.", len(all_items))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, force_reload: bool = False) -> list[dict]:
        if force_reload or not self.processed_path.exists():
            self.download()
        with open(self.processed_path, "r", encoding="utf-8") as f:
            all_items = json.load(f)
        # Filter to requested categories
        if self._categories:
            return [item for item in all_items if item.get("category") in self._categories]
        return all_items

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, n: int = 500, seed: int = 42) -> list[dict]:
        import json as _json

        all_items = self.load()
        sample_path = self.get_sample_path(n)

        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                indices = _json.load(f)
            logger.info("BFCL: using fixed sample of %d items.", len(indices))
            return [all_items[i] for i in indices]

        rng = random.Random(seed)
        if n >= len(all_items):
            return all_items

        # Stratify by category
        by_cat: dict[str, list[int]] = {}
        for i, item in enumerate(all_items):
            cat = item.get("category", "unknown")
            by_cat.setdefault(cat, []).append(i)

        n_per_cat = max(1, n // max(len(by_cat), 1))
        selected: list[int] = []
        for cat, idx_list in by_cat.items():
            k = min(n_per_cat, len(idx_list))
            selected.extend(rng.sample(idx_list, k))

        if len(selected) < n:
            remaining = [i for i in range(len(all_items)) if i not in set(selected)]
            extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

        selected = sorted(selected)[:n]
        with open(sample_path, "w", encoding="utf-8") as f:
            _json.dump(selected, f)
        logger.info("BFCL: sampled %d items, saved indices.", len(selected))
        return [all_items[i] for i in selected]

    # ------------------------------------------------------------------
    # Internal conversion — handles the actual BFCL JSONL format from GitHub
    # ------------------------------------------------------------------

    def _bfcl_function_to_openai_tool(self, fn: dict) -> dict:
        """Convert BFCL function dict to OpenAI tool dict with schema normalization.

        BFCL:  {"name":"...","description":"...","parameters":{"type":"dict","properties":{...}}}
        OpenAI: {"type":"function","function":{"name":"...","description":"...","parameters":{"type":"object",...}}}

        Also normalizes non-standard types: float→number, int→integer, dict→object, etc.
        """
        params = fn.get("parameters", {})
        if isinstance(params, dict):
            params = dict(params)
            if params.get("type") == "dict":
                params["type"] = "object"
            self._normalize_schema_types(params)
        return {
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": params,
            },
        }

    @staticmethod
    def _normalize_schema_types(obj) -> None:
        """Recursively normalize non-standard JSON Schema type values in-place."""
        TYPE_MAP = {
            'float': 'number', 'double': 'number',
            'int': 'integer', 'long': 'integer',
            'dict': 'object', 'HashMap': 'object',
            'list': 'array', 'ArrayList': 'array', 'Array': 'array',
            'tuple': 'array',
            'bool': 'boolean',
            'str': 'string', 'String': 'string',
            'char': 'string',
            'any': 'string',
        }
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key == 'type' and isinstance(obj[key], str):
                    if obj[key] in TYPE_MAP:
                        obj[key] = TYPE_MAP[obj[key]]
                    elif obj[key] not in {'string', 'integer', 'number', 'boolean', 'array', 'object', 'null'}:
                        obj[key] = 'string'
                else:
                    BFCLDataset._normalize_schema_types(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                BFCLDataset._normalize_schema_types(item)

    def _bfcl_ground_truth_to_expected(self, gt_list: list) -> list[dict]:
        """Convert BFCL ground_truth to list of {name, arguments, anyof_params}.

        BFCL possible_answer format:
          [{"func_name": {"param1": ["val1"], "param2": ["val1", "val2"]}}]

        In BFCL, parameter values are wrapped in arrays:
        - Single-element arrays: only one valid value, unwrap to scalar.
        - Multi-element arrays: multiple valid values (any-of), store as list.

        Target:
          [{"name": "func_name", "arguments": {"param1": "val1", "param2": ["val1", "val2"]}}]

        The scorer handles list-valued arguments as "any of these is acceptable".
        """
        result = []
        for item in gt_list:
            if not isinstance(item, dict):
                continue
            for func_name, params in item.items():
                args = {}
                if isinstance(params, dict):
                    for k, v in params.items():
                        if isinstance(v, list):
                            # Single value: unwrap. Multiple: keep as any-of list.
                            args[k] = v[0] if len(v) == 1 else v
                        else:
                            args[k] = v
                result.append({"name": func_name, "arguments": args})
        return result

    def _flatten_bfcl_question(self, question) -> list[dict]:
        """Flatten BFCL's nested question format into simple messages list.

        BFCL:    [[{role,content}], ...]
        Target:  [{role,content}, ...]
        """
        msgs = []
        if isinstance(question, list):
            for item in question:
                if isinstance(item, list):
                    for inner in item:
                        if isinstance(inner, dict) and "role" in inner:
                            msgs.append({"role": inner["role"], "content": str(inner.get("content", ""))})
                elif isinstance(item, dict) and "role" in item:
                    msgs.append({"role": item["role"], "content": str(item.get("content", ""))})
        return msgs

    def _convert(self, ds, category: str) -> list[dict]:
        """Convert BFCL JSONL data (from GitHub clone) to unified dict format.

        Handles actual BFCL keys: 'id', 'question' (nested), 'function' (singular),
        and possible_answer ground_truth format.
        """
        items = []
        for i, entry in enumerate(ds):
            eid = str(entry.get("id", f"{category}_{i}"))

            # Flatten question → messages
            messages = self._flatten_bfcl_question(entry.get("question", []))

            # Convert functions → OpenAI tools
            functions = entry.get("function", entry.get("functions", entry.get("tools", [])))
            if not isinstance(functions, list):
                functions = [functions] if functions else []
            tools = [self._bfcl_function_to_openai_tool(fn) for fn in functions]

            # Expected tool calls — may come from a separate possible_answer file;
            # if not present in entry, leave empty (will be set during scoring)
            # For irrelevance/relevance categories: empty expected_tool_calls = model should NOT call tools
            expected = []
            gt = entry.get("ground_truth", entry.get("expected_tool_calls", []))
            if isinstance(gt, list) and gt:
                # Check if it's the BFCL ground_truth format {func: {param: [val]}}
                if isinstance(gt[0], dict):
                    first_val = list(gt[0].values())[0] if gt[0] else None
                    if isinstance(first_val, dict) and all(isinstance(v, list) for v in first_val.values()):
                        expected = self._bfcl_ground_truth_to_expected(gt)
                    else:
                        expected = gt

            # Detect language from the entry ID or category
            language = "python"
            eid_lower = eid.lower()
            if "javascript" in category or "javascript" in eid_lower or "javascript" in str(entry.get("id", "")).lower():
                language = "javascript"
            elif "java" in category or "java" in eid_lower or "java" in str(entry.get("id", "")).lower():
                language = "java"

            # Ensure system message exists.
            # For irrelevance: do NOT encourage function calling (tests check that
            # model correctly identifies when functions are NOT needed).
            if not any(m["role"] == "system" for m in messages):
                if category == "irrelevance":
                    sys_content = (
                        "You are a helpful assistant. "
                        "You may be given access to functions, but only use them "
                        "if they are actually needed to answer the user's query. "
                        "If a query can be answered without function calls, "
                        "respond directly."
                    )
                else:
                    sys_content = (
                        "You are a helpful assistant with access to functions. "
                        "Use the provided functions to answer the user's query."
                    )
                messages.insert(0, {"role": "system", "content": sys_content})

            items.append({
                "item_id": f"bfcl_{category}_{i:05d}",
                "category": category,
                "language": language,
                "messages": messages,
                "tools": tools,
                "expected_tool_calls": expected,
            })

        return items

    def _normalize_tools(self, functions: Any) -> list[dict]:
        """Normalize various BFCL function formats to OpenAI tool format.

        Also fixes BFCL's 'type: dict' → 'type: object' for JSON Schema compliance.
        """
        if not functions:
            return []
        if not isinstance(functions, list):
            return []

        tools = []
        for fn in functions:
            if isinstance(fn, dict):
                # Already in OpenAI format?
                if "type" in fn and "function" in fn:
                    tools.append(fn)
                elif "name" in fn:
                    params = fn.get("parameters", fn.get("arguments", {}))
                    if isinstance(params, dict):
                        params = dict(params)
                        if params.get("type") == "dict":
                            params["type"] = "object"
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "description": fn.get("description", ""),
                            "parameters": params,
                        },
                    })
                else:
                    # Minimal fallback
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": str(fn),
                            "description": "",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    })
        return tools

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_processed(self, items: list[dict]) -> None:
        # Ensure all tool schemas are normalized before saving
        for item in items:
            for tool in item.get("tools", []):
                params = tool.get("function", {}).get("parameters", {})
                self._normalize_schema_types(params)
        with open(self.processed_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _load_from_local_clone(self, all_items: list[dict]) -> None:
        """Load BFCL data from the local GitHub clone (dataset_repos/gorilla).

        Auto-discovers all BFCL_v4_*.json files in the data directory and maps
        them to categories via _BFCL_FILE_CATEGORY_MAP. Loads possible_answer
        ground truth from parallel JSONL files.
        """
        clone_dir = Path(self.cache_dir).parent.parent / "dataset_repos" / "gorilla"
        bfcl_data_dir = clone_dir / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"

        if not bfcl_data_dir.exists():
            logger.warning("BFCL local clone not found at %s. Run scripts/extract_bfcl.py first.", bfcl_data_dir)
            return

        # Discover all BFCL v4 data files
        data_files = sorted(bfcl_data_dir.glob("BFCL_v4_*.json"))
        if not data_files:
            logger.warning("No BFCL v4 data files found in %s", bfcl_data_dir)
            return

        loaded_files = 0
        skipped_cats: set[str] = set()

        for data_path in data_files:
            data_name = data_path.stem  # e.g. "BFCL_v4_simple_python"

            # Map file to category
            category = _BFCL_FILE_CATEGORY_MAP.get(data_name)
            if category is None:
                logger.debug("BFCL: unknown file %s, skipping", data_name)
                continue

            # Skip excluded categories by default
            if category in _BFCL_EXCLUDED_CATEGORIES and self._categories != BFCL_SIMPLE_CATEGORIES:
                skipped_cats.add(category)
                continue

            # Find corresponding possible_answer file
            answer_path = bfcl_data_dir / "possible_answer" / f"{data_name}.json"

            # Load answer mappings
            answers: dict[str, dict] = {}
            if answer_path.exists():
                with open(answer_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                a = json.loads(line)
                                answers[str(a.get("id", ""))] = a
                            except json.JSONDecodeError:
                                pass

            # Load entries (JSONL — one JSON object per line)
            with open(data_path, encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]

            # Attach ground_truth from answers
            for entry in entries:
                eid = str(entry.get("id", ""))
                ans = answers.get(eid)
                if ans:
                    entry["ground_truth"] = ans.get("ground_truth", [])

            items = self._convert(entries, category)
            all_items.extend(items)
            loaded_files += 1
            logger.info("BFCL/%s from local clone: %d items (%s).", category, len(items), data_name)

        if skipped_cats:
            logger.info("BFCL: skipped %d excluded categories: %s", len(skipped_cats),
                        sorted(skipped_cats))
