# -*- coding: utf-8 -*-
"""TAU2-Bench dataset loader.

Download from Hugging Face ``tau-bench/TAU2-Bench`` and convert to
multi-turn agent trajectory format.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from .base import BenchmarkDataset

logger = logging.getLogger("benchmark.datasets.tau2_bench")

TAU2_DOMAINS = ["airline", "retail", "telecom"]

# Original TAU2-Bench agent instruction from tau2/agent/llm_agent.py
_AGENT_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

# Original system prompt template from tau2/agent/llm_agent.py
_SYSTEM_PROMPT_TEMPLATE = """\
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>"""

# Domain policy file names (some domains use different file names)
_POLICY_FILES: dict[str, list[str]] = {
    "airline": ["policy.md"],
    "retail": ["policy.md"],
    "telecom": ["main_policy.md"],
}


class TAU2BenchDataset(BenchmarkDataset):
    dataset_name = "tau2_bench"
    hf_repo = "tau-bench/TAU2-Bench"
    hf_config = "default"
    hf_split = "test"

    def __init__(self, cache_dir: str | Path = "dataset"):
        super().__init__(cache_dir)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> None:
        if self.processed_path.exists():
            logger.info("TAU2-Bench already downloaded and processed at %s", self.processed_path)
            return

        logger.info("Downloading TAU2-Bench from %s...", self.hf_repo)
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "TAU2-Bench requires 'datasets' package. Install with: pip install datasets"
            )

        all_items: list[dict] = []
        downloaded_any = False
        for domain in TAU2_DOMAINS:
            try:
                ds = load_dataset(self.hf_repo, domain, split=self.hf_split)
                items = self._convert(ds, domain)
                all_items.extend(items)
                downloaded_any = True
                logger.info("TAU2-Bench/%s: %d tasks loaded.", domain, len(items))
            except Exception as exc:
                logger.debug("TAU2-Bench/%s from HF failed: %s. Will try local clone.", domain, exc)

        if not downloaded_any:
            self._load_from_local_clone(all_items)

        self._save_processed(all_items)
        logger.info("TAU2-Bench: %d total items downloaded.", len(all_items))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, force_reload: bool = False) -> list[dict]:
        if force_reload or not self.processed_path.exists():
            self.download()

        with open(self.processed_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
            logger.info("TAU2-Bench: using fixed sample of %d items.", len(indices))
            return [all_items[i] for i in indices]

        rng = random.Random(seed)
        if n >= len(all_items):
            return all_items

        # Stratify by domain
        by_domain: dict[str, list[int]] = {}
        for i, item in enumerate(all_items):
            domain = item.get("domain", "unknown")
            by_domain.setdefault(domain, []).append(i)

        n_per_domain = max(1, n // max(len(by_domain), 1))
        selected: list[int] = []
        for domain, idx_list in by_domain.items():
            k = min(n_per_domain, len(idx_list))
            selected.extend(rng.sample(idx_list, k))

        if len(selected) < n:
            remaining = [i for i in range(len(all_items)) if i not in set(selected)]
            extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

        selected = sorted(selected)[:n]
        with open(sample_path, "w", encoding="utf-8") as f:
            _json.dump(selected, f)
        logger.info("TAU2-Bench: sampled %d items, saved indices.", len(selected))
        return [all_items[i] for i in selected]

    # ------------------------------------------------------------------
    # Internal conversion
    # ------------------------------------------------------------------

    def _convert(self, ds, domain: str) -> list[dict]:
        """Convert HuggingFace TAU2-Bench dataset to unified multi-turn format."""
        items = []
        for i, row in enumerate(ds):
            turns = self._parse_turns(row, domain)
            if not turns:
                continue

            items.append({
                "item_id": f"tau2_{domain}_{i:05d}",
                "domain": domain,
                "user_task": row.get("user_task", row.get("task", "")),
                "turns": turns,
            })

        return items

    def _parse_turns(self, row, domain: str) -> list[dict]:
        """Parse a single TAU2-Bench trajectory into turn-by-turn data."""
        # TAU2-Bench format varies by version; handle common structures
        messages = row.get("messages", row.get("dialogue", []))
        if not messages:
            return []

        # Extract tools from the trajectory
        tools = row.get("tools", row.get("functions", []))

        turns = []
        turn_index = 0
        pending_user_content = ""
        system_content = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
                continue
            elif role == "user":
                pending_user_content = content
                continue
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    fn = tc.get("function", {})
                    expected_name = fn.get("name", "")
                    expected_args = fn.get("arguments", {})
                    if isinstance(expected_args, str):
                        try:
                            expected_args = json.loads(expected_args)
                        except json.JSONDecodeError:
                            expected_args = {}

                    # Build messages for this turn
                    turn_msgs = []
                    if system_content:
                        turn_msgs.append({"role": "system", "content": system_content})
                    turn_msgs.append({"role": "user", "content": pending_user_content})

                    turns.append({
                        "turn_index": turn_index,
                        "messages": turn_msgs,
                        "tools": tools if tools else None,
                        "expected_tool_name": expected_name,
                        "expected_tool_arguments": expected_args,
                        "tool_response": None,  # Will be set from next tool message
                    })
                    turn_index += 1
            elif role == "tool":
                # Attach tool response to the previous turn
                if turns and turns[-1]["tool_response"] is None:
                    turns[-1]["tool_response"] = {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }

        # If no turns were extracted via tool_calls, try simpler format
        if not turns and pending_user_content:
            turns.append({
                "turn_index": 0,
                "messages": [
                    {"role": "system", "content": system_content or _AGENT_INSTRUCTION},
                    {"role": "user", "content": pending_user_content},
                ],
                "tools": tools if tools else None,
                "expected_tool_name": "",
                "expected_tool_arguments": None,
                "tool_response": None,
            })

        return turns

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_processed(self, items: list[dict]) -> None:
        with open(self.processed_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _load_from_local_clone(self, all_items: list[dict]) -> None:
        """Fallback: load TAU2-Bench data from local GitHub clone (dataset_repos/tau2-bench)."""
        import os as _os

        clone_dir = Path(self.cache_dir).parent.parent / "dataset_repos" / "tau2-bench"
        if not clone_dir.exists():
            logger.warning("TAU2-Bench local clone not found at %s. Run scripts/extract_datasets.py first.", clone_dir)
            return

        for domain in TAU2_DOMAINS:
            tasks_json = clone_dir / "data" / "tau2" / "domains" / domain / "tasks.json"
            if not tasks_json.exists():
                continue
            with open(tasks_json, encoding="utf-8") as f:
                tasks = json.load(f)
            items = self._convert_from_tasks(tasks, domain, clone_dir)
            all_items.extend(items)
            logger.info("TAU2-Bench/%s from local clone: %d tasks.", domain, len(items))

    def _convert_from_tasks(self, tasks: list[dict], domain: str, clone_dir: Path) -> list[dict]:
        """Convert TAU2-Bench tasks.json format to unified multi-turn dict format.

        Uses the original TAU2-Bench system prompt format with embedded domain policy.
        Each action in evaluation_criteria.actions is grouped by action_id prefix
        (e.g., '0_0', '0_1' → same turn; '1_0' → next turn).
        """
        import re as _re
        from collections import defaultdict as _defaultdict

        items = []
        # Read tools from the domain's tools.py + user_tools.py
        tools_py = clone_dir / "src" / "tau2" / "domains" / domain / "tools.py"
        user_tools_py = clone_dir / "src" / "tau2" / "domains" / domain / "user_tools.py"
        tools = TAU2BenchDataset._parse_domain_tools(tools_py, tasks, extra_py=user_tools_py)

        # Load domain policy (original clone only)
        domain_policy_content = TAU2BenchDataset._load_domain_policy(domain, clone_dir)

        for task in tasks:
            tid = task["id"]
            us = task.get("user_scenario", {}).get("instructions", {})
            reason = us.get("reason_for_call", "")
            known = us.get("known_info", "")
            task_inst = us.get("task_instructions", "")

            # Build system prompt matching original TAU2-Bench format
            if domain_policy_content:
                system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                    agent_instruction=_AGENT_INSTRUCTION,
                    domain_policy=domain_policy_content,
                )
            else:
                system_prompt = _AGENT_INSTRUCTION

            user_msg = f"Reason for call: {reason}\nInformation: {known}"
            if task_inst:
                user_msg += f"\nAdditional instructions: {task_inst}"

            # Base messages for the conversation start
            base_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]

            actions = task.get("evaluation_criteria", {}).get("actions", [])
            turns = []
            if actions:
                action_groups = _defaultdict(list)
                for a in actions:
                    aid = str(a.get("action_id", "0_0"))
                    parts = aid.split("_")
                    group_key = parts[0] if parts else "0"
                    action_groups[group_key].append(a)

                sorted_groups = sorted(action_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0)

                # Build cumulative conversation history
                conversation_history: list[dict] = []

                for gi, group_key in enumerate(sorted_groups):
                    group_actions = action_groups[group_key]
                    all_expected = [
                        {"name": a["name"], "arguments": a.get("arguments")}
                        for a in group_actions
                    ]

                    # Build messages for this turn: base + conversation history
                    turn_msgs = list(base_msgs) + list(conversation_history)

                    # Generate mock tool responses for the NEXT turn's history
                    # (these represent what the environment would return)
                    mock_responses = []
                    for a in group_actions:
                        mock_responses.append({
                            "role": "tool",
                            "tool_call_id": f"tau2_{domain}_{tid}_{group_key}_{a['name']}",
                            "content": json.dumps({"status": "success"}),
                        })

                    turn_idx = int(group_key) if group_key.isdigit() else gi

                    turns.append({
                        "turn_index": turn_idx,
                        "messages": turn_msgs,
                        "tools": tools,
                        "expected_tool_name": all_expected[0]["name"],
                        "expected_tool_arguments": all_expected[0].get("arguments"),
                        "all_expected_actions": all_expected,
                        "tool_response": None,
                        "_mock_responses": mock_responses,
                    })

                    # Append expected assistant + mock tool responses to conversation
                    # history for the next turn (model needs to see prior interaction)
                    if gi < len(sorted_groups) - 1:
                        next_actions = action_groups[sorted_groups[gi + 1]]
                        for a in group_actions:
                            conversation_history.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": f"tau2_{domain}_{tid}_{group_key}_{a['name']}",
                                    "type": "function",
                                    "function": {
                                        "name": a["name"],
                                        "arguments": json.dumps(a.get("arguments", {})),
                                    },
                                }],
                            })
                        for mr in mock_responses:
                            conversation_history.append(mr)
            else:
                turns.append({
                    "turn_index": 0,
                    "messages": list(base_msgs),
                    "tools": tools,
                    "expected_tool_name": "",
                    "expected_tool_arguments": None,
                    "all_expected_actions": [],
                    "tool_response": None,
                    "_mock_responses": [],
                })

            reward_basis = task.get("evaluation_criteria", {}).get("reward_basis", [])

            items.append({
                "item_id": f"tau2_{domain}_{tid}",
                "domain": domain,
                "user_task": reason,
                "turns": turns,
                "reward_basis": reward_basis,
            })

        return items

    @staticmethod
    def _load_domain_policy(domain: str, clone_dir: Path) -> str:
        """Load the domain policy markdown file from the local clone.

        Returns the policy content, or empty string if not found.
        """
        domain_policy_dir = clone_dir / "data" / "tau2" / "domains" / domain
        for policy_name in _POLICY_FILES.get(domain, ["policy.md"]):
            policy_path = domain_policy_dir / policy_name
            if policy_path.exists():
                return policy_path.read_text(encoding="utf-8")
        return ""

    # Python type annotation → JSON Schema type mapping
    _TYPE_MAP: dict[str, str] = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "dict": "object",
        "list": "array",
    }
    # Known TAU2 domain enum types (map to string in JSON Schema)
    _ENUM_TYPES: set[str] = {
        "FlightType", "CabinClass", "Insurance", "AirportCode",
        "OrderStatus", "ProductType",
    }

    @staticmethod
    def _parse_domain_tools(tools_py: Path, tasks: list[dict], extra_py: Path | None = None) -> list[dict]:
        """Parse TAU2-Bench tools.py for OpenAI tool definitions with correct types.

        Extracts Python type annotations from function signatures and maps them
        to JSON Schema types (str→string, int→integer, List[X]→array, etc.),
        matching the original repo's Pydantic-based `model_json_schema()` output.
        """
        import re as _re

        # Collect all expected tool names
        expected_names: set[str] = set()
        for task in tasks:
            for a in task.get("evaluation_criteria", {}).get("actions", []):
                name = a.get("name", "")
                if name and not name.startswith("_"):
                    expected_names.add(name)

        if not expected_names:
            return []

        # Combine content from all source files
        sources: list[str] = []
        if tools_py.exists():
            sources.append(tools_py.read_text(encoding="utf-8"))
        if extra_py and extra_py.exists():
            sources.append(extra_py.read_text(encoding="utf-8"))

        tools = []

        for func_name in sorted(expected_names):
            doc = None
            signature = None
            # Search all source files for func def + docstring
            for content in sources:
                # Match: def func_name(params) -> ReturnType:\n    """docstring"""
                pat = _re.compile(
                    r'def\s+' + _re.escape(func_name) + r'\s*\(([^)]*)\)\s*(?:->.*?)?:\s*\n(\s+)"""(.*?)"""',
                    _re.DOTALL,
                )
                m = pat.search(content)
                if m:
                    signature = m.group(1).strip()
                    doc = m.group(3).strip()
                    break

            if not doc:
                tools.append({"type": "function", "function": {
                    "name": func_name,
                    "description": func_name.replace("_", " "),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }})
                continue

            desc_lines = doc.split("\n")
            desc = desc_lines[0].strip() if desc_lines else func_name.replace("_", " ")

            # Parse parameter types from the function signature
            param_types: dict[str, dict] = {}
            if signature:
                param_types = TAU2BenchDataset._parse_signature_types(signature)

            # Extract Args section from docstring for descriptions
            args_section = _re.search(
                r"Args:\s*\n(.*?)(?:\n\s*(?:Returns|Raises|Note|Attribute):|\n\s*$|\Z)",
                doc, _re.DOTALL,
            )

            properties: dict[str, dict] = {}
            required: list[str] = []

            if args_section:
                for al in args_section.group(1).strip().split("\n"):
                    al = al.strip()
                    if not al or ":" not in al:
                        continue
                    colon_idx = al.index(":")
                    pname = al[:colon_idx].strip()
                    # Strip type annotation in parens from docstring: "param (Type): desc"
                    pname = _re.sub(r'\s*\(.*?\)\s*$', '', pname).strip()
                    pdesc = al[colon_idx + 1:].strip()

                    # Get type from signature, fallback to string
                    type_info = param_types.get(pname, {"type": "string"})
                    prop = {"type": type_info["type"], "description": pdesc}

                    # Handle array types
                    if type_info.get("items"):
                        prop["items"] = type_info["items"]
                    elif type_info["type"] == "array":
                        prop["items"] = {"type": "string"}

                    properties[pname] = prop

                    # Add to required if not Optional
                    if not type_info.get("optional", False):
                        required.append(pname)

            else:
                # No Args section — build properties from signature types alone
                for pname, type_info in param_types.items():
                    prop = {"type": type_info["type"], "description": pname}
                    if type_info.get("items"):
                        prop["items"] = type_info["items"]
                    properties[pname] = prop
                    if not type_info.get("optional", False):
                        required.append(pname)

            tools.append({"type": "function", "function": {
                "name": func_name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }})

        return tools

    @staticmethod
    def _parse_signature_types(signature: str) -> dict[str, dict]:
        """Parse Python function signature to extract parameter types.

        Handles: self, name: str, name: int, name: List[str],
        name: Optional[str], name: FlightType (enum → string).

        Returns: {param_name: {type, optional, items}}
        """
        import re as _re

        result: dict[str, dict] = {}
        if not signature:
            return result

        # Split params by comma, respecting nested brackets
        params = TAU2BenchDataset._split_params(signature)

        for param in params:
            param = param.strip()
            if not param or param == "self":
                continue

            # Split on first ':' to get name and type annotation
            if ":" in param:
                name, type_str = param.split(":", 1)
                name = name.strip()
                type_str = type_str.strip()
            else:
                name = param.strip()
                type_str = "str"  # Default type

            # Check for default value (= ...)
            if "=" in name:
                name = name.split("=")[0].strip()
            if "=" in type_str:
                eq_pos = type_str.index("=")
                type_str = type_str[:eq_pos].strip()

            # Parse the type
            type_info = TAU2BenchDataset._python_type_to_json_schema(type_str)
            result[name] = type_info

        return result

    @staticmethod
    def _split_params(signature: str) -> list[str]:
        """Split comma-separated params respecting nested <>, [], {}."""
        params = []
        depth = 0
        current = ""
        for ch in signature:
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth -= 1
            elif ch == "," and depth == 0:
                params.append(current)
                current = ""
                continue
            current += ch
        if current.strip():
            params.append(current)
        return params

    @classmethod
    def _python_type_to_json_schema(cls, type_str: str) -> dict:
        """Convert a Python type annotation string to JSON Schema type info.

        Examples:
            "str" → {"type": "string"}
            "int" → {"type": "integer"}
            "List[str]" → {"type": "array", "items": {"type": "string"}}
            "Optional[str]" → {"type": "string", "optional": True}
            "FlightType" → {"type": "string"} (known enum)
        """
        import re as _re

        type_str = type_str.strip()

        # Handle Optional[X]
        optional = False
        opt_match = _re.match(r'Optional\[(.+)\]\s*$', type_str)
        if opt_match:
            optional = True
            type_str = opt_match.group(1).strip()

        # Handle List[X]
        list_match = _re.match(r'(?:list|List)\[(.+)\]\s*$', type_str)
        if list_match:
            inner = list_match.group(1).strip()
            # Handle List[X | Y] (union types)
            inner = inner.split("|")[0].strip()
            inner_type = cls._TYPE_MAP.get(inner, "string")
            if inner in cls._ENUM_TYPES:
                inner_type = "string"
            return {"type": "array", "items": {"type": inner_type}, "optional": optional}

        # Handle Union[X, Y] or X | Y — take first type
        if "|" in type_str:
            first = type_str.split("|")[0].strip()
            inner = cls._TYPE_MAP.get(first, "string")
            if first in cls._ENUM_TYPES:
                inner = "string"
            return {"type": inner, "optional": optional}

        # Direct type mapping
        json_type = cls._TYPE_MAP.get(type_str, "string")
        if type_str in cls._ENUM_TYPES:
            json_type = "string"

        return {"type": json_type, "optional": optional}
