import json
import logging
import re
import uuid
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
from json_repair import repair_json
import jsonschema
from .types import MessagePart


from fastapi import  HTTPException
from .types import ModelConfig, ChatMessage, ChatCompletionTool
from .tools_hook import get_tool_hook

logger = logging.getLogger("llm.utils")



def resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_stop(stop: Optional[Union[str, list[str]]]) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [item for item in stop if item]


def _read_sampler_params(config: dict) -> dict[str, object]:
    """Read sampler_params dict from model.json config.

    Prefers the ``sampler_params`` key if present. Otherwise, converts
    legacy individual keys (``top_k``, ``top_p``, ``temperature``,
    ``repeat_penalty``, ``presence_penalty``) to the new schema format.
    """
    if "sampler_params" in config and isinstance(config["sampler_params"], dict):
        return dict(config["sampler_params"])

    # Fallback: auto-convert legacy keys.
    params: dict[str, object] = {}
    if "top_k" in config:
        params["top_k"] = int(config["top_k"])
    if "top_p" in config:
        params["top_p"] = float(config["top_p"])
    if "temperature" in config:
        params["temp"] = float(config["temperature"])
    if "repeat_penalty" in config:
        params["penalty_repeat"] = float(config["repeat_penalty"])
    if "presence_penalty" in config:
        params["penalty_present"] = float(config["presence_penalty"])
    return params


def load_model(model_dir: Path) -> ModelConfig:
    model_json_path = model_dir / "model.json"
    if not model_json_path.exists():
        raise FileNotFoundError(f"Missing model.json: {model_json_path}")

    config = load_json(model_json_path)
    tokenizer = config.get("tokenizer")
    stop = config.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        raise ValueError("'stop' in model.json must be a string or list of strings")

    # Read sampler_params: prefer `sampler_params` key, fall back to legacy individual keys.
    sampler_params = _read_sampler_params(config)

    return ModelConfig(
        name=str(config.get("id") or config.get("name") or model_dir.name),
        model_path=resolve_path(model_dir, str(config.get("weights", "weights.bin"))),
        tokenizer_path=resolve_path(model_dir, str(tokenizer)) if tokenizer else "",
        backend=str(config.get("backend", "adla")).lower(),
        model_type=str(config.get("model_type", "none")),
        sampling_mode=str(config.get("sampling_mode", "chain_sampler")),
        sampler_params=sampler_params,
        system_prompt=str(config.get("system_prompt", "")),
        prompt_prefix=str(config.get("prompt_prefix", "")),
        prompt_postfix=str(config.get("prompt_postfix", "")),
        stop=[str(item) for item in stop if str(item)],
        retain_history=bool(config.get("retain_history", False)),
        loglevel=str(config.get("loglevel", "ERROR")).upper(),
        context_size=int(config.get("context_size", 4096)),
        threads=int(config.get("threads", 0)),
        n_gpu_layers=int(config.get("n_gpu_layers", 0)),
        chat_format=str(config.get("chat_format", "")),
        verbose=bool(config.get("verbose", False)),
        metadata=dict(config.get("metadata", {})),
        skill_workaround=bool(config.get("skill_workaround", False)),
        template_kwargs=dict(config.get("template_kwargs", {})),
        route_only=bool(config.get("route_only", False)),
        token_pair={int(k):int(v) for k,v in config.get("token_pair", {}).items()} if config.get("token_pair") else None,
        # VLM config
        mmproj_path=resolve_path(model_dir, str(config.get("mmproj", ""))) if config.get("mmproj") else "",
        vision_start=str(config.get("vision_start", "<|vision_start|>")),
        vision_end=str(config.get("vision_end", "<|vision_end|>")),
        image_pad=str(config.get("image_pad", "<|image_pad|>")),
    )


def builtin_template(model_type: str) -> Tuple[str, str, str]:
    model_type = (model_type or "none").lower()
    system_prompt = ""
    prompt_prefix = ""
    prompt_postfix = ""

    if model_type == "qwen":
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prompt_prefix = "<|im_start|>user\n"
        prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "deepseek":
        system_prompt = "<|begin_of_sentence|>"
        prompt_prefix = "<|User|>"
        prompt_postfix = "<|Assistant|>please don't include <think> tags in your answers\n"
    elif model_type in ("gemma", "gemma3", "gemma4", "gemma4-e2b"):
        system_prompt = "<bos>"
        prompt_prefix = "<start_of_turn>user\n"
        prompt_postfix = "<end_of_turn>\n<start_of_turn>model\n"
    elif model_type == "llama":
        date_str = datetime.now().strftime("%d %b %Y")
        system_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n"
            f"Today Date: {date_str}\n\n"
            "<|eot_id|>"
        )
        prompt_prefix = "<|start_header_id|>user<|end_header_id|>\n\n"
        prompt_postfix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "tiny_llama":
        system_prompt = "<|im_start|>system\nYou are a friendly chatbot.<|im_end|>\n"
        prompt_prefix = "<|im_start|>user\n"
        prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "phi_1_5":
        prompt_postfix = "\nAnswer:"
    elif model_type == "phi_2":
        prompt_prefix = "Instruct: "
        prompt_postfix = "\nOutput:"

    return system_prompt, prompt_prefix, prompt_postfix

def build_system_prompt(model_type: str, prompt:str, think: bool =False) -> str:
    model_type = (model_type or "none").lower()
    
    system_prompt = prompt  # default: return prompt as-is

    if model_type == "qwen":
        think_flag = "" if think else "\n/no_think"
        system_prompt = f"<|im_start|>system\n{prompt}{think_flag}<|im_end|>\n"
    elif model_type == "deepseek":
        system_prompt = "<|begin_of_sentence|>"
    elif model_type in ("gemma", "gemma3", "gemma4", "gemma4-e2b"):
        system_prompt = f"<bos>{prompt}"
    elif model_type == "llama":
        date_str = datetime.now().strftime("%d %b %Y")
        system_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n"
            f"Today Date: {date_str}\n\n"
            "<|eot_id|>"
        )
    elif model_type == "tiny_llama":
        system_prompt = f"<|im_start|>system\n{prompt}<|im_end|>\n"

    return system_prompt



def messages_to_prompt(messages: list[ChatMessage]) -> str:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    conversation = []
    for message in messages:
        role = (message.role or "user").strip().lower()
        if role == "system":
            continue
        content = message_text(message.content)
        if content:
            conversation.append((role, content))

    if not conversation:
        raise HTTPException(status_code=400, detail="messages do not contain usable text content")

    if len(conversation) == 1 and conversation[0][0] == "user":
        return conversation[0][1]

    lines = []
    for role, content in conversation:
        if role == "assistant":
            lines.append("Assistant: " + content)
        elif role == "tool":
            lines.append("Tool: " + content)
        else:
            lines.append("User: " + content)
    return "\n".join(lines)


def system_prompt_from_messages(messages: list[ChatMessage]) -> str:
    parts = []
    for message in messages:
        if (message.role or "").lower() != "system":
            continue
        content = message_text(message.content)
        if content:
            parts.append(content)
    return "\n".join(parts)


def build_tool_instruction(tools: Optional[list[dict]],
                            tool_choice: Optional[Union[str, dict]]) -> str:
    if not tools:
        return ""

    def parameter_summary(parameters: object) -> str:
        if not isinstance(parameters, dict):
            return "{}"
        properties = parameters.get("properties")
        required = parameters.get("required")
        summary = {}
        if isinstance(properties, dict):
            summary["properties"] = list(properties.keys())
        if isinstance(required, list):
            summary["required"] = [str(item) for item in required]
        return json.dumps(summary or parameters, ensure_ascii=False, sort_keys=True)

    parts = [
        "You have access to tools through the proxy.",
        "Never say you do not have tools, cannot access tools, or cannot browse/search when relevant tools are listed below.",
        "Available tools:",
    ]
    for tool in tools:
        function = tool.get("function") or {}
        if not function:
            continue
        parts.append("- name: " + str(function.get("name", "")))
        parts.append("  description: " + str(function.get("description", "")))
        parts.append("  parameters: " + parameter_summary(function.get("parameters", {})))
    parts.extend([
        "When a request needs current information, web access, file access, running commands, editing files, or any external action, call a tool instead of answering from memory.",
        "If tool_choice is auto, decide whether a tool is needed. If needed, output only this exact format with no extra text:",
        "<tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>",
        "Do not add markdown fences, explanations, or surrounding prose when calling a tool.",
    ])
    if tool_choice is not None and str(tool_choice) != "auto":
        parts.append("tool_choice: " + str(tool_choice))
    parts.append("If no tool is needed, answer normally.")
    return "\n".join(parts)


def should_use_prompt_only_tools(config: ModelConfig, tools: Optional[list[dict]]) -> bool:
    if not tools:
        return False
    if use_raw_tool_call_passthrough(config):
        return True
    model_type = (config.model_type or "").lower()
    chat_format = (config.chat_format or "").lower()
    return model_type in ("gemma", "gemma3", "gemma4", "gemma4-e2b") or chat_format == "gemma"


def use_raw_tool_call_passthrough(config: ModelConfig) -> bool:
    metadata = config.metadata or {}
    raw_tool_calls = metadata.get("raw_tool_calls")
    if isinstance(raw_tool_calls, bool):
        return raw_tool_calls

    tool_passthrough = metadata.get("tool_passthrough")
    if isinstance(tool_passthrough, bool):
        return tool_passthrough

    tool_call_mode = str(metadata.get("tool_call_mode", "")).strip().lower()
    return tool_call_mode in ("raw", "raw_text", "passthrough")


def merge_system_prompt(base_system: str,
                         tools: Optional[list[dict]],
                         tool_choice: Optional[Union[str, dict]]) -> str:
    tool_instruction = build_tool_instruction(tools, tool_choice)
    if not tool_instruction:
        return base_system
    if not base_system:
        return tool_instruction
    return base_system + "\n" + tool_instruction


def merge_stop_sequences(config_stop: list[str],
                          request_stop: Optional[Union[str, list[str]]]) -> list[str]:
    merged = []
    for item in list(config_stop) + normalize_stop(request_stop):
        if item and item not in merged:
            merged.append(item)
    return merged


def message_text(content: Optional[Union[str, list[MessagePart]]]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        part.text.strip()
        for part in content
        if part.type == "text" and part.text and part.text.strip()
    )


def apply_tool_hooks(
    messages: list[ChatMessage],
    user_agent: str,
    model_type: str,
) -> None:
    """Apply registered tool hooks to process tool message content in-place.

    Builds a tool_call_id → tool_name map from assistant messages, then
    looks up the appropriate hook via :func:`get_tool_hook` for every message
    whose role is ``"tool"`` and transforms its content.
    """
    if not user_agent or get_tool_hook is None:
        return

    # Build tool_call_id -> tool_name map from all assistant messages.
    tool_call_map: Dict[str, str] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                func = tc.get("function") or {}
                tc_name = func.get("name")
                if tc_id and tc_name:
                    tool_call_map[tc_id] = tc_name

    if not tool_call_map:
        return

    # Apply hooks to tool messages.
    for msg in messages:
        if msg.role != "tool":
            continue
        tool_name = tool_call_map.get(msg.tool_call_id or "", "")
        if not tool_name:
            continue
        content = message_text(msg.content)
        if not content:
            continue
        # Treat "none" / empty model_type as no model restriction.
        model = model_type if model_type and model_type.lower() != "none" else None
        logger.debug("Applying tool hook - user_agent: %s, tool_name: %s, model: %s", user_agent, tool_name, model)
        hook = get_tool_hook(user_agent, tool_name, model)
        processed = hook(content)
        msg.content = processed


def normalize_tools(tools: Optional[list[ChatCompletionTool]]) -> Optional[list[dict]]:
    if not tools:
        return None
    return [tool.model_dump(exclude_none=True) for tool in tools]


def normalize_tool_choice(tool_choice: Optional[Union[str, dict]]) -> Optional[Union[str, dict]]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    return dict(tool_choice)


def tool_names(tools: Optional[list[dict]]) -> list[str]:
    if not tools:
        return []
    names = []
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if name:
            names.append(str(name))
    return names


def make_tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def extract_thinking_content(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n".join(matches).strip()
    return None

def extract_tool_call_blocks(text: str, tools: list[dict] = None, exception: bool = False) -> list[dict]:
    matches = re.findall(r'<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)', text, flags=re.DOTALL)
    # logger.debug("Extracted tool call blocks: %s", matches)
    # logger.debug("Available tools for extraction: %s", tools)
    tool_schema = tools_to_json_schema(tools)
    tool_names_arr = tool_names(tools)

    parsed = []
    for block in matches:
        try:
            payload = json.loads(block)
            jsonschema.validate(instance=payload, schema=tool_schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            logger.debug("Tool call block failed validation, attempting repair - block: %s, error: %s", block, str(exc))
            if isinstance(exc, jsonschema.ValidationError):
                logger.debug("Json Schema: %s", json.dumps(tool_schema, ensure_ascii=False))
                logger.warning("Failed payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
            try:
                fixed_output = repair_json(block, schema=tool_schema, skip_json_loads=True, ensure_ascii=False)
                payload = json.loads(fixed_output)
                jsonschema.validate(instance=payload, schema=tool_schema)
                logger.debug("Successfully repaired tool call block: %s", fixed_output)
            except jsonschema.ValidationError as exc2:
                if exception:
                    tool_name = payload.get("name") if isinstance(payload, dict) else None
                    if tool_name not in tool_names_arr:
                        raise ValueError(f"Extracted tool name '{tool_name}' not in available tools") from exc2
                    else:
                        raise exc2
                continue
            except Exception as exc:
                if exception:
                    raise exc
                continue
        if isinstance(payload, dict):
            parsed.append(payload)
        elif isinstance(payload, list):
            parsed.extend(item for item in payload if isinstance(item, dict))
    return parsed


def fallback_tool_calls_from_text(text: str, tools: Optional[list[dict]], **kwargs) -> list[dict]:
    if not text:
        return []

    requested_tool_names = tool_names(tools)
    logger.debug("Extracting tool calls from text: %s. Requested tool names: %s", text, requested_tool_names)
    tool_calls = []
    for payload in extract_tool_call_blocks(text, tools, exception=kwargs.get("exception", False)):
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not name:
            function = payload.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", arguments)
        if not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"input": arguments}
        elif arguments is None:
            arguments = {}
        logger.debug("Parsed tool call - name: %s, arguments: %s", name, arguments)
        if requested_tool_names and name not in requested_tool_names:
            continue
        tool_calls.append(make_tool_call(str(name), dict(arguments)))
    logger.debug("Extracted tool calls: %s", tool_calls)
    if tool_calls:
        return tool_calls

    code_matches = re.findall(
        r"<start_of_turn>tool_code\s*(.*?)\s*</start_of_turn>",
        text,
        flags=re.DOTALL,
    )
    if not code_matches:
        return []

    selected_name = None
    if "python_exec" in requested_tool_names:
        selected_name = "python_exec"
    elif len(requested_tool_names) == 1:
        selected_name = requested_tool_names[0]
    if not selected_name:
        return []

    tool_calls = []
    seen_codes = set()
    for code in code_matches:
        normalized_code = code.strip()
        if not normalized_code or normalized_code in seen_codes:
            continue
        seen_codes.add(normalized_code)
        tool_calls.append(make_tool_call(selected_name, {"code": normalized_code}))
    return tool_calls


def clean_template_artifacts(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    cleaned = re.sub(r"<think>\s*.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>\s*.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    for marker in (
        "<start_of_turn>",
        "</start_of_turn>",
        "<end_of_turn>",
        "</end_of_turn>",
        "<bos>",
    ):
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned or None


def text_without_tool_markup(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r'<tool_call>.*?(?:</tool_call>|(?=<tool_call>)|\Z)', "", text, flags=re.DOTALL)
    cleaned = re.sub(
        r"</start_of_turn>\s*<start_of_turn>tool_code\s*.*?(?=(</start_of_turn>\s*<start_of_turn>tool_code|$))",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"<start_of_turn>tool_output\s*.*?(?=(</start_of_turn>|$))",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"</?start_of_turn>|</?end_of_turn>", "", cleaned)
    cleaned = re.sub(r"\btool_code\b|\btool_output\b", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return clean_template_artifacts(cleaned)


def messages_to_llama_cpp(messages: list[ChatMessage], default_system: Optional[str]) -> list[dict]:
    normalized = []
    system_parts = []

    for message in messages:
        role = (message.role or "user").strip().lower()
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"

        # Handle multimodal content (list of MessagePart).
        if isinstance(message.content, list):
            content_parts = []
            for part in message.content:
                if isinstance(part, MessagePart):
                    part_dict = {"type": part.type}
                    if part.type == "text" and part.text:
                        part_dict["text"] = part.text
                    elif part.type == "image_url":
                        image_url = getattr(part, "image_url", None)
                        if image_url:
                            part_dict["image_url"] = image_url.model_dump() if hasattr(image_url, "model_dump") else image_url
                    content_parts.append(part_dict)
                elif isinstance(part, dict):
                    content_parts.append(part)

            if not content_parts:
                continue

            item = {"role": role, "content": content_parts}
            if message.name:
                item["name"] = message.name
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if role == "system":
                system_parts.append("\n".join(p.get("text", "") for p in content_parts if p.get("type") == "text"))
                continue
            normalized.append(item)
            continue

        text = message_text(message.content)
        if not text and not message.tool_calls:
            continue
        item = {"role": role, "content": text}
        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            item["tool_calls"] = message.tool_calls
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        normalized.append(item)

    if not normalized and not system_parts:
        raise HTTPException(status_code=400, detail="messages do not contain usable text content")

    merged_system_parts = []
    if default_system:
        merged_system_parts.append(default_system)
    merged_system_parts.extend(system_parts)
    if merged_system_parts:
        normalized.insert(0, {"role": "system", "content": "\n".join(merged_system_parts)})

    return normalized


def extract_images_from_messages(messages: list[ChatMessage]) -> list[np.ndarray]:
    """Extract images from chat messages.

    Parses message.content (which can be a list of MessagePart) for parts
    with type == "image_url". Downloads or decodes the image data into
    numpy uint8 RGB arrays.

    Args:
        messages: List of ChatMessage objects.

    Returns:
        List of numpy arrays (H×W×3, uint8, RGB). Empty if no images found.
    """
    import base64
    import io
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow is required for image processing. Install with: pip install Pillow")
        return []

    images = []

    for message in messages:
        content = message.content
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, MessagePart):
                continue
            if part.type != "image_url":
                continue

            image_url = part.image_url if hasattr(part, "image_url") else part.get("image_url", {}) if isinstance(part, dict) else {}
            if hasattr(image_url, "url"):
                url = image_url.url or ""
            elif isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = str(image_url)

            if not url:
                continue

            img = None
            if url.startswith("data:"):
                # data:image/png;base64,xxxxx
                try:
                    header, encoded = url.split(",", 1)
                    data = base64.b64decode(encoded)
                    img = Image.open(io.BytesIO(data))
                except Exception as exc:
                    logger.warning("Failed to decode base64 image: %s", exc)
                    continue
            elif url.startswith("http://") or url.startswith("https://"):
                try:
                    import httpx
                    resp = httpx.get(url, timeout=30.0)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content))
                except Exception as exc:
                    logger.warning("Failed to download image from %s: %s", url, exc)
                    continue
            elif url.startswith("file://"):
                try:
                    file_path = url[7:]
                    img = Image.open(file_path)
                except Exception as exc:
                    logger.warning("Failed to load image from %s: %s", url, exc)
                    continue
            else:
                logger.warning("Unsupported image URL scheme: %s", url[:50])
                continue

            if img is None:
                continue

            # Convert to RGB numpy array.
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            images.append(arr)
            logger.debug("Extracted image: shape=%s", arr.shape)

    return images


def preprocess_vlm_image(
    image: "np.ndarray",
    preprocess_cfg: dict,
    model_width: int = 0,
    model_height: int = 0,
):
    """Preprocess an image for VLM input.

    Steps:
    1. Expand to square (pad with gray 128).
    2. Bilinear resize to model dimensions.
    3. Normalize: (pixel/255 - mean) / std.
    4. Convert to CHW float32 format.

    Args:
        image: numpy array (H×W×3, uint8 RGB).
        preprocess_cfg: Preprocess config dict with mean, std, patch_size, merge_size, etc.
        model_width: Target width from mmproj model.
        model_height: Target height from mmproj model.

    Returns:
        numpy array (C×H×W, float32).
    """
    mean = np.array(preprocess_cfg.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.array(preprocess_cfg.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)

    h, w = image.shape[:2]

    # Expand to square.
    size = max(w, h)
    square = np.full((size, size, 3), 128, dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    square[y_off:y_off + h, x_off:x_off + w] = image

    # Bilinear resize to target dims.
    if model_width > 0 and model_height > 0:
        target_w, target_h = model_width, model_height
    else:
        # Use patch_size and merge_size from config to determine target.
        patch_size = preprocess_cfg.get("patch_size", 16)
        merge_size = preprocess_cfg.get("merge_size", 2)
        longest_edge = preprocess_cfg.get("size", {}).get("longest_edge", 16777216)
        # Default: use square size divisible by patch_size * merge_size.
        target = size
        grid_size = target // (patch_size * merge_size)
        if grid_size > 0:
            target = grid_size * patch_size * merge_size
        else:
            target = patch_size * merge_size
        target_w = target_h = target

    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(square)
        pil_img = pil_img.resize((target_w, target_h), PILImage.BILINEAR)
        resized = np.array(pil_img, dtype=np.uint8)
    except ImportError:
        # Fallback: use simple numpy resize.
        import warnings
        warnings.warn("Pillow not available for image resize; using crude numpy resize")
        # Simple downsampling.
        resized = square[::square.shape[0] // target_h, ::square.shape[1] // target_w][:target_h, :target_w]

    # Normalize.
    resized_f = resized.astype(np.float32) / 255.0
    normalized = (resized_f - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)

    # Convert to CHW.
    chw = np.transpose(normalized, (2, 0, 1)).copy()
    chw = np.ascontiguousarray(chw, dtype=np.float32)

    logger.debug("Preprocessed VLM image: shape=%s, range=[%.3f, %.3f]", chw.shape, chw.min(), chw.max())
    return chw


def parse_tokenizer_config(chat_format: str) -> dict:
    """Parse config.chat_format into a dict with chat_template, bos_token, eos_token.

    Supports:
    - A file path to a tokenizer_config.json
    - A JSON string containing the tokenizer_config dict
    - A raw Jinja2 template string
    """
    if not chat_format:
        return {}

    # Try as file path
    try:
        path = Path(chat_format)
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass

    # Try as JSON string
    try:
        data = json.loads(chat_format)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try as raw Jinja2 template string
    if "{%" in chat_format or "{{" in chat_format:
        return {"chat_template": chat_format}

    return {}


def render_chat_template(
    template_str: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    add_generation_prompt: bool = True,
    bos_token: str = "",
    eos_token: str = "",
    **kwargs,
) -> str:
    """Render messages through a Jinja2 chat template.

    Args:
        template_str: Jinja2 template string (e.g. from tokenizer_config.json).
        messages: list of message dicts with 'role' and 'content' keys.
        tools: Optional list of tool definitions.
        add_generation_prompt: Whether to append the assistant generation prompt.
        bos_token: Beginning-of-sequence token.
        eos_token: End-of-sequence token.

    Returns:
        The rendered prompt string.
    """
    import jinja2

    env = jinja2.Environment()
    env.policies['json.dumps_kwargs'] = {'ensure_ascii': False, 'sort_keys': False}
    # Provide common variables that tokenizer config templates expect
    template = env.from_string(template_str)
    tools_rsort = [{"type": tool.get('type'), 'function': {'name': tool['function'].get('name', ''), 
                                                           'description': tool['function'].get('description', ''),
                                                           'parameters': tool['function'].get('parameters', {})}} for tool in (tools or []) if isinstance(tool, dict) and 'function' in tool]
    
    return template.render(
        messages=messages,
        tools=tools_rsort or [],
        add_generation_prompt=add_generation_prompt,
        bos_token=bos_token,
        eos_token=eos_token,
        image_pad=kwargs.pop("image_pad", "<|image_pad|>"),
        vision_start=kwargs.pop("vision_start", "<|vision_start|>"),
        vision_end=kwargs.pop("vision_end", "<|vision_end|>"),
        **kwargs,
    )

def tool_debug_summary(tools: Optional[list[ChatCompletionTool]]) -> list[dict]:
    if not tools:
        return []
    summary = []
    for tool in tools:
        summary.append({
            "type": tool.type,
            "name": tool.function.name,
        })
    return summary


def message_debug_summary(messages: list[dict], limit: int = 2) -> list[dict]:
    summary = []
    for message in messages[:limit]:
        content = message.get("content") or ""
        summary.append({
            "role": message.get("role"),
            "content_preview": content[:400],
        })
    return summary


def tools_to_json_schema(tools: list[ChatCompletionTool]) -> Dict[str, Any]:
    """
    将 OpenAI tools 列表转换为 JSON Schema，用于验证模型输出的 tool call 格式。

    生成的 schema 描述一个对象，包含：
        - name: 字符串，必须是某个 tool 的名称
        - arguments: 对象，其具体字段由 name 对应的 tool 的 parameters 决定

    Args:
        tools: ChatCompletionTool 实例或字典的列表，符合 OpenAI tools 参数格式

    Returns:
        一个 JSON Schema 字典，可用于 jsonschema.validate() 等验证工具
    """
    if not tools:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    definitions = {}
    all_of_conditions = []
    name_enum = []
    for idx, tool in enumerate(tools):
        # 兼容字典和 Pydantic 对象
        if isinstance(tool, dict):
            func = tool.get("function", {})
            name = func.get("name")
            parameters = func.get("parameters")
        else:
            name = tool.function.name
            parameters = tool.function.parameters
        if not name:
            continue
        name_enum.append(name)
        # 处理参数 schema
        if parameters is None:
            param_schema = {"type": "object", "additionalProperties": False}
        else:
            param_schema = parameters.copy() if isinstance(parameters, dict) else {}
            if "type" not in param_schema:
                # 默认为 object 类型
                param_schema = {"type": "object", "properties": param_schema.get("properties", {})}
        def_key = f"Tool{idx}Params"
        definitions[def_key] = param_schema
        condition = {
            "if": {
                "properties": {"name": {"const": name}}
            },
            "then": {
                "properties": {"arguments": {"$ref": f"#/definitions/{def_key}"}}
            }
        }
        all_of_conditions.append(condition)
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": name_enum},
            "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
        "allOf": all_of_conditions,
        "definitions": definitions
    }
    return schema


def preprocess_system_skills(input: str) -> tuple[str, str, list[dict[str, str]]]:
    """
    从 System Prompt 中提取并移除 Skills 相关内容。
    
    返回:
        1. 移除 Skill 相关节后的文本 (str)
        2. Skill 文件路径模板，其中 `{skill_name}` 可用于 str.format (str)
        3. Skill 列表，每个元素为 {"skill_name": name, "description": desc} (list[dict])
    """
    # 1. 提取 Skill 文件路径模板 (Workspace 部分)
    skill_location_template = ""
    # 匹配 ## Workspace 下的 Skills 行，允许 %s 占位符
    workspace_pattern = r'## Workspace.*?- Skills:\s*(.*?)\n'
    match_ws = re.search(workspace_pattern, input, re.DOTALL)
    if match_ws:
        raw_path = match_ws.group(1).strip()
        # 将 {skill-name} 替换为 {skill_name} 以支持 str.format
        skill_location_template = raw_path.replace('{skill-name}', '{skill_name}')
    
    # 2. 查找并移除 # Skills 节（包括标题和后续 <skills>...</skills> 块）
    # 使用非贪婪匹配 .*? 以及 re.DOTALL 让 . 匹配换行
    skills_section_pattern = r'(^|\n)# Skills\b.*?<skills>.*?</skills>'
    match_skills_section = re.search(skills_section_pattern, input, re.DOTALL)
    
    if match_skills_section:
        # 删除整个匹配节，同时注意保留前后的分隔符，避免多余空行
        start, end = match_skills_section.span()
        # 如果匹配开头是 \n，则一起删除，否则删除到结束
        processed_text = input[:start] + input[end:]
        # 处理可能产生的连续空行：将三个以上换行压缩为两个换行（保留段落间距）
        processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
        # 提取技能列表
        skills_xml = match_skills_section.group(0)
    else:
        # 没有找到 Skills 节，直接使用原文，空列表
        processed_text = input
        skills_xml = ""
    
    # 3. 从 skills_xml 中解析每个 <skill> 的名称和描述
    skills_list: list[dict[str, str]] = []
    if skills_xml:
        # 找到所有 <skill>...</skill> 块
        skill_blocks = re.finditer(r'<skill>(.*?)</skill>', skills_xml, re.DOTALL)
        for block in skill_blocks:
            skill_content = block.group(1)
            # 提取 name
            name_match = re.search(r'<name>(.*?)</name>', skill_content, re.DOTALL)
            # 提取 description
            desc_match = re.search(r'<description>(.*?)</description>', skill_content, re.DOTALL)
            if name_match and desc_match:
                skill_name = name_match.group(1).strip()
                description = desc_match.group(1).strip()
                skills_list.append({"skill_name": skill_name, "description": description})
    
    return processed_text, skill_location_template, skills_list


def build_skill_routing_prompt(skills: list[dict[str, str]]) -> str:
    """
    根据技能列表和模板构建技能路由提示词。
    
    Args:
        skills: 每个元素为 {"skill_name": name, "description": desc} 的列表
    
    Returns:
        拼接所有技能信息后的提示词字符串
    """
    prompt = '''你是一个请求路由分类器。仅输出 JSON，不要任何解释。

## 分类规则
local（须同时满足）：
- 简单常识/事实问答、简短对话、问候、闲聊，或基础文本处理（翻译/摘要/改写）
- 上下文 ≤ 3000 token，无需深度推理、长上下文或多模态
- 若是实时信息查询（天气/股票/新闻），必须命中下方技能列表且请求仅需该技能

cloud（满足任一即 cloud）：
- 复杂编程（代码生成/调试/重构/架构设计）或需要未列出技能的工具/多步操作（如文件、命令、git）
- 数学或逻辑推理、复杂分析
- 长文档（>3000 token）分析或生成
- 需要多模态理解（图像/音频/视频）
- 专业领域或敏感内容：医疗、法律、金融投资建议、隐私信息分析
- 网络不可用但需要实时信息
- 不确定时一律 cloud

## 意图
仅允许：general / coding / reasoning / multimodal / long_context

## 技能（用不到填 null；命中技能且无其他 cloud 触发 → local）
{{skill_list}}

## 输出（严格遵守）
{"local": true/false, "intent": "类别", "confidence": 0.0-1.0, "use_skill": "技能"或null}

## 硬约束（优先级最高，违反即判错）
- 仅输出 JSON
- intent=coding/reasoning/multimodal/long_context → local=false
- 含图片/音频/视频 → intent=multimodal 且 local=false
- 上下文估算 >3000 → local=false
- [CONTEXT] network_available=false 且请求需要实时/最新信息 → local=false
- [CONTEXT] tools 非空且不在技能列表（weather/stock-price）→ local=false'''

    skill_entries = []
    for skill in skills:
        entry = " - **{skill_name}**: {description}".format(skill_name=skill["skill_name"], description=skill["description"])
        skill_entries.append(entry)
    skill_text = "\n".join(skill_entries)

    prompt = prompt.replace("{{skill_list}}", skill_text)
    return prompt


def try_decode_routing_response(text: str) -> tuple[bool, Optional[str]]:
    """
    尝试从模型输出文本中解析技能路由决策的 JSON。
    
    Args:
        text: 模型输出的文本，预期包含 JSON 格式的技能路由决策
    
    Returns:
        解析后的字典，例如 {"use_skill": true, "name": "Skill Name"}，如果解析失败则返回 {"use_skill": false, "name": null}
    """

    json_schema = {
        "type": "object",
        "properties": {
            "local": {"type": "boolean"},
            "intent": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "use_skill": {"type": ["string", "null"]},
        },
        "required": ["local"],
        "additionalProperties": False
    }
    try:
        # 使用正则提取第一个 JSON 对象
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            obj = json.loads(json_str)
            jsonschema.validate(instance=obj, schema=json_schema)
            skill_name = obj.get("use_skill", None)
            if skill_name is not None and skill_name.lower() in ["null", "none"]:
                skill_name = None
            return obj['local'], obj.get('intent'), skill_name
    except (json.JSONDecodeError, jsonschema.ValidationError):
        try:
            # 如果直接解析失败，尝试修复 JSON 格式后再解析
            repaired = repair_json(text, ensure_ascii=False, skip_json_loads=True, schema=json_schema)
            obj = json.loads(repaired)
            return obj.get('local', False), obj.get('intent', None), obj.get('use_skill', None),
        except:
            pass
    return False, None, None


def get_skill_content(input: str, skill_dir:str) -> str:
    """
    移除 SKILL.md 文件开头的元数据块（YAML front matter）。
    
    分隔符由至少三个连字符（"-"）组成的独立行表示，
    例如 "---"、"----" 等，行内允许前后空白。
    删除从第一个分隔符行到第二个分隔符行（含）之间的所有内容，
    返回剩余正文（去除首尾空白）。
    若未找到成对的分隔符，则直接返回原文。
    """
    # 匹配开头分隔符行 + 中间内容（非贪婪） + 结束分隔符行
    # 结束分隔符行之后允许可选换行，以便正文从下一行开始
    pattern = r'^\s*-{3,}\s*\r?\n.*?\r?\n\s*-{3,}\s*(?:\r?\n)?'
    match = re.match(pattern, input, re.DOTALL)
    if match:
        body = input[match.end():]
        return body.strip().replace('{{SKILL_DIR}}',skill_dir)
    # 无 front matter 时直接返回原文
    return input.strip().replace('{{SKILL_DIR}}',skill_dir)


def estimate_token_count(messages: list[ChatMessage]) -> int:
    """Estimate the number of tokens in a list of messages.

    Uses a simple character-based heuristic: total characters divided by 4.
    This is a conservative estimate suitable for context-length pre-checks.
    """
    total_chars = 0
    for msg in messages:
        content = message_text(msg.content)
        total_chars += len(content)
        # account for tool_calls JSON serialization overhead
        if msg.tool_calls:
            total_chars += len(json.dumps(msg.tool_calls, ensure_ascii=False))
        # account for role and structural tokens
        total_chars += len(msg.role)
    return max(1, total_chars // 4)


def inject_skills_for_forwarding(messages: list[ChatMessage], skill_name: Optional[str] = None) -> list[ChatMessage]:
    """Inject resolved skill content into the system prompt for upstream forwarding.

    Scans the first system-prompt message for skill references (using
    :func:`preprocess_system_skills`), loads each referenced skill file,
    strips YAML front matter, and replaces the skill XML blocks with the
    actual skill content in a new system prompt.

    Returns a new list of ChatMessage with the injected system prompt as
    the first message, followed by the remaining original messages.
    """
    if not messages:
        return list(messages)

    # Find the first system message
    system_idx = None
    system_text = ""
    for i, msg in enumerate(messages):
        if msg.role == "system":
            system_idx = i
            system_text = message_text(msg.content)
            break

    if system_idx is None or not system_text:
        return list(messages)

    # Use raw (unstripped) content for preprocess_system_skills, since it
    # relies on newline positions for regex matching.
    raw_system_text = messages[system_idx].content
    if isinstance(raw_system_text, list):
        # Multi-part content — join all text parts without stripping
        raw_system_text = "\n".join(
            part.text for part in raw_system_text
            if part.type == "text" and part.text
        )
    elif raw_system_text is None:
        raw_system_text = ""

    # Extract skill references and path template
    _fixed_prompt, skill_path_template, skill_list = preprocess_system_skills(raw_system_text)

    if not skill_list or not skill_path_template:
        # No skills to inject
        return list(messages)

    # Build injected system prompt: start with the fixed (skill-stripped) prompt
    injected_parts: list[str] = [_fixed_prompt.strip()] if _fixed_prompt.strip() else []

    for skill in skill_list:
        skill_name_ = skill.get("skill_name", "")
        if not skill_name_:
            continue
        if skill_name is None:
            break
        # If a specific skill_name is requested, only inject that one
        if skill_name is not None and skill_name_ != skill_name:
            continue
        try:
            skill_file_path = skill_path_template.format(skill_name=skill_name_)
        except (KeyError, ValueError):
            logger.warning("Could not format skill path template '%s' with skill '%s'",
                           skill_path_template, skill_name)
            continue
        try:
            injected_parts.append(f"# Actived Skill: {skill_name_}")
            with open(skill_file_path, "r", encoding="utf-8") as fp:
                raw_skill = fp.read()
            skill_dir = str(Path(skill_file_path).parent)
            resolved = get_skill_content(raw_skill, skill_dir)
            injected_parts.append(resolved)
        except (OSError, IOError) as exc:
            logger.warning("Failed to load skill file '%s': %s", skill_file_path, exc)

    injected_system = "\n\n".join(injected_parts)

    # Build new messages list: injected system prompt + all non-system original messages
    new_messages: list[ChatMessage] = [ChatMessage(role="system", content=injected_system)]
    for i, msg in enumerate(messages):
        if i == system_idx:
            continue
        new_messages.append(msg)

    return new_messages
