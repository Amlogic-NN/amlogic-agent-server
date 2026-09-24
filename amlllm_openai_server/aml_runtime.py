from .model_runtime import BaseModelRuntime
from .types import ModelConfig, ChatMessage, ChatCompletionTool, RoutingConfig, RoutedResponse
from .tools_hook import tool_hook, get_tool_hook
from typing import Optional, Union
from .llm_utils import parse_tokenizer_config, render_chat_template, system_prompt_from_messages, messages_to_prompt 
from .llm_utils import build_system_prompt, tool_names, merge_stop_sequences, normalize_tools, normalize_tool_choice
from .llm_utils import apply_tool_hooks, message_text, merge_system_prompt, fallback_tool_calls_from_text, make_tool_call
from .llm_utils import text_without_tool_markup, messages_to_llama_cpp, builtin_template, extract_thinking_content
from .llm_utils import build_skill_routing_prompt, try_decode_routing_response, preprocess_system_skills, get_skill_content
from .llm_utils import inject_skills_for_forwarding
from .llm_utils import extract_images_from_messages, preprocess_vlm_image
from .projectors import get_projector

import logging
import json
import queue
import os
import time


def _json_default(obj: object) -> object:
    """Serialize pydantic models (and anything else) for debug logging.

    ``preprocess_vlm_image`` returns a list whose items are either ``ChatMessage``
    objects or lists of ``MessagePart``, so a plain ``model_dump()`` loop breaks
    as soon as an image is present.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    return str(obj)

try:
    from .aml_llmsdk import LLMSDK
    from .aml_llmsdk.types import Result, RunStatus
except ImportError:
    LLMSDK = None
    class Result:
        text: bytes = b""
        token_id: int = 0
    class RunStatus:
        NORMAL = 0
        FINISH = 1
        ERROR = 2


logger = logging.getLogger("llm.adla")


class AmlLlmModelRuntime(BaseModelRuntime):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: LLMSDK = None
        self.queue = None
        self.initialized = False
        self.generated_text = ""
        self.generated_reasoning = ""
        self.generated_tool_text = ""
        self.generated_token_count = 0
        self.stop_sequences: list[str] = []
        self.max_tokens: Optional[int] = None
        self.finish_reason = "stop"
        self.stop_requested = False
        self._previous_messages: Optional[list[ChatMessage]] = None
        # Parse config.chat_format to extract Jinja2 chat template (compatible with tokenizer_config.json)
        self._tokenizer_config = parse_tokenizer_config(config.chat_format)
        self._chat_template_str = self._tokenizer_config.get("chat_template", None)
        self.prefill_start = 0
        self.prefill_end = 0
        self.decode_end = 0
        self._loop_detector = {}
        self.loop_detected = False
        self._sdk_tool_calls: list[tuple[str, str]] = []

    def _tool_callback(self, tool_name: str, tool_args: str, _user_data):
        """Python-side tool-call callback (mirrors ``on_token``).

        Invoked by the SDK's PEG parser via the C bridge; collects each
        ``(name, args)`` pair for the current run so :meth:`direct_run` can
        convert them into OpenAI tool-call dicts after inference completes.
        """
        self._sdk_tool_calls.append((tool_name or "", tool_args or ""))

    def _callback(self, payload: Result, _user_data, status: RunStatus):
        raw_text = payload.text or b""
        if isinstance(raw_text, bytes):
            text = raw_text.decode("utf-8", errors="ignore")
        else:
            text = str(raw_text)
        status_str = status.name if isinstance(status, RunStatus) else str(status)
        if self.prefill_end == 0:
            self.prefill_end = time.perf_counter()
            logger.info(f"Prefill latency: {(self.prefill_end - self.prefill_start)*1000:.2f} ms")
        if status_str != "NORMAL":
            if self.queue is not None:
                self.queue.put(("status", status_str))
            return
        if not text:
            return
        if self.stop_requested:
            return
        token_id = payload.token_id
        text_out = text.replace('\n', '\\n')
        logger.debug(f"Received token: {text_out} id: {token_id} (status: {status_str})")
        content_type = getattr(payload, "content_type", 0) or 0
        # REASONING (thinking chain) and TOOL-markup fragments are routed to
        # separate buffers; they do not run through stop-sequence or loop
        # detection, which are answer-text only. All fragment types count
        # toward the max_tokens budget.
        if content_type == 1:  # REASONING
            self.generated_reasoning += text
            self.generated_token_count += 1
            if self.queue is not None:
                self.queue.put(("reasoning", text))
            if self.max_tokens is not None and self.generated_token_count >= self.max_tokens:
                self.finish_reason = "length"
                if not self.stop_requested:
                    self.stop_requested = True
                    self.client.interrupt()
            return
        if content_type == 2:  # TOOL markup
            self.generated_tool_text += text
            self.generated_token_count += 1
            if self.max_tokens is not None and self.generated_token_count >= self.max_tokens:
                self.finish_reason = "length"
                if not self.stop_requested:
                    self.stop_requested = True
                    self.client.interrupt()
            return
        if self.max_tokens is not None and self.generated_token_count >= self.max_tokens:
            self.finish_reason = "length"
            if not self.stop_requested:
                self.stop_requested = True
                self.client.interrupt()
            return
        if self.config.token_pair and token_id in self.config.token_pair:
            if token_id in self._loop_detector:
                self.loop_detected = True
                logger.warning(f"Loop detected at token id {token_id}")
                self.client.interrupt()
                return
            else:
                self._loop_detector[token_id] = True
                logger.debug(f"Token id {token_id} added to loop detector")
        reversd_token_pair = {v:k for k,v in self.config.token_pair.items()} if self.config.token_pair else None
        if reversd_token_pair and token_id in reversd_token_pair:
            loop_token_id = reversd_token_pair[token_id]
            if loop_token_id in self._loop_detector:
                self._loop_detector.pop(loop_token_id)
                logger.debug(f"Token id {loop_token_id} removed from loop detector as closing pair of {token_id}")
                

        emit_text = text
        next_text = self.generated_text + text
        for stop_text in self.stop_sequences:
            stop_index = next_text.find(stop_text)
            if stop_index != -1:
                emit_text = next_text[len(self.generated_text):stop_index]
                self.generated_text = next_text[:stop_index]
                self.generated_token_count += 1
                self.finish_reason = "stop"
                if emit_text and self.queue is not None:
                    self.queue.put(("token", emit_text))
                if not self.stop_requested:
                    self.stop_requested = True
                    self.client.interrupt()
                return

        self.generated_text = next_text
        self.generated_token_count += 1
        if self.queue is not None:
            self.queue.put(("token", emit_text))

        if self.max_tokens is not None and self.generated_token_count >= self.max_tokens:
            self.finish_reason = "length"
            if not self.stop_requested:
                self.stop_requested = True
                self.client.interrupt()

    def _ensure_ready(self):
        """Initialize the LLMSDK client once. Sampling params are passed per-run."""
        if self.client is None:
            if LLMSDK is None:
                raise ImportError("aml_llmsdk is not installed or failed to import")
            self.client = LLMSDK()
        if self.initialized:
            return
        # Build sampler_params from config (runtime overrides handled in run()).
        sampler_params = dict(self.config.sampler_params) if self.config.sampler_params else {}
        self.client.init(
            model_path=self.config.model_path,
            sampling_mode=self.config.sampling_mode,
            sampler_params=sampler_params if sampler_params else None,
            on_token=self._callback,
            on_tool_call=self._tool_callback,
            mmproj_path=self.config.mmproj_path,
            custom_jinja_template=self._chat_template_str,
        )
        # The tool-call callback bridge is auto-registered inside LLMSDK.init()
        # (stop_output=0 so the full token stream stays available for the
        # text-based tool-call fallback in direct_run).
        self.initialized = True

    def _tool_calls_from_sdk(self, sdk_tool_calls: list[dict], tools: Optional[list[dict]]) -> list[dict]:
        """Convert SDK-parsed tool calls (``{"name", "arguments"(JSON str)}``)
        into OpenAI tool-call dicts, filtered by the requested tool names."""
        if not sdk_tool_calls:
            return []
        requested = tool_names(tools) if tools else None
        converted = []
        for tc in sdk_tool_calls:
            name = tc.get("name") or ""
            if not name:
                continue
            if requested and name not in requested:
                continue
            args_raw = tc.get("arguments")
            try:
                arguments = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else {}
                if not isinstance(arguments, dict):
                    arguments = {"input": arguments}
            except (json.JSONDecodeError, TypeError):
                arguments = {"input": args_raw} if args_raw else {}
            converted.append(make_tool_call(str(name), arguments))
        return converted

    def _normalize_tool_calls_for_comparison(self, tool_calls: Optional[list[dict]]) -> Optional[str]:
        """Normalize tool_calls for comparison by parsing & re-serializing
        each function's arguments JSON string with sorted keys."""
        if tool_calls is None:
            return None
        normalized = []
        for tc in tool_calls:
            tc_copy = dict(tc)
            function = tc_copy.get("function")
            if isinstance(function, dict):
                func_copy = dict(function)
                arguments = func_copy.get("arguments")
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                        func_copy["arguments"] = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
                tc_copy["function"] = func_copy
            normalized.append(tc_copy)
        return json.dumps(normalized, sort_keys=True)

    def _detect_continuation(self, messages: list[ChatMessage], skip:int = 0) -> Optional[list[ChatMessage]]:
        """If messages is a strict continuation of the previous conversation,
        return only the delta (new) messages. Otherwise return None."""
        logger.debug("Checking for continuation..., current messages: %d,\n previous messages: %d", len(messages), len(self._previous_messages or []))
        if not self.config.retain_history:
            return None
        if not self._previous_messages:
            return None
        if len(messages) <= len(self._previous_messages):
            return None
        logger.debug("Comparing messages for continuation...")
        for i, prev_msg in enumerate(self._previous_messages[skip:]):
            cur_msg = messages[i + skip]
            if prev_msg.role != cur_msg.role:
                logger.debug("Role mismatch at index %d: \nprevious '%s'\n current '%s'", i, prev_msg.role, cur_msg.role)
                return None
            prev_text = message_text(prev_msg.content)
            cur_text = message_text(cur_msg.content)
            if prev_text != cur_text:
                logger.debug("Content text mismatch at index %d:\nprevious: '%s'\n current: '%s'", i, prev_text, cur_text)
                return None
            prev_tool_calls = prev_msg.tool_calls
            cur_tool_calls = cur_msg.tool_calls
            if prev_tool_calls != cur_tool_calls:
                if prev_tool_calls is None or cur_tool_calls is None:
                    logger.debug("Tool calls mismatch at index %d: \nprevious: '%s'\n current: '%s'", i, prev_tool_calls, cur_tool_calls)
                    return None
                prev_normalized = self._normalize_tool_calls_for_comparison(prev_tool_calls)
                cur_normalized = self._normalize_tool_calls_for_comparison(cur_tool_calls)
                if prev_normalized != cur_normalized:
                    logger.debug("Tool calls mismatch at index %d:\nprevious: '%s'\n current: '%s'", i, prev_normalized, cur_normalized)
                    return None
            if prev_msg.tool_call_id != cur_msg.tool_call_id:
                logger.debug("Tool call ID mismatch at index %d: \nprevious: '%s'\n current: '%s'", i, prev_msg.tool_call_id, cur_msg.tool_call_id)
                return None
            if prev_msg.name != cur_msg.name:
                logger.debug("Name mismatch at index %d: \nprevious: '%s'\n current: '%s'", i, prev_msg.name, cur_msg.name)
                return None
        return messages[len(self._previous_messages):]
    
    def _process_skill_workaround(self, messages: list[ChatMessage]) -> RoutingConfig:
        if not self.config.skill_workaround:
            return None
        if messages is None or len(messages) == 0:
            return None
        # if messages[-1].role == "tool":
        #     return None
        
        msg = messages[0]
        if msg.role != "system":
            return None
        text = message_text(msg.content)
        fixed_prompt, skill_path, skill_lists = preprocess_system_skills(text)
        routing_prompt = build_skill_routing_prompt(skill_lists)

        return RoutingConfig(
            original_prompt=text,
            fixed_prompt=fixed_prompt,
            routing_prompt=routing_prompt,
            path_template=skill_path,
        )
                

    def run(self,
            messages: list[ChatMessage],
            user_data: Optional[str],
            stream_queue: Optional["queue.Queue[object]"] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stop: Optional[Union[str, list[str]]] = None,
            tools: Optional[list[ChatCompletionTool]] = None,
            tool_choice: Optional[Union[str, dict]] = None,
            user_agent: Optional[str] = None):
        if not self.config.skill_workaround:
            return self.direct_run(messages, user_data, stream_queue, temperature, top_p, max_tokens, stop, tools, tool_choice, user_agent)
        else:
            routing_config = self._process_skill_workaround(messages)
            if routing_config is None:
                return self.direct_run(messages, user_data, stream_queue, temperature, top_p, max_tokens, stop, tools, tool_choice, user_agent)
            else:
                apply_tool_hooks(messages, user_agent or "", self.config.model_type)
                
                delta_messages = self._detect_continuation(messages, skip=1) # Skip the first message which is the system prompt containing skill info, since it will be replaced by the routing prompt
                if delta_messages is not None:
                    logger.debug("Continuation detected at skill workaround check, skipping skill workaround and proceeding with normal processing")
                    messages[0] = self._previous_messages[0]  # restore original system prompt for continuation
                    return self.direct_run(messages, user_data, stream_queue, temperature, top_p, max_tokens, stop, tools, tool_choice, user_agent, tool_hooks_enabled=False)
                
                logger.debug(f"Skill workaround enabled, routing config: {routing_config}")
                msg = [ChatMessage(role="system", content=routing_config.routing_prompt)]
                msg.append(messages[-1]) 
                t_router = time.time()
                routing_response = self.direct_run(
                    messages=msg,
                    user_data=user_data,
                    stream_queue=None,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    tools=None,
                    tool_choice=None,
                    user_agent=user_agent,
                    single_turn=True,
                    tool_hooks_enabled=False,
                    template_kwargs={"enable_thinking": False}
                )
                local_model, intent, skill_name = try_decode_routing_response(routing_response.get("text", ""))
                logger.debug(
                    "[skill-routing] decision=%s intent=%s skill=%s router_ms=%.1f input_chars=%d raw=%.200s",
                    "local" if local_model else "cloud", intent, skill_name,
                    (time.time() - t_router) * 1000.0,
                    len(message_text(msg[-1].content)), routing_response.get("text", ""),
                )
                if not local_model or self.config.route_only:
                    logger.debug("[skill-routing] outcome=forward_cloud")
                    raise RoutedResponse(
                        messages=inject_skills_for_forwarding(messages, skill_name),
                        original_messages=messages,
                        dest="cloud")
                if skill_name is not None:
                    routed_path = routing_config.path_template.format(skill_name=skill_name)
                    logger.debug(f"Routing to skill '{skill_name}' with path '{routed_path}'")
                    # In a real implementation, you would now route the request to the new path.
                    # For this example, we'll just return the routing decision in a special format.
                    with open(routed_path, "r") as fp:
                        skill_content = fp.read()
                    skill_dir = os.path.dirname(routed_path)
                    skill_content = get_skill_content(skill_content, skill_dir)
                    logger.debug("[skill-routing] outcome=skill_local skill=%s", skill_name)
                    msg = [ChatMessage(role="system", content=skill_content)]
                    msg.extend(messages[1:])  # Append the rest of the original messages after the new system prompt
                    return self.direct_run(
                        messages=msg,
                        user_data=user_data,
                        stream_queue=stream_queue,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop=stop,
                        tools=tools,
                        tool_choice=tool_choice,
                        user_agent=user_agent,
                        tool_hooks_enabled=False
                    )
                        
                else:
                    logger.debug("No valid skill route found in routing response, proceeding with normal processing")
                    logger.debug("[skill-routing] outcome=local")
                    return self.direct_run(messages, user_data, stream_queue, temperature, top_p, max_tokens, stop, tools, tool_choice, user_agent, tool_hooks_enabled=False)

    def direct_run(self,
            messages: list[ChatMessage],
            user_data: Optional[str],
            stream_queue: Optional["queue.Queue[object]"] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stop: Optional[Union[str, list[str]]] = None,
            tools: Optional[list[ChatCompletionTool]] = None,
            tool_choice: Optional[Union[str, dict]] = None,
            user_agent: Optional[str] = None,
            single_turn: bool = False,
            tool_hooks_enabled: bool = True,
            template_kwargs: dict = {},
            **kwargs):
        normalized_tools = normalize_tools(tools)
        normalized_tool_choice = normalize_tool_choice(tool_choice)
        # Apply tool hooks to process tool message content before prompt building.
        if tool_hooks_enabled:
            apply_tool_hooks(messages, user_agent or "", self.config.model_type)
        with self.lock:
            self._ensure_ready()
            delta_messages = self._detect_continuation(messages)

            # --- Extract and preprocess images for VLM ---

            vlm_images, preprocessed_message = extract_images_from_messages(messages if delta_messages is None else delta_messages)
            if vlm_images and not self.config.mmproj_path:
                logger.warning("Images in request but no mmproj_path configured; ignoring images")

            logger.debug('Messages for prompt building: %s', json.dumps(preprocessed_message, ensure_ascii=False, indent=2, default=_json_default))

            sdk_render = False
            template_kwargs.update(self.config.template_kwargs) 
            if not self._tokenizer_config.get('sdk_render', True) and self._chat_template_str:
                # Make a copy to avoid mutating the input
                # --- Python renders the Jinja2 chat template (disable_chat_template=1) ---
                if delta_messages is not None:
                    logger.debug("Continuation detected, use cached KV cache with Jinja2 template (python render)")
                    msgs_for_template = messages_to_llama_cpp(delta_messages, "")
                    prompt = render_chat_template(
                        self._chat_template_str,
                        messages=msgs_for_template,
                        # tools=normalized_tools,
                        add_generation_prompt=True,
                        bos_token=self._tokenizer_config.get("bos_token", ""),
                        eos_token=self._tokenizer_config.get("eos_token", ""),
                        **template_kwargs
                    )
                else:
                    logger.debug("New chat detected, cleaning KV cache and applying Jinja2 template (python render)")
                    self.client.reset()
                    # Clear default chat template (required by state machine)
                    self.client.set_chat_template("\0", "\0", "\0")
                    messages_system_prompt = system_prompt_from_messages(messages)
                    base_system = messages_system_prompt or self.config.system_prompt
                    merged_system = merge_system_prompt(base_system, normalized_tools, normalized_tool_choice)
                    msgs_for_template = messages_to_llama_cpp(messages, None)
                    prompt = render_chat_template(
                        self._chat_template_str,
                        messages=msgs_for_template,
                        tools=normalized_tools,
                        add_generation_prompt=True,
                        bos_token=self._tokenizer_config.get("bos_token", ""),
                        eos_token=self._tokenizer_config.get("eos_token", ""),
                        **template_kwargs
                    )
                # For logging consistency
                final_system = merged_system if not delta_messages else ""
            else:
                # --- No Python-side template: hand (delta) messages JSON to the
                # SDK and let it render the chat template
                # (AML_LLM_INPUT_MESSAGES + disable_chat_template=0). ---
                sdk_render = True
                # The SDK renders the template, so it must receive the messages
                # where every decoded image is the "@bin:<n>" placeholder that
                # matches the `images` argument — also on a new chat (passing
                # the raw messages would hand it the base64 data URI instead).
                if delta_messages is not None:
                    logger.debug("Continuation detected, passing delta messages JSON to SDK for template rendering")
                    msgs_for_template = messages_to_llama_cpp(preprocessed_message, "")
                else:
                    logger.debug("New chat detected, cleaning KV cache, passing messages JSON to SDK for template rendering")
                    self.client.reset()
                    msgs_for_template = messages_to_llama_cpp(preprocessed_message, None)
                # Must stay legal JSON: the SDK parses this as the messages input.
                prompt = json.dumps(msgs_for_template, ensure_ascii=False)
                final_system = ""
            self.queue = stream_queue
            self.generated_text = ""
            self.generated_reasoning = ""
            self.generated_tool_text = ""
            self.generated_token_count = 0
            self.stop_sequences = merge_stop_sequences(self.config.stop, stop)
            self.max_tokens = max_tokens
            self.finish_reason = "stop"
            self.stop_requested = False
            self._sdk_tool_calls = []
            # print(f"Final system prompt:\n{final_system}\n--- End of system prompt ---")
            if normalized_tools and not delta_messages:
                logger.debug(
                    "adla tools request "
                    + json.dumps(
                        {
                            "tool_choice": normalized_tool_choice,
                            "tool_count": len(normalized_tools),
                            "tool_names": tool_names(normalized_tools),
                            "message_count": len(messages),
                            "system_prompt": len(final_system),
                        },
                        ensure_ascii=False,
                    )
                )
            try:
                self.prefill_start = time.perf_counter()
                self.prefill_end = 0
                self.decode_end = 0
                self.loop_detected = False
                self._loop_detector.clear()
                # Build per-run sampler_params from runtime overrides only (SDK merges with init defaults).
                # Only include keys recognised by the LLMSDK sampler_params schema.
                _VALID_SAMPLER_KEYS = frozenset({
                    "temp", "dynatemp_range", "dynatemp_exponent", "top_k", "top_p",
                    "min_keep", "min_p", "typ_p", "top_n_sigma", "xtc_probability",
                    "xtc_threshold", "penalty_last_n", "penalty_repeat", "penalty_freq",
                    "penalty_present", "dry_multiplier", "dry_base", "dry_allowed_length",
                    "dry_penalty_last_n", "dry_sequence_breakers", "mirostat", "mirostat_tau",
                    "mirostat_eta", "adaptive_target", "adaptive_decay", "seed", "n_ctx_train",
                    "ignore_eos", "no_perf", "token_eot", "token_eos", "logit_bias",
                    "samplers", "grammar_gbnf", "grammar_root", "grammar_lazy",
                    "grammar_triggers", "grammar_trigger_tokens", "reasoning_budget_start",
                    "reasoning_budget_end", "reasoning_budget_forced", "reasoning_budget_tokens",
                    "reasoning_budget_message", "reasoning_control",
                })
                run_sampler_params = {}
                if temperature is not None:
                    run_sampler_params["temp"] = float(temperature)
                if top_p is not None:
                    run_sampler_params["top_p"] = float(top_p)
                for k, v in kwargs.items():
                    if v is not None and k in _VALID_SAMPLER_KEYS:
                        run_sampler_params[k] = v
                result = self.client.run(
                    prompt_input=prompt,
                    retain_history=True,
                    userdata=user_data,
                    images=vlm_images if vlm_images else None,
                    img_content=self.config.image_pad,
                    disable_chat_template=0 if sdk_render else 1,
                    messages_mode=sdk_render,
                    tool_schemas=normalized_tools if normalized_tools else None,
                    sampler_params=run_sampler_params if run_sampler_params else None,
                    extra_template_params=template_kwargs if sdk_render and template_kwargs else None,
                )
                self.decode_end = time.perf_counter()
                if self.generated_text:
                    result["text"] = self.generated_text
                    result["token_count"] = self.generated_token_count
                elif self.stop_sequences:
                    text = result.get("text", "")
                    for stop_text in self.stop_sequences:
                        stop_index = text.find(stop_text)
                        if stop_index != -1:
                            result["text"] = text[:stop_index]
                            self.finish_reason = "stop"
                            break
                # Reasoning (thinking chain) is delivered separately by the SDK
                # (content_type=REASONING); prefer the streamed accumulation,
                # fall back to the SDK's concatenated reasoning, then to the
                # legacy regex extraction from the answer text.
                if self.generated_reasoning:
                    result["reasoning"] = self.generated_reasoning
                elif result.get("reasoning"):
                    result["reasoning"] = result["reasoning"]
                if self.max_tokens is not None and result.get("token_count", 0) > self.max_tokens:
                    result["token_count"] = self.max_tokens
                    self.finish_reason = "length"
                result_text = result.get("text", "")
                tool_calls = []
                if normalized_tools:
                    sdk_tool_calls = [{"name": n, "arguments": a} for n, a in self._sdk_tool_calls]
                    if not sdk_tool_calls:
                        sdk_tool_calls = result.get("tool_calls") or []
                    if sdk_tool_calls:
                        tool_calls = self._tool_calls_from_sdk(sdk_tool_calls, normalized_tools)
                        logger.debug(
                            "adla sdk tool callback "
                            + json.dumps(
                                {
                                    "sdk_tool_calls": len(sdk_tool_calls),
                                    "parsed_tool_calls": len(tool_calls),
                                    "tool_choice": normalized_tool_choice,
                                },
                                ensure_ascii=False,
                            )
                        )
                    if not tool_calls:
                        # Fallback: parse tool-call markup. With content_type
                        # tagging the markup lives in tool_text; otherwise it
                        # may still be embedded in the answer text.
                        fallback_text = self.generated_tool_text or result_text
                        tool_calls = fallback_tool_calls_from_text(fallback_text, normalized_tools, **kwargs)
                    logger.debug(
                        "adla tool debug "
                        + json.dumps(
                            {
                                "tool_choice": normalized_tool_choice,
                                "finish_reason": self.finish_reason,
                                # "raw_content_preview": result_text[:500],
                                "parsed_tool_calls": [
                                    {
                                        "id": tc.get("id"),
                                        "name": (tc.get("function") or {}).get("name"),
                                    }
                                    for tc in tool_calls
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                # Prefer the SDK-tagged reasoning channel; fall back to the
                # legacy regex extraction from the answer text.
                thinking_content = result.get("reasoning") or extract_thinking_content(result_text)
                if tool_calls:
                    self.finish_reason = "tool_calls"
                    result["text"] = text_without_tool_markup(result_text) or None
                else:
                    result["text"] = text_without_tool_markup(result_text)
                result["tool_calls"] = tool_calls
                result["finish_reason"] = self.finish_reason
                assistant_content = result.get("text") or None
                assistant_msg = ChatMessage(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=tool_calls if tool_calls else None,
                    reasoning_content=thinking_content
                )
                if not single_turn:
                    self._previous_messages = list(messages) + [assistant_msg]
                return result
            finally:
                self.queue = None
                if not self.config.retain_history:
                    self.client.reset()
    
    def interrupt(self):
        if self.client is not None:
            self.client.interrupt()
