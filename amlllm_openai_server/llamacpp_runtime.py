from .types import ModelConfig, ChatMessage, ChatCompletionTool
from .model_runtime import BaseModelRuntime
from .llm_utils import merge_system_prompt, normalize_tools, normalize_tool_choice, should_use_prompt_only_tools 
from .llm_utils import merge_stop_sequences, use_raw_tool_call_passthrough, messages_to_llama_cpp
from .llm_utils import text_without_tool_markup, clean_template_artifacts, fallback_tool_calls_from_text
from .llm_utils import message_debug_summary

from typing import Optional, Union
import json
import logging
import queue


logger = logging.getLogger("llm.llamacpp")

class LlamaCppModelRuntime(BaseModelRuntime):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = None

    def _ensure_ready(self):
        if self.client is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "Model backend 'llama.cpp' requires the 'llama-cpp-python' package to be installed."
            ) from exc

        init_kwargs = {
            "model_path": self.config.model_path,
            "n_ctx": self.config.context_size,
            "n_gpu_layers": self.config.n_gpu_layers,
            "verbose": self.config.verbose,
        }
        if self.config.threads > 0:
            init_kwargs["n_threads"] = self.config.threads
        if self.config.chat_format:
            init_kwargs["chat_format"] = self.config.chat_format
        self.client = Llama(**init_kwargs)

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.client is not None and hasattr(self.client, "tokenize"):
            try:
                return len(self.client.tokenize(text.encode("utf-8")))
            except Exception:
                return len(text)
        return len(text)

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
        del user_data, user_agent
        with self.lock:
            self._ensure_ready()
            request_messages = messages_to_llama_cpp(
                messages,
                merge_system_prompt(
                    self.config.system_prompt,
                    normalize_tools(tools),
                    normalize_tool_choice(tool_choice),
                ),
            )
            normalized_tools = normalize_tools(tools)
            normalized_tool_choice = normalize_tool_choice(tool_choice)
            raw_tool_call_passthrough = use_raw_tool_call_passthrough(self.config)
            prompt_only_tools = should_use_prompt_only_tools(self.config, normalized_tools)
            sp = self.config.sampler_params or {}
            request_kwargs = {
                "messages": request_messages,
                "temperature": float(sp.get("temp", 1.0)) if temperature is None else float(temperature),
                "top_p": float(sp.get("top_p", 0.9)) if top_p is None else float(top_p),
                "max_tokens": max_tokens,
                "stop": merge_stop_sequences(self.config.stop, stop) or None,
                "stream": stream_queue is not None,
                "tools": None if prompt_only_tools else normalized_tools,
                "tool_choice": None if prompt_only_tools else normalized_tool_choice,
            }
            self.finish_reason = "stop"
            if normalized_tools:
                logger.debug(
                    "llama.cpp request debug "
                    + json.dumps(
                        {
                            "prompt_only_tools": prompt_only_tools,
                            "raw_tool_call_passthrough": raw_tool_call_passthrough,
                            "tool_choice": normalized_tool_choice,
                            "message_count": len(request_messages),
                            "messages": message_debug_summary(request_messages),
                        },
                        ensure_ascii=False,
                    )
                )

            if stream_queue is not None:
                if normalized_tools:
                    request_kwargs["stream"] = False
                    result = self.client.create_chat_completion(**request_kwargs)
                    choice = (result.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    text = message.get("content", "")
                    tool_calls = [] if raw_tool_call_passthrough else (message.get("tool_calls") or [])
                    if not tool_calls and not raw_tool_call_passthrough:
                        tool_calls = fallback_tool_calls_from_text(text, normalized_tools)
                    self.finish_reason = str(choice.get("finish_reason") or "stop")
                    logger.debug(
                        "llama.cpp tool debug "
                        + json.dumps(
                            {
                                "prompt_only_tools": prompt_only_tools,
                                "raw_tool_call_passthrough": raw_tool_call_passthrough,
                                "tool_choice": normalized_tool_choice,
                                "finish_reason": self.finish_reason,
                                "raw_content": text,
                                "raw_tool_calls": message.get("tool_calls") or [],
                                "parsed_tool_calls": tool_calls,
                            },
                            ensure_ascii=False,
                        )
                    )
                    if tool_calls:
                        self.finish_reason = "tool_calls"
                        text = text_without_tool_markup(text)
                    else:
                        text = clean_template_artifacts(text)
                    return {
                        "text": text,
                        "tool_calls": tool_calls,
                        "token_count": int(
                            (result.get("usage") or {}).get("completion_tokens", self._count_tokens(text))
                        ),
                        "finish_reason": self.finish_reason,
                    }
                text_parts = []
                completion_tokens = 0
                for chunk in self.client.create_chat_completion(**request_kwargs):
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        text_parts.append(content)
                        completion_tokens += self._count_tokens(content)
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        self.finish_reason = str(finish_reason)
                return {
                    "text": clean_template_artifacts("".join(text_parts)),
                    "token_count": completion_tokens,
                    "finish_reason": self.finish_reason,
                }

            result = self.client.create_chat_completion(**request_kwargs)
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content", "")
            tool_calls = [] if raw_tool_call_passthrough else (message.get("tool_calls") or [])
            if not tool_calls and normalized_tools and not raw_tool_call_passthrough:
                tool_calls = fallback_tool_calls_from_text(text, normalized_tools)
            usage = result.get("usage") or {}
            self.finish_reason = str(choice.get("finish_reason") or "stop")
            if normalized_tools:
                logger.debug(
                    "llama.cpp tool debug "
                    + json.dumps(
                    {
                        "prompt_only_tools": prompt_only_tools,
                        "raw_tool_call_passthrough": raw_tool_call_passthrough,
                        "tool_choice": normalized_tool_choice,
                        "finish_reason": self.finish_reason,
                        "raw_content": text,
                        "raw_tool_calls": message.get("tool_calls") or [],
                        "parsed_tool_calls": tool_calls,
                        },
                        ensure_ascii=False,
                    )
                )
            if tool_calls:
                self.finish_reason = "tool_calls"
                text = text_without_tool_markup(text) or None
            else:
                text = clean_template_artifacts(text)
            return {
                "text": text,
                "tool_calls": tool_calls,
                "token_count": int(usage.get("completion_tokens", self._count_tokens(text))),
                "finish_reason": self.finish_reason,
            }
