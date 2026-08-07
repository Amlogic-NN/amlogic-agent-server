# -*- coding: utf-8 -*-
"""Runners for local and cloud model inference in the evaluation framework."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union
import traceback


from .schema import CloudApiConfig


from .types import ModelConfig, ChatMessage, ChatCompletionTool  # noqa: E402
from .aml_runtime import AmlLlmModelRuntime  # noqa: E402

logger = logging.getLogger("tools.runners")


class LocalModelRunner:
    """Runner that invokes ``AmlLlmModelRuntime.direct_run`` for each conversation."""

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config
        self._runtime: Optional[AmlLlmModelRuntime] = None

    @property
    def model_name(self) -> str:
        """Human-readable model identifier: basename of the model path."""
        return os.path.basename(self._model_config.model_path)

    def _ensure_runtime(self, max_tokens: int) -> AmlLlmModelRuntime:
        """Create or reuse the runtime, clamping context_size."""
        self._model_config.context_size = max(
            self._model_config.context_size, max_tokens
        )
        if self._runtime is None:
            self._runtime = AmlLlmModelRuntime(self._model_config)
        return self._runtime

    def reset_session(self):
        """Reset the KV cache so the next conversation is independent."""
        if self._runtime is not None:
            self._runtime._previous_messages = None
            self._runtime.stop_requested = False
            self._runtime.generated_text = ""
            self._runtime.generated_token_count = 0

    def run_single(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[ChatCompletionTool]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, list[str]]] = None,
    ) -> dict:
        """Run one inference and return ``{text, tool_calls, finish_reason, token_count, error, latency_ms}``."""
        runtime = self._ensure_runtime(max_tokens or 2048)
        self.reset_session()

        start = time.perf_counter()
        try:
            result = runtime.direct_run(
                messages=messages,
                user_data=None,
                stream_queue=None,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                user_agent=None,
                single_turn=True,
                tool_hooks_enabled=False,
                exception=True,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            looping = runtime.loop_detected
            return {
                "text": result.get("text"),
                "tool_calls": result.get("tool_calls") or [],
                "finish_reason": result.get("finish_reason", "stop"),
                "token_count": result.get("token_count", 0),
                "error": "Loop detected" if looping else None,
                "latency_ms": latency_ms,
                "prefill_ms": (runtime.prefill_end - runtime.prefill_start) * 1000.0 if runtime.prefill_end and runtime.prefill_start else None,
                "decode_ms": (runtime.decode_end - runtime.prefill_end) * 1000.0 if runtime.decode_end and runtime.prefill_end else None
            }
        except KeyboardInterrupt as exc:
            raise exc
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            # Capture raw model output for debugging.
            # Prefer _last_raw_response_text (set immediately after client.run()
            # and before tool parsing, so it survives exception propagation).
            # Fall back to generated_text for backward compatibility.
            raw_text = (
                getattr(runtime, "_last_raw_response_text", None)
                or getattr(runtime, "generated_text", "")
                or None
            )
            traceback.print_exc()
            logger.warning("Local model run failed: %s | raw_text preview: %.200s", exc, raw_text or "(empty)")
            return {
                "text": raw_text,
                "tool_calls": [],
                "finish_reason": "error",
                "token_count": runtime.generated_token_count if hasattr(runtime, "generated_token_count") else 0,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": latency_ms,
                "prefill_ms": (runtime.prefill_end - runtime.prefill_start) * 1000.0 if runtime.prefill_end and runtime.prefill_start else None,
                "decode_ms": (runtime.decode_end - runtime.prefill_end) * 1000.0 if runtime.decode_end and runtime.prefill_end else None,
            }

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config


class CloudModelRunner:
    """Runner that calls an OpenAI-compatible cloud API."""

    def __init__(self, cloud_config: CloudApiConfig):
        self._config = cloud_config
        self._client = None
        self._imports_checked = False

    @property
    def model_name(self) -> str:
        """Human-readable model identifier from cloud config."""
        return self._config.model

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._imports_checked:
            try:
                import openai  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Cloud runner requires 'openai' package. "
                    "Install it with: pip install openai"
                )
            self._imports_checked = True
        import openai
        self._client = openai.OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            timeout=600.0,
            max_retries=2,
        )

    def run_single(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[ChatCompletionTool]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, list[str]]] = None,
    ) -> dict:
        """Run one cloud inference synchronously."""
        self._ensure_client()

        # Build messages payload
        payload_messages = [
            msg.model_dump(exclude_none=True) for msg in messages
        ]

        # Build tools payload
        payload_tools = None
        if tools:
            payload_tools = [
                t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t
                for t in tools
            ]

        kwargs: dict = {
            "model": self._config.model,
            "messages": payload_messages,
        }
        if payload_tools:
            kwargs["tools"] = payload_tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop is not None:
            kwargs["stop"] = stop
        if self._config.thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000.0

            choice = response.choices[0]
            message = choice.message

            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": tc.type or "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

            return {
                "text": message.content,
                "tool_calls": tool_calls,
                "finish_reason": choice.finish_reason or "stop",
                "token_count": response.usage.completion_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "error": None,
                "latency_ms": latency_ms,
                "prefill_ms": response.timings.get("prompt_ms") if hasattr(response, "timings") and response.timings else None,
                "decode_ms": response.timings.get("predicted_ms") if hasattr(response, "timings") and response.timings else None,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("Cloud model run failed: %s", exc)
            return {
                "text": None,
                "tool_calls": [],
                "finish_reason": "error",
                "token_count": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": latency_ms,
                "prefill_ms": None,
                "decode_ms": None,
            }
