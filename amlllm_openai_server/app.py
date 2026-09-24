# -*- coding: utf-8 -*-
"""Simple OpenAI-compatible proxy for AMLLLM."""

from __future__ import annotations

import json
import logging
import os
import queue
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import time
import uuid
import yaml

from typing import List, Optional, Tuple, Union, Annotated
from enum import Enum
import traceback

import httpx

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, RedirectResponse
from .types import ModelConfig, ServerConfig, ChatCompletionRequest, RoutedResponse, is_asr_model
from .model_runtime import BaseModelRuntime
from .tools_hook import  get_url, set_base_url
from .aml_runtime import AmlLlmModelRuntime
from .llamacpp_runtime import LlamaCppModelRuntime
from .asr_runtime import AsrModelRuntime
from .llm_utils import resolve_path, load_model, estimate_token_count, inject_skills_for_forwarding


logger = logging.getLogger("uvicorn.error")



def now_ts() -> int:
    return int(time.time())


def load_proxy_config(config_path: str) -> Tuple[ServerConfig, list[ModelConfig]]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Server config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    server_raw = raw.get("server", {})
    models_raw = raw.get("models", {})
    root_dir = Path(resolve_path(path.parent, str(models_raw.get("root_dir", "../models"))))
    enabled = models_raw.get("enabled", ["default"])
    if not isinstance(enabled, list) or not enabled:
        raise ValueError("'models.enabled' must be a non-empty list")

    upstream_raw = raw.get("upstream", {}) or {}
    server_raw_cors = server_raw.get("cors_origins", "*")
    server = ServerConfig(
        host=str(server_raw.get("host", "0.0.0.0")),
        port=int(server_raw.get("port", 8000)),
        api_key=server_raw.get("api_key"),
        log_level=str(server_raw.get("log_level", "info")),
        title=str(server_raw.get("title", "AMLLLM OneAPI Proxy")),
        version=str(server_raw.get("version", "0.2.0")),
        upstream_base_url=upstream_raw.get("base_url"),
        upstream_api_key=upstream_raw.get("api_key"),
        upstream_model=upstream_raw.get("model"),
        skill_injection=bool(upstream_raw.get("skill_injection", False)),
        force_upstream=bool(upstream_raw.get("force_upstream", False)),
        cors_origins=str(server_raw_cors) if server_raw_cors else "*",
        reject_when_busy=bool(models_raw.get("reject_when_busy", True)),
    )

    models = []
    for item in enabled:
        model_dir = Path(str(item))
        if not model_dir.is_absolute():
            model_dir = root_dir / model_dir
        models.append(load_model(model_dir.resolve()))
    return server, models


# Streaming loop cadence: poll the client for a disconnect often enough to stop
# an abandoned run promptly, but keep the SSE keep-alive cadence as before.
STREAM_POLL_INTERVAL = 0.2         # seconds between disconnect polls / queue reads
STREAM_HEARTBEAT_INTERVAL = 10.0   # seconds between empty-delta keep-alives


def _client_disconnected(http_request: Request) -> bool:
    """Best-effort client-disconnect check from a *sync* streaming generator.

    ``Request.is_disconnected()`` is a coroutine, and this generator is sync, so
    it cannot be awaited directly — calling it without awaiting returns a
    (always truthy) coroutine object. Starlette iterates sync generators in an
    anyio worker thread, where ``anyio.from_thread.run`` can await it; outside
    such a thread we fall back to the ``_is_disconnected`` flag Starlette sets
    when an awaited check saw ``http.disconnect``.
    """
    try:
        import anyio
        return bool(anyio.from_thread.run(http_request.is_disconnected))
    except Exception:
        return bool(getattr(http_request, "_is_disconnected", False))


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "value"):
            return str(obj.value)
        if isinstance(obj, Enum):
            return obj.value      # 或 obj.name
        return super().default(obj)


class ProxyRuntime:
    def __init__(self, models: List[ModelConfig], server_config: ServerConfig):
        self.server_config = server_config
        self.api_key = server_config.api_key
        llm_models = [model for model in models if not is_asr_model(model.model_type)]
        asr_models = [model for model in models if is_asr_model(model.model_type)]
        self.models = {model.name: _create_model_runtime(model) for model in llm_models}
        self.asr_models = {model.name: AsrModelRuntime(model) for model in asr_models}
        self.model_configs = {model.name: model for model in models}
        if not self.models and not self.asr_models:
            raise ValueError("No models configured")

    def authorize(self, authorization: Optional[str]):
        if not self.api_key:
            return
        if authorization != f"Bearer {self.api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    def get_model(self, model_name: str) -> BaseModelRuntime:
        if model_name in self.asr_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} is ASR; use POST /v1/audio/transcriptions",
            )
        model = self.models.get(model_name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
        return model

    def get_asr_model(self, model_name: str) -> AsrModelRuntime:
        asr = self.asr_models.get(model_name)
        if asr is not None:
            return asr
        if model_name in self.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} is not an ASR model",
            )
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    def list_models(self) -> List[dict]:
        created = now_ts()
        return [
            {
                "id": model.name,
                "object": "model",
                "created": created,
                "owned_by": "amlllm",
                "metadata": model.metadata,
            }
            for model in self.model_configs.values()
        ]


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _sse_empty_delta(request_id: str, created: int, model_name: str) -> str:
    return _sse({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"content": ""},
            "finish_reason": None,
        }],
    })


def _usage(result: dict) -> dict:
    token_count = result.get("token_count")
    text = result.get("text", "")
    completion_tokens = token_count if token_count is not None else len(text)
    return {
        "prompt_tokens": 0,
        "completion_tokens": completion_tokens,
        "total_tokens": completion_tokens,
    }


def _finish_reason(result: dict) -> str:
    return str(result.get("finish_reason", "stop"))


def _response_message(result: dict) -> dict:
    message = {
        "role": "assistant",
        "content": result.get("text", ""),
    }
    reasoning = result.get("reasoning")
    if reasoning:
        message["reasoning_content"] = reasoning
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _stream_tool_call_events(request_id: str, created: int, model_name: str, result: dict):
    yield _sse({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    })

    for index, tool_call in enumerate(result.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        yield _sse({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": index,
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": function.get("name"),
                            "arguments": function.get("arguments", ""),
                        },
                    }],
                },
                "finish_reason": None,
            }],
        })

    yield _sse({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": result.get("finish_reason", "tool_calls"),
        }],
    })
    yield "data: [DONE]\n\n"


def _create_model_runtime(model: ModelConfig) -> BaseModelRuntime:
    backend = (model.backend or "adla").lower()
    if backend == "adla":
        return AmlLlmModelRuntime(model)
    if backend in ("llama.cpp", "llama_cpp", "llamacpp"):
        return LlamaCppModelRuntime(model)
    raise ValueError(f"Unsupported model backend: {model.backend}")


def _build_routed_response(
    messages: List[ChatMessage],
    dest: str,
) -> RoutedResponse:
    """Build a RoutedResponse with the given messages.

    Skill injection (if enabled) is deferred to :func:`_forward_to_upstream`
    so that the lower-level runtime does not need to know about it.
    """
    return RoutedResponse(messages=messages, original_messages=messages, dest=dest)


def _forward_to_upstream(
    routed: RoutedResponse,
    request: ChatCompletionRequest,
    server_config: ServerConfig,
) -> dict:
    """Forward a non-streaming request to the upstream API and return the response dict.

    If ``server_config.skill_injection`` is enabled, skill content is resolved
    from ``routed.original_messages`` before forwarding.
    """
    if not server_config.upstream_base_url:
        raise HTTPException(status_code=503, detail="No upstream configured")

    url = server_config.upstream_base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {server_config.upstream_api_key or ''}",
        "Content-Type": "application/json",
    }

    # Apply skill injection if configured — use original_messages as the base
    # so that the forwarding layer owns the injection decision.
    messages = routed.original_messages
    if server_config.skill_injection:
        if routed.messages is None:
            messages = inject_skills_for_forwarding(messages)
        else:
            messages = routed.messages

    body = {
        "model": server_config.upstream_model or request.model,
        "messages": [msg.model_dump(exclude_none=True) for msg in messages],
        "stream": False,
    }
    for key in ("temperature", "top_p", "max_tokens", "tools", "tool_choice"):
        val = getattr(request, key, None)
        if val is not None:
            if isinstance(val, list):
                body[key] = [v.model_dump(exclude_none=True) if hasattr(v, "model_dump") else v for v in val]
            else:
                body[key] = val.model_dump(exclude_none=True) if hasattr(val, "model_dump") else val

    logger.info("Forwarding to upstream: %s", url)
    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Upstream returned error: %s %s", exc.response.status_code, exc.response.text[:500])
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream error: {exc.response.text[:500]}",
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Upstream request failed: %s", str(exc))
        raise HTTPException(status_code=502, detail=f"Upstream unreachable: {str(exc)}") from exc


def _forward_to_upstream_stream(
    routed: RoutedResponse,
    request: ChatCompletionRequest,
    server_config: ServerConfig,
):
    """Forward a streaming request to the upstream API, yielding SSE chunks.

    If ``server_config.skill_injection`` is enabled, skill content is resolved
    from ``routed.original_messages`` before forwarding.
    """
    if not server_config.upstream_base_url:
        raise HTTPException(status_code=503, detail="No upstream configured")

    url = server_config.upstream_base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {server_config.upstream_api_key or ''}",
        "Content-Type": "application/json",
    }

    # Apply skill injection if configured — use original_messages as the base
    # so that the forwarding layer owns the injection decision.
    messages = routed.original_messages
    if server_config.skill_injection:
        if routed.messages is None:
            messages = inject_skills_for_forwarding(messages)
        else:
            messages = routed.messages

    body = {
        "model": server_config.upstream_model or request.model,
        "messages": [msg.model_dump(exclude_none=True) for msg in messages],
        "stream": True,
    }
    for key in ("temperature", "top_p", "max_tokens", "tools", "tool_choice"):
        val = getattr(request, key, None)
        if val is not None:
            if isinstance(val, list):
                body[key] = [v.model_dump(exclude_none=True) if hasattr(v, "model_dump") else v for v in val]
            else:
                body[key] = val.model_dump(exclude_none=True) if hasattr(val, "model_dump") else val

    logger.info("Forwarding stream to upstream: %s (model=%s)", url, body.get("model"))
    try:
        with httpx.stream("POST", url, json=body, headers=headers, timeout=300.0) as resp:
            resp.raise_for_status()
            logger.debug("Upstream stream response status=%d", resp.status_code)
            chunk_count = 0
            for line in resp.iter_lines():
                chunk_count += 1
                if line:
                    yield line + "\n"
                else:
                    yield "\n"
            logger.info("Upstream stream finished, %d chunks received", chunk_count)
    except httpx.HTTPStatusError as exc:
        logger.error("Upstream stream error: %s %s", exc.response.status_code, exc.response.text[:500])
        yield _sse({"error": {"message": f"Upstream error: {exc.response.status_code}", "type": "upstream_error"}})
        yield "data: [DONE]\n\n"
    except httpx.RequestError as exc:
        logger.error("Upstream stream request failed: %s", str(exc))
        yield _sse({"error": {"message": f"Upstream unreachable: {str(exc)}", "type": "upstream_error"}})
        yield "data: [DONE]\n\n"


def create_app(config_path: str,
               api_key: Optional[str] = None,
               host: Optional[str] = None,
               port: Optional[int] = None,
               log_level: Optional[str] = None) -> Tuple[FastAPI, ServerConfig]:
    server_config, model_configs = load_proxy_config(config_path)
    if api_key is not None:
        server_config.api_key = api_key
    if host is not None:
        server_config.host = host
    if port is not None:
        server_config.port = port
    if log_level is not None:
        server_config.log_level = log_level

    runtime = ProxyRuntime(model_configs, server_config)

    # Configure CORS
    cors_origins_list = [origin.strip() for origin in server_config.cors_origins.split(",")]
    

    # 设置短链接的基础 URL，使 encode_url 返回完整可访问的重定向地址
    set_base_url(f"http://{server_config.host}:{server_config.port}")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Graceful shutdown: interrupt all LLMSDK inference loops."""
        try:
            yield
        finally:
            for model in runtime.models.values():
                try:
                    model.interrupt()
                except Exception:
                    pass
            for asr in runtime.asr_models.values():
                try:
                    asr.close()
                except Exception:
                    pass

    app = FastAPI(title=server_config.title, version=server_config.version, lifespan=lifespan)

    app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/r/{short_code}")
    def redirect_short_url(short_code: str):
        """将短代码重定向到原始 URL。"""
        original_url = get_url(short_code)
        if original_url is None:
            raise HTTPException(status_code=404, detail=f"Short URL not found: {short_code}")
        return RedirectResponse(url=original_url, status_code=307)

    @app.get("/v1/models")
    def list_models(authorization: Annotated[Optional[str], Header()] = None):
        runtime.authorize(authorization)
        return {"object": "list", "data": runtime.list_models()}

    @app.post("/v1/audio/transcriptions")
    def audio_transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Annotated[Optional[str], Form()] = None,
        task: Annotated[Optional[str], Form()] = None,
        response_format: Annotated[Optional[str], Form()] = "json",
        authorization: Annotated[Optional[str], Header()] = None,
    ):
        runtime.authorize(authorization)
        asr = runtime.get_asr_model(model)
        # One session per model: a transcription cannot be cancelled by the SDK,
        # so a second request while one is running is rejected instead of queued.
        reserved = False
        if runtime.server_config.reject_when_busy:
            reserved = asr.try_begin()
            if not reserved:
                logger.info("Model %s is busy; rejecting transcription with 429", model)
                raise HTTPException(
                    status_code=429,
                    detail=f"Model {model} is busy (one session per model)")
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
            if not tmp_path or os.path.getsize(tmp_path) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file")
            result = asr.transcribe(tmp_path, language=language, task=task)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(str(exc) + "\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if reserved:
                asr.end()
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        fmt = (response_format or "json").lower()
        if fmt == "text":
            return PlainTextResponse(result.get("text") or "")
        return JSONResponse({"text": result.get("text") or ""})

    @app.post("/v1/chat/completions")
    def chat_completions(http_request: Request,
                         request: ChatCompletionRequest,
                         user_agent: Annotated[str | None, Header()] = None,
                         authorization: Annotated[Optional[str], Header()] = None,
                         ):
        runtime.authorize(authorization)

        logger.debug(
            "chat.completions request "
            + request.model_dump_json(ensure_ascii=False),
        )
        model = runtime.get_model(request.model)

        # --- context-length pre-check: route to upstream if too long ---
        if runtime.server_config.upstream_base_url:
            estimated_tokens = estimate_token_count(request.messages)
            if estimated_tokens > (model.config.context_size // 4 * 3) or runtime.server_config.force_upstream:
                logger.info(
                    "Estimated %d tokens exceeds context_size %d, routing to upstream",
                    estimated_tokens, model.config.context_size,
                )
                routed = _build_routed_response(
                    messages=request.messages,
                    dest=runtime.server_config.upstream_model or request.model,
                )
                if request.stream:
                    return StreamingResponse(
                        _forward_to_upstream_stream(routed, request, runtime.server_config),
                        media_type="text/event-stream",
                    )
                try:
                    upstream_result = _forward_to_upstream(routed, request, runtime.server_config)
                except HTTPException:
                    raise
                return JSONResponse({
                    "id": upstream_result.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                    "object": "chat.completion",
                    "created": upstream_result.get("created", now_ts()),
                    "model": upstream_result.get("model", request.model),
                    "choices": upstream_result.get("choices", []),
                    "usage": upstream_result.get("usage", {}),
                })

        # --- one session per model: reserve it, or answer 429 ---
        reserved = False
        if runtime.server_config.reject_when_busy:
            reserved = model.try_begin()
            if not reserved:
                logger.info("Model %s is busy; rejecting request %s with 429",
                            request.model, request.model)
                raise HTTPException(
                    status_code=429,
                    detail=f"Model {request.model} is busy (one session per model)")

        if request.stream:
            token_queue: "queue.Queue[object]" = queue.Queue()
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = now_ts()

            def worker():
                logger.debug("Streaming worker started for %s", request_id)
                try:
                    result = model.run(
                        messages=request.messages,
                        user_data=request.user,
                        stream_queue=token_queue,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                        stop=request.stop,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        user_agent=user_agent,
                    )
                    has_tool_calls = bool(result.get("tool_calls"))
                    logger.debug("model.run completed for %s, has_tool_calls=%s, content_len=%d",
                                 request_id, has_tool_calls, len(result.get("text") or ""))
                    token_queue.put(("result", result))
                except RoutedResponse as routed:
                    # On runtime-triggered route, forward to upstream (non-streaming)
                    # and put the result as a normal completion so the event stream
                    # can deliver it as a final chunk.
                    logger.info("Runtime routed to upstream (streaming fallback) for %s", request_id)
                    try:
                        upstream_result = _forward_to_upstream(routed, request, runtime.server_config)
                        token_queue.put(("upstream_result", upstream_result))
                    except Exception as exc:
                        logger.error("Upstream forwarding failed for %s: %s", request_id, str(exc))
                        token_queue.put(("error", str(exc)))
                except Exception as exc:
                    logger.error("Streaming worker failed for %s: %s\n Traceback: %s", request_id, str(exc), traceback.format_exc())
                    token_queue.put(("error", str(exc)))
                finally:
                    if reserved:
                        model.end()          # release the per-model session
                    logger.debug("Streaming worker finished for %s, finish_reason=%s", request_id, model.finish_reason)
                    token_queue.put(("done", model.finish_reason))

            threading.Thread(target=worker, daemon=True).start()

            # A client that goes away must not keep the model busy: Starlette
            # cancels (and closes) this generator on disconnect, so the run is
            # interrupted both when we notice the disconnect and on any early
            # teardown of the generator.
            stream_state = {"finished": False, "interrupted": False}

            def stop_run(reason: str) -> None:
                if stream_state["interrupted"]:
                    return
                stream_state["interrupted"] = True
                logger.info("Interrupting the run for %s (%s)", request_id, reason)
                try:
                    model.interrupt()
                except Exception as exc:  # never mask the original reason
                    logger.warning("interrupt for %s failed: %s", request_id, exc)

            def event_stream():
                logger.info("Streaming response started for %s (model=%s)", request_id, request.model)
                started = False
                empty_delta_count = 0
                token_streamed = False
                yield _sse({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }],
                })
                started = True
                last_heartbeat = time.monotonic()
                while True:
                    if _client_disconnected(http_request):
                        logger.info("Client disconnected, stopping streaming for %s", request_id)
                        # Stop the running inference, not just the response: the
                        # model stays busy (and answers 429) until the SDK run
                        # returns, so an abandoned client must not hold the session.
                        stop_run("client disconnected")
                        break
                    try:
                        item_type, value = token_queue.get(timeout=STREAM_POLL_INTERVAL)
                        logger.debug("event_stream got item_type=%s for %s", item_type, request_id)
                    except queue.Empty:
                        now = time.monotonic()
                        if now - last_heartbeat >= STREAM_HEARTBEAT_INTERVAL:
                            last_heartbeat = now
                            empty_delta_count += 1
                            logger.debug("event_stream idle (no token for %.0fs) for %s",
                                         STREAM_HEARTBEAT_INTERVAL, request_id)
                            if started:
                                yield _sse_empty_delta(request_id, created, request.model)
                        continue
                    if item_type == "token":
                        token_streamed = True
                        yield _sse({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": value},
                                "finish_reason": None,
                            }],
                        })
                        continue
                    if item_type == "reasoning":
                        yield _sse({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": value},
                                "finish_reason": None,
                            }],
                        })
                        continue
                    if item_type == "result":
                        tool_calls = value.get("tool_calls") or []
                        if tool_calls:
                            logger.info("Streaming delivering %d tool calls for %s", len(tool_calls), request_id)
                            yield from _stream_tool_call_events(
                                request_id=request_id,
                                created=created,
                                model_name=request.model,
                                result=value,
                            )
                            return
                        # Only emit content from result if not already streamed token-by-token
                        if not token_streamed:
                            content = value.get("text") or ""
                            if content:
                                yield _sse({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }],
                                })
                        continue
                    if item_type == "error":
                        logger.error("Streaming error for %s: %s", request_id, value)
                        yield _sse({"error": {"message": value, "type": "server_error"}})
                        yield "data: [DONE]\n\n"
                        return
                    if item_type == "upstream_result":
                        # Upstream responded with a full completion (non-streaming).
                        # Deliver the text content as a final delta.
                        logger.info("Streaming delivering upstream_result for %s", request_id)
                        choices = value.get("choices") or []
                        if choices:
                            message = choices[0].get("message") or {}
                            content = message.get("content") or ""
                            tool_calls = message.get("tool_calls") or []
                            if tool_calls:
                                # Patch in tool call info then reuse existing stream handler
                                value["tool_calls"] = tool_calls
                                value["finish_reason"] = choices[0].get("finish_reason", "tool_calls")
                                yield from _stream_tool_call_events(
                                    request_id=request_id,
                                    created=created,
                                    model_name=request.model,
                                    result=value,
                                )
                                return
                            if content:
                                yield _sse({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }],
                                })
                        continue
                    if item_type == "done":
                        stream_state["finished"] = True
                        logger.info("Streaming completed for %s, finish_reason=%s", request_id, value or "stop")
                        yield _sse({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": value or "stop",
                            }],
                        })
                        yield "data: [DONE]\n\n"
                        return

            def stream_with_teardown():
                """Close the run when the response ends before ``[DONE]``."""
                try:
                    yield from event_stream()
                finally:
                    if not stream_state["finished"]:
                        stop_run("response ended before the run finished")

            return StreamingResponse(stream_with_teardown(), media_type="text/event-stream")

        try:
            result = model.run(
                messages=request.messages,
                user_data=request.user,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                tools=request.tools,
                tool_choice=request.tool_choice,
                user_agent=user_agent,
            )
            logger.info(
                "chat.completions response " + json.dumps(result, cls=EnumEncoder))
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": now_ts(),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": _response_message(result),
                    "finish_reason": _finish_reason(result),
                }],
                "usage": _usage(result),
            })
        except RoutedResponse as routed:
            logger.info("Model routed to upstream: %s", routed.dest)
            if not runtime.server_config.upstream_base_url:
                raise HTTPException(status_code=503, detail="No upstream configured")
            upstream_result = _forward_to_upstream(routed, request, runtime.server_config)
            return JSONResponse({
                "id": upstream_result.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                "object": "chat.completion",
                "created": upstream_result.get("created", now_ts()),
                "model": upstream_result.get("model", request.model),
                "choices": upstream_result.get("choices", []),
                "usage": upstream_result.get("usage", {}),
            })
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(str(exc) + "\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if reserved:
                model.end()                  # release the per-model session

    return app, server_config
