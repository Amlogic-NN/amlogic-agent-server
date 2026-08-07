from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
import json
from enum import Enum


@dataclass
class ModelConfig:
    name: str
    model_path: str
    tokenizer_path: str = ""
    backend: str = "adla"
    model_type: str = "none"
    sampling_mode: str = "chain"
    sampler_params: dict[str, object] = field(default_factory=dict)
    system_prompt: str = ""
    prompt_prefix: str = ""
    prompt_postfix: str = ""
    stop: list[str] = field(default_factory=list)
    retain_history: bool = False
    loglevel: str = "ERROR"
    context_size: int = 4096
    threads: int = 0
    n_gpu_layers: int = 0
    chat_format: str = ""
    verbose: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    skill_workaround: bool = False
    template_kwargs: dict[str, object] = field(default_factory=dict)
    route_only: bool = False
    token_pair: Optional[dict[int, int]] = None
    # VLM (multimodal) config
    mmproj_path: str = ""
    vision_start: str = "<|vision_start|>"
    vision_end: str = "<|vision_end|>"
    image_pad: str = "<|image_pad|>"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None
    log_level: str = "info"
    title: str = "AMLLLM OneAPI Proxy"
    version: str = "0.2.0"
    upstream_base_url: Optional[str] = None
    upstream_api_key: Optional[str] = None
    upstream_model: Optional[str] = None
    skill_injection: bool = False
    force_upstream: bool = False
    cors_origins: str = "*"


@dataclass
class RoutingConfig:
    original_prompt: str
    fixed_prompt: str
    routing_prompt: str
    path_template: str


class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = None


class MessagePart(BaseModel):
    type: str = Field(default="text")
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[MessagePart]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    reasoning_content: Optional[Union[str, List[MessagePart]]] = None


class ChatCompletionToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ChatCompletionTool(BaseModel):
    type: str = Field(default="function")
    function: ChatCompletionToolFunction


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    tools: Optional[List[ChatCompletionTool]] = None
    tool_choice: Optional[Union[str, dict]] = None


class RoutedResponse(Exception):
    def __init__(self, original_messages: List[ChatMessage], messages: List[ChatMessage], dest: str = ""):
        self.messages = messages
        self.original_messages = original_messages
        self.dest = dest
        super().__init__(f"Routed to {dest}")
