from abc import ABC, abstractmethod
from typing import List, Optional, Union
from .types import ModelConfig, ChatMessage, ChatCompletionTool
import queue
import threading

class BaseModelRuntime(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.lock = threading.Lock()
        self.finish_reason = "stop"

    @abstractmethod
    def run(self,
            messages: List[ChatMessage],
            user_data: Optional[str],
            stream_queue: Optional["queue.Queue[object]"] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stop: Optional[Union[str, List[str]]] = None,
            tools: Optional[List[ChatCompletionTool]] = None,
            tool_choice: Optional[Union[str, dict]] = None,
            user_agent: Optional[str] = None, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def interrupt(self):
        pass