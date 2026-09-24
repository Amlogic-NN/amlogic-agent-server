from abc import ABC, abstractmethod
from typing import List, Optional, Union
from .types import ModelConfig, ChatMessage, ChatCompletionTool
import queue
import threading

class ModelSessionMixin:
    """One session per model: reserve it before dispatch, release it when done.

    Shared by the LLM/VLM runtimes (``BaseModelRuntime``) and the ASR runtime,
    so ``/v1/chat/completions`` and ``/v1/audio/transcriptions`` can answer 429
    for a model that is already running (``models.reject_when_busy``).
    """

    def __init__(self):
        self._session_guard = threading.Lock()
        self._session_busy = False

    def try_begin(self) -> bool:
        """Reserve this model's single session; ``False`` when it is already taken."""
        with self._session_guard:
            if self._session_busy:
                return False
            self._session_busy = True
            return True

    def end(self) -> None:
        """Release the session reserved by :meth:`try_begin` (idempotent)."""
        with self._session_guard:
            self._session_busy = False

    def is_busy(self) -> bool:
        with self._session_guard:
            return self._session_busy


class BaseModelRuntime(ModelSessionMixin, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
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