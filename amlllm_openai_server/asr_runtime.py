# -*- coding: utf-8 -*-
"""On-device ASR runtime: model.json paths + aml_asr_transcribe."""

from __future__ import annotations

import logging
import os
import threading

from .asr_sdk import AsrSdk
from .model_runtime import ModelSessionMixin
from .types import ModelConfig

logger = logging.getLogger("llm.asr")


class AsrModelRuntime(ModelSessionMixin):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.lock = threading.Lock()
        self._sdk: AsrSdk | None = None
        self.initialized = False

    def _ensure_ready(self) -> None:
        if self.initialized:
            return
        backend = (self.config.model_type or "").lower()
        if backend not in ("whisper", "sensevoice"):
            raise ValueError(f"Unsupported ASR model_type: {self.config.model_type}")
        if not self.config.tokenizer_path:
            raise ValueError("ASR model.json requires tokenizer (tokens.txt or data_bin)")
        if backend == "whisper" and not self.config.decoder_path:
            raise ValueError("whisper model.json requires decoder")
        missing = [p for p in (
            self.config.model_path,
            self.config.tokenizer_path,
            self.config.decoder_path if backend == "whisper" else "",
        ) if p and not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "ASR files not found (check model.json weights/decoder/tokenizer): "
                + ", ".join(missing)
            )
        if AsrSdk is None:
            raise RuntimeError("ASR Cython extension is not built (_asrsdk)")
        sdk = AsrSdk()
        extra = self.config.asr_extra_json or None
        logger.info(
            "asr init model=%s type=%s weights=%s decoder=%s tokenizer=%s",
            self.config.name,
            backend,
            self.config.model_path,
            self.config.decoder_path or "",
            self.config.tokenizer_path,
        )
        sdk.init(
            backend=backend,
            model_path=self.config.model_path,
            tokenizer_path=self.config.tokenizer_path,
            decoder_path=self.config.decoder_path or None,
            extra_json=extra,
        )
        self._sdk = sdk
        self.initialized = True

    def transcribe(self, wav_path: str, language: str | None = None, task: str | None = None) -> dict:
        with self.lock:
            self._ensure_ready()
            lang = language or self.config.language or "auto"
            used_task = task or "transcribe"
            return self._sdk.transcribe(wav_path, language=lang, task=used_task)

    def close(self) -> None:
        with self.lock:
            if self._sdk is not None:
                self._sdk.close()
                self._sdk = None
            self.initialized = False
