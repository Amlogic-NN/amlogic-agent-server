# -*- coding: utf-8 -*-
"""Python entry for libasrsdk.so (Cython dlopen of aml_asr_*)."""

from __future__ import annotations

try:
    from .aml_llmsdk._asrsdk import AsrSdk, find_asrsdk_path
except ImportError:
    AsrSdk = None  # type: ignore[assignment,misc]

    def find_asrsdk_path() -> str:
        return "libasrsdk.so"


__all__ = ["AsrSdk", "find_asrsdk_path"]
