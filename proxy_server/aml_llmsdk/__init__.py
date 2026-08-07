"""
LLM SDK Python Binding
======================

A Python wrapper for the Amlogic LLM SDK, generated with Cython.

Quick start::

    from aml_llmsdk import LLMSDK, Result, RunStatus

    def my_callback(result: Result, userdata, status: RunStatus) -> None:
        if status == RunStatus.NORMAL:
            print(result.text.decode("utf-8", errors="replace"), end="", flush=True)
        elif status == RunStatus.FINISH:
            print()

    sdk = LLMSDK()
    sdk.init("/path/to/model", on_token=my_callback)
    sdk.run("Hello!")
    sdk.uninit()
"""

from ._llmsdk import LLMSDK
from .types import (
    Result,
    RetStatus,
    RunMode,
    RunStatus,
    SamplingMode,
    MessageType,
)

__all__ = [
    "LLMSDK",
    "Result",
    "RetStatus",
    "RunMode",
    "RunStatus",
    "SamplingMode",
    "MessageType",
]
