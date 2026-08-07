# cython: language_level=3
"""
Cython bindings for the AML LLM SDK (``llmsdk_ext.h``).

Provides the :class:`LLMSDK` class which wraps the C API exposed by
``libllmsdk.so`` (loaded via ``dlopen`` on aarch64), exposing them as Python
methods with the ``aml_llm_`` prefix removed.
"""

from libc.stdint cimport uint8_t, int16_t, int32_t, uint32_t, uint16_t, uint64_t, uintptr_t
from libc.string cimport const_char, memset
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free
from cpython.exc cimport PyErr_CheckSignals
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import json
import signal
import threading
import numpy as np
import os
from pathlib import Path

from .types import (
    RunStatus,
    RetStatus,
    Result,
    SamplingMode,
    RunMode,
)

# Mapping from string/int to C enum for sampling_mode parameter.
_SAMPLING_MODE_MAP = {
    # string
    "arg_max":         AML_LLM_ARG_Max,
    "argmax":          AML_LLM_ARG_Max,
    "max":             AML_LLM_ARG_Max,
    "top_p":           AML_LLM_TOP_P,
    "topp":            AML_LLM_TOP_P,
    "top_k":           AML_LLM_TOP_K,
    "topk":            AML_LLM_TOP_K,
    "chain_sampler":   AML_LLM_CHAIN_SAMPLER,
    "chain":           AML_LLM_CHAIN_SAMPLER,
    # int (raw enum value)
    0:                 AML_LLM_ARG_Max,
    1:                 AML_LLM_TOP_P,
    2:                 AML_LLM_TOP_K,
    3:                 AML_LLM_CHAIN_SAMPLER,
}
_SAMPLING_MODE_MAP.update({
    SamplingMode.ARG_MAX:        AML_LLM_ARG_Max,
    SamplingMode.TOP_P:          AML_LLM_TOP_P,
    SamplingMode.TOP_K:          AML_LLM_TOP_K,
    SamplingMode.CHAIN_SAMPLER:  AML_LLM_CHAIN_SAMPLER,
})

cdef AML_LLMSamplingMode _normalize_sampling_mode(object mode) except *:
    """Coerce a sampling mode from str, int, or SamplingMode to the C enum."""
    if isinstance(mode, SamplingMode):
        mode = mode.value  # fall through to int lookup
    if isinstance(mode, str):
        mode = mode.strip().lower()
    try:
        return <AML_LLMSamplingMode>_SAMPLING_MODE_MAP[mode]
    except KeyError:
        raise ValueError(
            f"Invalid sampling_mode: {mode!r}. "
            f"Expected one of: 'arg_max', 'top_p', 'top_k', 'chain_sampler', "
            f"or the corresponding SamplingMode/int value."
        )


# ---------------------------------------------------------------------------
# External C API declarations (mirrors sdk/include/llmsdk_ext.h)
# ---------------------------------------------------------------------------

cdef extern from "llmsdk_ext.h":
    ctypedef void* LLMContext

    ctypedef enum AML_LLMRunStatus:
        AML_LLM_RUN_NORMAL = 0
        AML_LLM_RUN_FINISH = 1
        AML_LLM_RUN_ERROR  = 2

    ctypedef enum AML_LLMRetStatus:
        AML_LLM_Status_Success = 0
        AML_LLM_Status_Failed  = 1

    ctypedef enum AML_LLMSamplingMode:
        AML_LLM_ARG_Max        = 0
        AML_LLM_TOP_P          = 1
        AML_LLM_TOP_K          = 2
        AML_LLM_CHAIN_SAMPLER  = 3

    ctypedef enum AML_LLMInputType:
        AML_LLM_INPUT_PROMPT     = 0
        AML_LLM_INPUT_MULTIMODAL = 1
        AML_LLM_INPUT_TOKEN      = 2
        AML_LLM_INPUT_MESSAGES   = 3

    ctypedef enum AML_LLMRunMode:
        AML_LLM_RUN_GENERATE = 0
        AML_LLM_RUN_EMBEDDING = 1

    ctypedef enum AML_LLMImageType:
        AML_LLM_IMAGE_TYPE_BUFFER = 0
        AML_LLM_IMAGE_TYPE_TENSOR = 1
        AML_LLM_IMAGE_TYPE_FILE   = 2
        AML_LLM_IMAGE_TYPE_BASE64 = 3

    ctypedef enum AML_LLMVisionDataType:
        AML_LLM_VISION_TYPE_INVALID = -1
        AML_LLM_VISION_TYPE_UINT8 = 0
        AML_LLM_VISION_TYPE_INT8 = 1
        AML_LLM_VISION_TYPE_UINT16 = 2
        AML_LLM_VISION_TYPE_INT16 = 3
        AML_LLM_VISION_TYPE_UINT32 = 4
        AML_LLM_VISION_TYPE_INT32 = 5
        AML_LLM_VISION_TYPE_UINT64 = 6
        AML_LLM_VISION_TYPE_INT64 = 7
        AML_LLM_VISION_TYPE_FP16 = 8
        AML_LLM_VISION_TYPE_FP32 = 9

    ctypedef enum AML_LLMMessageType:
        AML_LLM_MSG_TYPE_TEXT = 0
        AML_LLM_MSG_TYPE_REASONING = 1
        AML_LLM_MSG_TYPE_TOOL = 2

    # --- Init path ---
    ctypedef struct AML_LLMInitExtend:
        const char* vision_model_path
        const char* sampler_params_json
        const char* model_name_override
        const char* custom_jinja_template
        void* mmproj_input
        const char* mmproj_config
        uint8_t reserved[976]

    ctypedef struct AML_LLMInitConfig:
        const char* model_path
        AML_LLMSamplingMode sampling_mode
        int32_t top_k
        float top_p
        float temperature
        float repeat_penalty
        AML_LLMInitExtend init_extend

    # --- Input path ---
    ctypedef struct AML_LLMImageInput:
        AML_LLMImageType type
        const void* data
        uint32_t width
        uint32_t height
        uint64_t timestamp

    ctypedef struct AML_LLMMultimodalInput:
        const char* prompt
        const AML_LLMImageInput* img_inputs
        uint32_t img_count
        const char* img_start
        const char* img_end
        const char* img_content
        uint32_t sample_rate_num
        uint32_t sample_rate_den

    ctypedef struct AML_LLMTokenInput:
        const int32_t* input_ids
        int32_t n_tokens

    # AML_LLMInput contains an anonymous union; members are promoted to the
    # struct level in the C header. We declare the union members directly so
    # the C compiler resolves their (overlapping) offsets from the header.
    ctypedef struct AML_LLMInput:
        AML_LLMInputType input_type
        const char* prompt_input
        AML_LLMMultimodalInput multimodal_input
        AML_LLMTokenInput token_input
        const char* role

    # --- Run path ---
    ctypedef struct AML_LLMRunExtend:
        uint32_t sampling_config_valid
        AML_LLMSamplingMode sampling_mode
        int32_t top_k
        float top_p
        float temperature
        float repeat_penalty
        const char* sampler_params_json
        int32_t disable_chat_template
        const char* extra_template_params
        const char* tools_schemas
        uint8_t reserved[972]

    ctypedef struct AML_LLMRunConfig:
        AML_LLMRunMode run_mode
        int retain_history
        int enable_think
        AML_LLMRunExtend run_extend

    # --- Result ---
    ctypedef struct AML_LLMGenerationResult:
        const char* text
        int32_t token_id
        AML_LLMMessageType content_type

    ctypedef struct AML_LLMEmbeddingResult:
        const float* values
        uint32_t dimension

    # AML_LLMResult is an anonymous union of generation/embedding.
    ctypedef struct AML_LLMResult:
        AML_LLMGenerationResult generation
        AML_LLMEmbeddingResult embedding

    # --- Vision info ---
    ctypedef struct AML_LLMVisionTensorInfo:
        int32_t index
        char name[128]
        AML_LLMVisionDataType type
        int32_t height
        int32_t width
        int32_t channels
        int32_t size

    ctypedef struct AML_LLMVisionMetadata:
        char projector_type[64]
        float image_mean[3]
        float image_std[3]
        int32_t patch_size
        int32_t spatial_merge_size

    ctypedef struct AML_LLMVisionInfo:
        int32_t n_input
        const AML_LLMVisionTensorInfo* input
        AML_LLMVisionMetadata metadata
        int32_t has_metadata

    # --- Callbacks ---
    ctypedef void(*LLMResultCallback)(AML_LLMResult* result, void* userdata, AML_LLMRunStatus state)
    ctypedef void(*LLM_ToolUseCallback)(const char* tool_name, const char* tool_args, void* userdata)

    AML_LLMRetStatus aml_llm_init(LLMContext* context, AML_LLMInitConfig* init_config, LLMResultCallback callback) nogil
    AML_LLMRetStatus aml_llm_uninit(LLMContext context)
    AML_LLMRetStatus aml_llm_run(LLMContext context, AML_LLMInput* input, AML_LLMRunConfig* run_config, void* userdata) nogil
    AML_LLMRetStatus aml_llm_reset(LLMContext context)
    AML_LLMRetStatus aml_llm_break(LLMContext context)
    AML_LLMRetStatus aml_llm_set_chat_template(LLMContext context, const char* system_prompt, const char* prompt_prefix, const char* prompt_postfix)
    AML_LLMRetStatus aml_llm_set_chat_template_jinja(LLMContext context, const char* jinja_template)
    AML_LLMRetStatus aml_llm_set_toolcall_callback(LLMContext context, LLM_ToolUseCallback callback, int stop_output)
    AML_LLMRetStatus aml_llm_get_vision_info(LLMContext context, AML_LLMVisionInfo* vision_info)


cdef extern from "llmsdk_func.h":
    int load_llmsdk_func(const char* lib_path)
    void unload_llmsdk_func()


# ---------------------------------------------------------------------------
# C callback bridges
# ---------------------------------------------------------------------------

cdef void _callback_bridge(AML_LLMResult* c_result, void* userdata, AML_LLMRunStatus c_state) noexcept with gil:
    """
    C callback that always accumulates results internally and optionally
    forwards them to the user-provided Python callback.

    Always registered with the C SDK so that ``Ctrl+C`` (SIGINT) can be
    detected during a blocking ``aml_llm_run`` call. ``userdata`` is the
    ``LLMSDK`` instance (cast to ``void*``).
    """
    if userdata == NULL:
        return

    cdef LLMSDK sdk = <LLMSDK>userdata
    cdef bytes py_text = b""
    cdef int py_token_id = 0
    cdef AML_LLMMessageType ct = AML_LLM_MSG_TYPE_TEXT

    # Always check for KeyboardInterrupt so the user can cancel a long-running
    # inference with Ctrl+C, even when no Python callback is set.
    if sdk._signal_interrupt:
        sdk.interrupt()

    if c_result != NULL:
        py_token_id = c_result.generation.token_id
        ct = c_result.generation.content_type
        if c_result.generation.text != NULL:
            py_text = c_result.generation.text

    py_status = RunStatus(c_state)

    # Route fragments by content_type so reasoning (thinking chain) and tool
    # markup are accumulated separately from the answer text.
    if ct == AML_LLM_MSG_TYPE_REASONING:
        sdk._result_reasoning.append(py_text)
    elif ct == AML_LLM_MSG_TYPE_TOOL:
        sdk._result_tool_text.append(py_text)
    else:
        sdk._result_texts.append(py_text)
    sdk._result_token_ids.append(py_token_id)
    sdk._result_last_status = py_status

    # Optionally forward to the user's streaming callback.
    if sdk._py_callback is not None:
        py_result = Result(
            text=py_text,
            token_id=py_token_id,
            content_type=int(ct),
        )
        sdk._py_callback(py_result, sdk._py_userdata, py_status)


cdef void _tool_callback_bridge(const char* tool_name, const char* tool_args, void* userdata) noexcept with gil:
    """
    C callback invoked by the SDK's PEG tool-call parser when the model emits
    a tool call. ``userdata`` is the same pointer passed to ``aml_llm_run`` —
    the ``LLMSDK`` instance — so tool calls are collected into per-run state
    AND forwarded to the user-provided Python ``on_tool_call`` callback
    (mirroring the ``on_token`` forwarding in :func:`_callback_bridge`).
    """
    if userdata == NULL:
        return

    cdef LLMSDK sdk = <LLMSDK>userdata
    cdef bytes py_name = b""
    cdef bytes py_args = b""
    if tool_name != NULL:
        py_name = tool_name
    if tool_args != NULL:
        py_args = tool_args
    sdk._result_tool_calls.append((py_name, py_args))

    # Forward to the user's Python tool-call callback, if registered.
    if sdk._py_tool_callback is not None:
        sdk._py_tool_callback(
            py_name.decode("utf-8", errors="replace"),
            py_args.decode("utf-8", errors="replace"),
            sdk._py_userdata,
        )


cdef int _llmsdk_loaded = 0


# ---------------------------------------------------------------------------
# LLMSDK class
# ---------------------------------------------------------------------------

cdef class LLMSDK:
    """Python wrapper around the AML LLM SDK C API.

    Usage::

        sdk = LLMSDK()
        sdk.init(my_callback, model_path="/path/to/model")
        sdk.run("Hello, world!")
        sdk.uninit()
    """

    cdef LLMContext _context
    cdef object _py_callback
    cdef object _py_tool_callback
    cdef object _py_userdata
    cdef object _result_texts       # list[bytes]  (TEXT fragments)
    cdef object _result_reasoning   # list[bytes]  (REASONING fragments)
    cdef object _result_tool_text   # list[bytes]  (TOOL fragments, raw markup)
    cdef object _result_token_ids   # list[int]
    cdef object _result_tool_calls  # list[tuple[bytes, bytes]]
    cdef object _result_last_status # RunStatus | None
    cdef bool _signal_interrupt
    cdef object _old_sigint_handler # previous SIGINT handler (restored after run)
    cdef object _vision_metadata    # dict with model dims, mean, std, etc.
    cdef bool __mmproj_model

    def __cinit__(self):
        cdef bytes llmsdk_path_bytes
        cdef const char* c_path
        cdef int ret
        global _llmsdk_loaded
        self._context = NULL
        self._py_callback = None
        self._py_tool_callback = None
        self._py_userdata = None
        self._result_texts = []
        self._result_reasoning = []
        self._result_tool_text = []
        self._result_token_ids = []
        self._result_tool_calls = []
        self._result_last_status = None
        self._vision_metadata = {}
        self._signal_interrupt = False
        self._old_sigint_handler = None
        self.__mmproj_model = False
        if(_llmsdk_loaded == 0):
            llmsdk_path = Path(__file__).resolve().parent / 'libllmsdk.so'
            llmsdk_path_bytes = os.fsencode(llmsdk_path)
            c_path = llmsdk_path_bytes
            ret = load_llmsdk_func(c_path)
            if(ret != 0):
                raise RuntimeError("Cannot load libllmsdk.so")
            _llmsdk_loaded += 1
        self._install_sigint_handler()


    def _install_sigint_handler(self):
        """Install our SIGINT handler, saving the previous one.

        Called before entering the ``nogil`` block in :meth:`run` so that
        Ctrl+C is detected even while the native inference loop runs.

        Only active on the main thread; worker threads (e.g. FastAPI thread
        pool) cannot register signals and must interrupt via the public
        :meth:`interrupt` API instead.
        """
        if threading.current_thread() is threading.main_thread():
            self._old_sigint_handler = signal.signal(signal.SIGINT, self.signal_handler)

    def _restore_sigint_handler(self):
        """Restore the previous SIGINT handler saved by :meth:`_install_sigint_handler`.

        Called after :meth:`run` returns from the native call.  No-op on
        non-main threads.
        """
        if self._old_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._old_sigint_handler)
            self._old_sigint_handler = None

    def signal_handler(self, sig, frame):
        """SIGINT handler: flag the interrupt and break the native inference loop.

        This runs with the GIL re-acquired by Python's signal dispatch, so
        calling ``interrupt()`` (which calls ``aml_llm_break``) is safe even
        while the main thread is in the ``nogil`` block of :meth:`run`.
        """
        self._signal_interrupt = True
        self.interrupt()
        if self._old_sigint_handler is not None:
            self._old_sigint_handler(sig, frame)

    def __dealloc__(self):
        cdef LLMContext ctx = self._context
        global _llmsdk_loaded
        if ctx != NULL:
            aml_llm_uninit(ctx)
            self._context = NULL
        _llmsdk_loaded -= 1
        if _llmsdk_loaded == 0:
            unload_llmsdk_func()



    # -- Internal helpers ----------------------------------------------------

    cdef void _populate_vision_metadata(self, LLMContext ctx):
        """Query vision info from the SDK and populate ``self._vision_metadata``."""
        cdef AML_LLMVisionInfo vis_info
        cdef AML_LLMRetStatus rc
        cdef bytes _proj_type

        # Default metadata dict with sensible fallbacks.
        self._vision_metadata = {
            "projector_type": "",
            "patch_size": 14,
            "proj_scale_factor": 0,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "spatial_merge_size": 2,
            "image_size": 0,
            "n_wa_pattern": 0,
            "embed_dim": 0,
            "has_class_embed": 0,
            "nx": 0,
            "ny": 0,
            "has_metadata": 0,
            "image_fmt": 1,  # NCHW
            "model_width": 0,
            "model_height": 0,
        }

        rc = aml_llm_get_vision_info(ctx, &vis_info)
        if rc != AML_LLM_Status_Success:
            return

        # Populate from AML_LLMVisionMetadata.
        if vis_info.has_metadata != 0:
            self._vision_metadata["has_metadata"] = 1
            _proj_type = vis_info.metadata.projector_type
            self._vision_metadata["projector_type"] = _proj_type.decode("utf-8") if len(_proj_type) > 0 and _proj_type[0] != 0 else ""
            self._vision_metadata["image_mean"] = [
                vis_info.metadata.image_mean[0],
                vis_info.metadata.image_mean[1],
                vis_info.metadata.image_mean[2],
            ]
            self._vision_metadata["image_std"] = [
                vis_info.metadata.image_std[0],
                vis_info.metadata.image_std[1],
                vis_info.metadata.image_std[2],
            ]
            if vis_info.metadata.patch_size > 0:
                self._vision_metadata["patch_size"] = vis_info.metadata.patch_size
            if vis_info.metadata.spatial_merge_size > 0:
                self._vision_metadata["spatial_merge_size"] = vis_info.metadata.spatial_merge_size

        # Populate model dimensions from first tensor info (if available).
        if vis_info.n_input > 0 and vis_info.input != NULL:
            self._vision_metadata["model_width"] = vis_info.input[0].width
            self._vision_metadata["model_height"] = vis_info.input[0].height


    # -- Public API ----------------------------------------------------------

    def init(
        self,
        str model_path,
        object on_token = None,
        object on_tool_call = None,
        object sampling_mode = SamplingMode.ARG_MAX,
        object sampler_params = None,
        str mmproj_path = "",
        str model_type_override = "",
        str custom_jinja_template = "",
        str mmproj_config = "",
    ):
        """Initialize the LLM instance.

        Args:
            model_path: Path to the LLM model file.
            on_token: A Python callable with signature
                ``callback(result: Result, userdata: Any, status: RunStatus) -> None``,
                or ``None`` to suppress all callbacks (results are silently discarded).
                Called for every generated token.
            on_tool_call: A Python callable with signature
                ``callback(tool_name: str, tool_args: str, userdata: Any) -> None``,
                invoked by the SDK's PEG tool-call parser whenever the model
                emits a tool call. ``tool_args`` is a JSON string. Mirrors
                ``on_token``; pass ``None`` to only collect tool calls into
                the :meth:`run` result dict.
            sampling_mode: Sampling strategy for text generation.
            sampler_params: A ``dict[str,any]`` with sampling parameters
                (e.g. ``{"top_k": 3, "top_p": 0.9, "temperature": 1.0, ...}``).
                Serialized to JSON and passed as ``init_extend.sampler_params_json``.
            mmproj_path: Path to the mmproj (vision/projector) model for VLM support.
            model_type_override: Override model_type (e.g. "QWEN") when ADLA misdetects.
                Pass empty string or None for default behaviour.
            custom_jinja_template: Optional Jinja2 chat template string registered
                on the context at init time (mirrors ``init_extend.custom_jinja_template``).
            mmproj_config: Optional JSON config for a user-custom VLM projector.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` (0) or ``FAILED`` (1).
        """
        cdef AML_LLMInitConfig c_cfg
        cdef bytes _b_model_path
        cdef bytes _b_sampler_params_json
        cdef bytes _b_mmproj_path
        cdef bytes _b_model_type_override
        cdef bytes _b_custom_jinja
        cdef bytes _b_mmproj_config

        # Zero the whole config (handles reserved + all fields); the C
        # compiler's sizeof (from the packed header) is authoritative here.
        memset(&c_cfg, 0, sizeof(c_cfg))

        _b_model_path = model_path.encode("utf-8") if model_path else b""
        cdef const char* c_model_path = NULL
        if len(_b_model_path) > 0:
            c_model_path = _b_model_path
        c_cfg.model_path = c_model_path
        c_cfg.sampling_mode = _normalize_sampling_mode(sampling_mode)
        # Per-init sampling defaults live in init_extend now.
        if sampler_params is not None:
            _b_sampler_params_json = json.dumps(sampler_params, ensure_ascii=False).encode("utf-8")
            c_cfg.init_extend.sampler_params_json = _b_sampler_params_json
        else:
            c_cfg.init_extend.sampler_params_json = NULL

        if mmproj_path and len(mmproj_path) > 0:
            _b_mmproj_path = mmproj_path.encode("utf-8")
            c_cfg.init_extend.vision_model_path = _b_mmproj_path
            self.__mmproj_model = True
        else:
            self.__mmproj_model = False

        if model_type_override and len(model_type_override) > 0:
            _b_model_type_override = model_type_override.encode("utf-8")
            c_cfg.init_extend.model_name_override = _b_model_type_override

        if custom_jinja_template and len(custom_jinja_template) > 0:
            _b_custom_jinja = custom_jinja_template.encode("utf-8")
            c_cfg.init_extend.custom_jinja_template = _b_custom_jinja

        if mmproj_config and len(mmproj_config) > 0:
            _b_mmproj_config = mmproj_config.encode("utf-8")
            c_cfg.init_extend.mmproj_config = _b_mmproj_config

        self._py_callback = on_token
        self._py_tool_callback = on_tool_call
        self._py_userdata = None

        cdef AML_LLMRetStatus rc
        cdef LLMContext ctx = NULL
        with nogil:
            rc = aml_llm_init(&ctx, &c_cfg, _callback_bridge)
        self._context = ctx

        # Register the internal tool-call callback bridge (stop_output=0 so the
        # full token stream is preserved for text-based fallback parsing).
        if ctx != NULL:
            aml_llm_set_toolcall_callback(ctx, _tool_callback_bridge, 1)

        # Populate vision metadata if mmproj model was loaded.
        if self.__mmproj_model and ctx != NULL:
            self._populate_vision_metadata(ctx)

        return RetStatus(rc)

    def uninit(self):
        """Uninitialize the LLM instance and release associated resources.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        cdef LLMContext ctx = self._context
        if ctx == NULL:
            return RetStatus.SUCCESS

        cdef AML_LLMRetStatus rc

        rc = aml_llm_uninit(ctx)
        self._context = NULL
        self._py_callback = None
        self._py_tool_callback = None
        self._py_userdata = None
        return RetStatus(rc)

    def run(
        self,
        str prompt_input = "",
        object run_mode = RunMode.GENERATE,
        int retain_history = 0,
        userdata = None,
        images: list = None,
        str img_start = "<|vision_start|>",
        str img_end = "<|vision_end|>",
        str img_content = "<|image_pad|>",
        object tool_schemas = None,
        object sampler_params = None,
        int disable_chat_template = 1,
        object extra_template_params = None,
        int enable_think = 0,
        object sampling_mode = None,
        object top_k = None,
        object top_p = None,
        object temperature = None,
        object repeat_penalty = None,
        bint messages_mode = False,
    ):
        """Execute an inference task and return the complete result.

        Results are also delivered incrementally via the ``on_token`` callback
        if one was registered at init time.

        Args:
            prompt_input: The text prompt to send to the LLM. When
                ``messages_mode`` is True this is the OpenAI-format messages
                JSON string and the SDK applies the chat template internally.
            run_mode: Inference mode (e.g., :class:`RunMode.GENERATE`).
            retain_history: Whether to retain conversation history (1 = retain, 0 = reset).
            userdata: Arbitrary Python object forwarded to the ``on_token`` callback.
            images: List of numpy arrays (float32) for multimodal input.
                May be NCHW (3×H×W) or NHWC (H×W×3); normalized to HWC for the SDK.
            img_start: Token text before image embedding (e.g. <|vision_start|>).
            img_end: Token text after image embedding (e.g. <|vision_end|>).
            img_content: Placeholder in the prompt text (e.g. <|image_pad|>).
            tool_schemas: A ``list[dict]`` of OpenAI-format tool definitions.
                Serialized to JSON and passed as ``run_extend.tools_schemas``.
            sampler_params: A ``dict[str,any]`` with per-run sampling parameter overrides.
                Serialized to JSON and passed as ``run_extend.sampler_params_json``.
            disable_chat_template: If 1, disables SDK-side chat template (default 1,
                since Jinja rendering is done in Python).
            extra_template_params: A ``dict[str,any]`` with extra Jinja template params
                (e.g. ``{"enable_think": true}``). Only used when ``disable_chat_template=0``.
            enable_think: Whether to enable thinking mode (1) or not (0).
            sampling_mode: Per-run sampling strategy override (str/int/SamplingMode).
            top_k/top_p/temperature/repeat_penalty: Per-run sampling overrides;
                when any is provided, ``sampling_config_valid`` is set to 1.
            messages_mode: If True, send ``prompt_input`` as ``AML_LLM_INPUT_MESSAGES``
                (OpenAI messages JSON) and let the SDK apply the Jinja template.

        Returns:
            ``dict`` with keys:
                - **text** (``str``): Complete decoded output.
                - **token_ids** (``list[int]``): All received token IDs in order.
                - **status** (:class:`RunStatus`): The final status.
                - **token_count** (``int``): Total number of tokens received.
                - **tool_calls** (``list[dict]``): SDK-parsed tool calls, each
                  ``{"name": str, "arguments": str}`` (arguments is a JSON string).
        """
        cdef LLMContext ctx = self._context
        if ctx == NULL:
            raise RuntimeError("LLMSDK has not been initialized – call init() first")

        # Declare all cdef variables at top of def function (Cython requirement).
        cdef AML_LLMInput c_input
        cdef bytes _b_prompt
        cdef const char* c_prompt = NULL
        cdef AML_LLMRunConfig c_run_cfg
        cdef AML_LLMRetStatus rc
        cdef bytes _full
        cdef size_t _i

        cdef object img_arr
        cdef int img_h, img_w
        cdef float[:, :, ::1] img_memview
        cdef bytes _b_img_start, _b_img_end, _b_img_content
        cdef int _n_images, _img_idx
        cdef AML_LLMImageInput* _image_inputs = NULL
        cdef list _img_refs

        cdef bytes _b_tool_schemas, _b_sampler_params, _b_extra_template
        cdef uint32_t _sampling_config_valid = 0
        cdef const char* c_img_start = NULL
        cdef const char* c_img_end = NULL
        cdef const char* c_img_content = NULL

        # Zero input + run config (reserved bytes included).
        memset(&c_input, 0, sizeof(c_input))
        memset(&c_run_cfg, 0, sizeof(c_run_cfg))

        # Build input struct: multimodal, messages-JSON, or text-only.
        if images is not None and len(images) > 0:
            if not self.__mmproj_model:
                raise RuntimeError("Images provided but mmproj not initialized — pass mmproj_path to init()")

            _n_images = len(images)
            _img_refs = []

            # Allocate C array of AML_LLMImageInput structs.
            _image_inputs = <AML_LLMImageInput*>malloc(_n_images * sizeof(AML_LLMImageInput))
            if _image_inputs == NULL:
                raise MemoryError("Failed to allocate AML_LLMImageInput array")
            memset(_image_inputs, 0, _n_images * sizeof(AML_LLMImageInput))

            _b_img_start = img_start.encode("utf-8") if img_start else b""
            _b_img_end = img_end.encode("utf-8") if img_end else b""
            _b_img_content = img_content.encode("utf-8") if img_content else b""

            if messages_mode:
                # SDK renders the Jinja template over the messages JSON; the
                # template owns the vision placeholders, so pass the prompt
                # verbatim (no img_start/img_end wrapping). See test_vlm_messages.
                _b_prompt = prompt_input.encode("utf-8") if prompt_input else b""
                # Let the template supply vision delimiters when not provided.
                if len(_b_img_start) == 0:
                    c_img_start = NULL
                if len(_b_img_end) == 0:
                    c_img_end = NULL
            else:
                # Python-rendered prompt: wrap img_content placeholders with
                # img_start/img_end so the SDK knows the image span.
                _wrapped_prompt = prompt_input
                if img_start and img_content and img_content in _wrapped_prompt:
                    _wrapped_prompt = _wrapped_prompt.replace(
                        img_content, img_start + img_content + img_end
                    )
                _b_prompt = _wrapped_prompt.encode("utf-8") if _wrapped_prompt else b""
            if len(_b_prompt) > 0:
                c_prompt = _b_prompt
            else:
                c_prompt = NULL

            for _img_idx in range(_n_images):
                img_arr = np.asarray(images[_img_idx], dtype=np.float32)
                if img_arr.ndim != 3:
                    raise ValueError(f"Image must be 3D (C,H,W) or (H,W,C), got shape {img_arr.shape}")
                # SDK tensor input expects HWC float32 (per test_vlm_input.cpp).
                # Normalize NCHW (3,H,W) -> HWC (H,W,3); keep HWC as-is.
                if img_arr.shape[0] == 3 and img_arr.shape[0] != img_arr.shape[2]:
                    img_arr = np.ascontiguousarray(np.transpose(img_arr, (1, 2, 0)))
                else:
                    img_arr = np.ascontiguousarray(img_arr)
                _img_refs.append(img_arr)
                img_memview = img_arr
                img_h = img_arr.shape[0]
                img_w = img_arr.shape[1]

                _image_inputs[_img_idx].type = AML_LLM_IMAGE_TYPE_TENSOR
                _image_inputs[_img_idx].data = <const void*>&img_memview[0, 0, 0]
                _image_inputs[_img_idx].width = <uint32_t>img_w
                _image_inputs[_img_idx].height = <uint32_t>img_h
                _image_inputs[_img_idx].timestamp = 0

            if len(_b_img_start) > 0:
                c_img_start = _b_img_start
            if len(_b_img_end) > 0:
                c_img_end = _b_img_end
            if len(_b_img_content) > 0:
                c_img_content = _b_img_content

            c_input.input_type = AML_LLM_INPUT_MULTIMODAL
            c_input.multimodal_input.prompt = c_prompt
            c_input.multimodal_input.img_inputs = _image_inputs
            c_input.multimodal_input.img_count = <uint32_t>_n_images
            c_input.multimodal_input.img_start = c_img_start
            c_input.multimodal_input.img_end = c_img_end
            c_input.multimodal_input.img_content = c_img_content
            c_input.multimodal_input.sample_rate_num = 0
            c_input.multimodal_input.sample_rate_den = 0
            c_input.role = NULL
        elif messages_mode:
            # Pass OpenAI messages JSON; the SDK applies the Jinja template.
            _b_prompt = prompt_input.encode("utf-8") if prompt_input else b""
            if len(_b_prompt) > 0:
                c_prompt = _b_prompt
            else:
                c_prompt = NULL
            c_input.input_type = AML_LLM_INPUT_MESSAGES
            c_input.prompt_input = c_prompt
            c_input.role = NULL
        else:
            # Text-only path.
            _b_prompt = prompt_input.encode("utf-8") if prompt_input else b""
            c_prompt = NULL
            if len(_b_prompt) > 0:
                c_prompt = _b_prompt
            c_input.input_type = AML_LLM_INPUT_PROMPT
            c_input.prompt_input = c_prompt
            c_input.role = NULL

        # Run config.
        c_run_cfg.run_mode = <AML_LLMRunMode>run_mode.value
        c_run_cfg.retain_history = retain_history
        c_run_cfg.enable_think = enable_think

        # run_extend: tools_schemas, disable_chat_template, extra_template_params,
        # and per-run sampling overrides (sampling_config_valid gates them).
        c_run_cfg.run_extend.disable_chat_template = disable_chat_template

        if tool_schemas is not None and len(tool_schemas) > 0:
            _b_tool_schemas = json.dumps(tool_schemas, ensure_ascii=False).encode("utf-8")
            c_run_cfg.run_extend.tools_schemas = _b_tool_schemas
        else:
            c_run_cfg.run_extend.tools_schemas = NULL

        if extra_template_params is not None:
            _b_extra_template = json.dumps(extra_template_params, ensure_ascii=False).encode("utf-8")
            c_run_cfg.run_extend.extra_template_params = _b_extra_template
        else:
            c_run_cfg.run_extend.extra_template_params = NULL

        if sampler_params is not None:
            _b_sampler_params = json.dumps(sampler_params, ensure_ascii=False).encode("utf-8")
            c_run_cfg.run_extend.sampler_params_json = _b_sampler_params
        else:
            c_run_cfg.run_extend.sampler_params_json = NULL

        # Per-run sampling overrides. Set sampling_config_valid=1 when any
        # per-run sampling knob (mode/top_k/top_p/temp/repeat/sampler_params)
        # is provided so the SDK honours them over init-time defaults.
        if sampling_mode is not None:
            c_run_cfg.run_extend.sampling_mode = _normalize_sampling_mode(sampling_mode)
            _sampling_config_valid = 1
        if top_k is not None:
            c_run_cfg.run_extend.top_k = <int32_t>int(top_k)
            _sampling_config_valid = 1
        if top_p is not None:
            c_run_cfg.run_extend.top_p = <float>float(top_p)
            _sampling_config_valid = 1
        if temperature is not None:
            c_run_cfg.run_extend.temperature = <float>float(temperature)
            _sampling_config_valid = 1
        if repeat_penalty is not None:
            c_run_cfg.run_extend.repeat_penalty = <float>float(repeat_penalty)
            _sampling_config_valid = 1
        if sampler_params is not None:
            _sampling_config_valid = 1
        c_run_cfg.run_extend.sampling_config_valid = _sampling_config_valid

        self._py_userdata = userdata
        self._result_texts = []
        self._result_reasoning = []
        self._result_tool_text = []
        self._result_token_ids = []
        self._result_tool_calls = []
        self._result_last_status = None
        self._signal_interrupt = False

        # Install SIGINT handler that can interrupt the native inference loop.

        with nogil:
            rc = aml_llm_run(
                ctx, &c_input, &c_run_cfg, <void*>self
            )


        # Free allocated image inputs array.
        if images is not None and len(images) > 0:
            free(_image_inputs)

        _full = b"".join(self._result_texts)
        _reasoning = b"".join(self._result_reasoning)
        _tool_text = b"".join(self._result_tool_text)
        if self._signal_interrupt:
            raise KeyboardInterrupt()

        # Normalize SDK-collected tool calls into {name, arguments} dicts.
        py_tool_calls = []
        for _name, _args in self._result_tool_calls:
            py_tool_calls.append({
                "name": _name.decode("utf-8", errors="replace"),
                "arguments": _args.decode("utf-8", errors="replace"),
            })

        return {
            "text": _full.decode("utf-8", errors="replace"),
            "reasoning": _reasoning.decode("utf-8", errors="replace"),
            "tool_text": _tool_text.decode("utf-8", errors="replace"),
            "token_ids": list(self._result_token_ids),
            "status": self._result_last_status if self._result_last_status is not None else RunStatus.ERROR,
            "token_count": len(self._result_token_ids),
            "tool_calls": py_tool_calls,
        }

    def reset(self):
        """Reset the LLM context, clearing any existing history or state.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        cdef LLMContext ctx = self._context
        if ctx == NULL:
            raise RuntimeError("LLMSDK has not been initialized – call init() first")

        cdef AML_LLMRetStatus rc

        rc = aml_llm_reset(ctx)
        return RetStatus(rc)

    def interrupt(self):
        """Interrupt an active inference process.

        Attempts to stop any ongoing generation immediately.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        cdef LLMContext ctx = self._context
        if ctx == NULL:
            raise RuntimeError("LLMSDK has not been initialized – call init() first")

        cdef AML_LLMRetStatus rc

        rc = aml_llm_break(ctx)
        return RetStatus(rc)

    def set_chat_template(self, str system_prompt, str prompt_prefix, str prompt_postfix):
        """Configure the chat formatting template.

        Defines how prompts are constructed during chat interactions.

        Args:
            system_prompt: Initial context or role of the LLM.
            prompt_prefix: String inserted before each user input.
            prompt_postfix: String appended after each user input.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        cdef LLMContext ctx = self._context
        if ctx == NULL:
            raise RuntimeError("LLMSDK has not been initialized – call init() first")

        cdef bytes b_sys = system_prompt.encode("utf-8") if system_prompt else b""
        cdef bytes b_pre = prompt_prefix.encode("utf-8") if prompt_prefix else b""
        cdef bytes b_post = prompt_postfix.encode("utf-8") if prompt_postfix else b""

        cdef const char* c_sys = NULL
        cdef const char* c_pre = NULL
        cdef const char* c_post = NULL
        if len(b_sys) > 0:
            c_sys = b_sys
        if len(b_pre) > 0:
            c_pre = b_pre
        if len(b_post) > 0:
            c_post = b_post

        cdef AML_LLMRetStatus rc

        rc = aml_llm_set_chat_template(
                ctx, c_sys, c_pre, c_post
            )
        return RetStatus(rc)


    def set_chat_template_jinja(self, str jinja_template):
        """Register a Jinja2 chat template on the context.

        Args:
            jinja_template: Jinja2 template string.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        cdef LLMContext ctx = self._context
        if ctx == NULL:
            raise RuntimeError("LLMSDK has not been initialized – call init() first")

        cdef bytes b_jinja = jinja_template.encode("utf-8") if jinja_template else b""
        cdef const char* c_jinja = NULL
        if len(b_jinja) > 0:
            c_jinja = b_jinja

        cdef AML_LLMRetStatus rc

        rc = aml_llm_set_chat_template_jinja(ctx, c_jinja)
        return RetStatus(rc)

    def set_toolcall_callback(self, enabled=True, int stop_output=0):
        """Enable/disable the SDK tool-call callback.

        When enabled, the internal bridge collects ``(name, args)`` pairs
        during :meth:`run` (exposed in the result dict as ``tool_calls``).
        ``stop_output=0`` (default) keeps the full token stream available for
        text-based fallback parsing; ``stop_output=1`` suppresses tool-call
        markup tokens from the normal result callback.

        Args:
            enabled: True to register the bridge, False to unregister.
            stop_output: 1 to suppress tool-call text in the result callback.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        cdef LLMContext ctx = self._context
        if ctx == NULL:
            raise RuntimeError("LLMSDK has not been initialized – call init() first")

        cdef LLM_ToolUseCallback cb = NULL
        if enabled:
            cb = _tool_callback_bridge
        cdef AML_LLMRetStatus rc

        rc = aml_llm_set_toolcall_callback(ctx, cb, stop_output)
        return RetStatus(rc)

    @property
    def vision_metadata(self):
        """Return vision metadata dict if image encoder was initialized, else empty dict."""
        return self._vision_metadata

    @property
    def mmproj_model(self):
        """True if an mmproj (vision) model was loaded during init()."""
        return self.__mmproj_model

    # -- Properties ----------------------------------------------------------

    @property
    def context(self):
        """The underlying ``LLMContext`` handle (``int`` address, or ``None``)."""
        if self._context == NULL:
            return None
        return <uintptr_t>self._context
