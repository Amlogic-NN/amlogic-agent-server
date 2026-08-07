"""
Type stubs for the Cython extension ``aml_llmsdk._llmsdk``.

Provides IDE-friendly signatures and documentation for the :class:`LLMSDK` class.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from aml_llmsdk.types import (
    MessageType,
    Result,
    RetStatus,
    RunMode,
    RunStatus,
    SamplingMode,
)


class LLMSDK:
    """Python wrapper around the AML LLM SDK C API.

    Usage::

        sdk = LLMSDK()
        sdk.init("/path/to/model", on_token=my_callback)
        sdk.run("Hello!")
        sdk.uninit()
    """

    @property
    def context(self) -> Optional[int]:
        """The underlying ``LLMContext`` handle as an integer address, or ``None`` if not initialized."""
        ...

    @property
    def vision_metadata(self) -> Dict[str, Any]:
        """Vision model metadata from the mmproj model.

        Populated after ``init()`` if ``mmproj_path`` was provided.
        Keys: ``projector_type``, ``patch_size``, ``image_mean``, ``image_std``,
        ``spatial_merge_size``, ``image_fmt`` (1=NCHW, 0=NHWC),
        ``model_width``, ``model_height``, ``has_metadata``.
        Returns an empty dict if no mmproj was loaded or metadata is unavailable."""
        ...

    @property
    def mmproj_model(self) -> bool:
        """True if an mmproj (vision) model was loaded during init()."""
        ...

    def init(
        self,
        model_path: str,
        on_token: Callable[[Result, Any, RunStatus], None] | None = None,
        on_tool_call: Callable[[str, str, Any], None] | None = None,
        sampling_mode: str | int | SamplingMode = SamplingMode.ARG_MAX,
        sampler_params: Dict[str, Any] | None = None,
        mmproj_path: str = "",
        model_type_override: str = "",
        custom_jinja_template: str = "",
        mmproj_config: str = "",
    ) -> RetStatus:
        """Initialize the LLM instance.

        Args:
            model_path: Path to the LLM model file.
            on_token: A Python callable ``(result: Result, userdata: Any, status: RunStatus) -> None``,
                or ``None`` to suppress all callbacks (results are silently discarded).
            on_tool_call: A Python callable ``(tool_name: str, tool_args: str, userdata: Any) -> None``
                invoked by the SDK's PEG tool-call parser; mirrors ``on_token``.
                ``None`` only collects tool calls into the :meth:`run` result dict.
            sampling_mode: Sampling strategy for text generation.
            sampler_params: Sampling parameter dict (serialized to JSON as ``init_extend.sampler_params_json``).
            mmproj_path: Path to the mmproj (vision/projector) model for VLM support.
            model_type_override: Override model_type (e.g. "QWEN") when ADLA misdetects.
            custom_jinja_template: Optional Jinja2 chat template registered at init time.
            mmproj_config: Optional JSON config for a user-custom VLM projector.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` (0) or ``FAILED`` (1).
        """
        ...

    def uninit(self) -> RetStatus:
        """Uninitialize the LLM instance and release associated resources.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        ...

    def run(
        self,
        prompt_input: str = "",
        *,
        run_mode: RunMode = RunMode.GENERATE,
        retain_history: int = 0,
        userdata: Any = None,
        images: list | None = None,
        img_start: str = "<|vision_start|>",
        img_end: str = "<|vision_end|>",
        img_content: str = "<|image_pad|>",
        tool_schemas: List[Dict[str, Any]] | None = None,
        sampler_params: Dict[str, Any] | None = None,
        disable_chat_template: int = 1,
        extra_template_params: Dict[str, Any] | None = None,
        enable_think: int = 0,
        sampling_mode: str | int | SamplingMode | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repeat_penalty: float | None = None,
        messages_mode: bool = False,
    ) -> dict[str, Union[str, List[int], RunStatus, int, List[Dict[str, str]]]]:
        """Execute an inference task and return the complete result.

        Results are also delivered incrementally via the ``on_token`` callback
        if one was registered at init time. Fragments are split by
        ``content_type``: answer text (``text``), reasoning chain
        (``reasoning``), and raw tool-call markup (``tool_text``).

        Args:
            prompt_input: The text prompt (or OpenAI messages JSON when ``messages_mode`` is True).
            run_mode: Inference mode (e.g., ``RunMode.GENERATE``).
            retain_history: Whether to retain conversation history (1 = retain, 0 = reset).
            userdata: Arbitrary Python object forwarded to the ``on_token`` callback.
            images: List of numpy arrays (float32) for multimodal input.
            img_start/img_end/img_content: Vision placeholder markers.
            tool_schemas: OpenAI-format tool definitions list (passed as ``run_extend.tools_schemas``).
            sampler_params: Per-run sampling parameter overrides.
            disable_chat_template: If 1, disables SDK-side template (default 1).
            extra_template_params: Extra Jinja template params when ``disable_chat_template=0``.
            enable_think: Whether to enable thinking mode (1) or not (0).
            sampling_mode/top_k/top_p/temperature/repeat_penalty: Per-run sampling overrides;
                providing any sets ``sampling_config_valid=1``.
            messages_mode: If True, send ``prompt_input`` as ``AML_LLM_INPUT_MESSAGES``.

        Returns:
            A ``dict`` with keys:
                - ``text`` (``str``): Decoded answer text (TEXT fragments only).
                - ``reasoning`` (``str``): Decoded reasoning/thinking chain (REASONING fragments).
                - ``tool_text`` (``str``): Raw tool-call markup (TOOL fragments) for fallback parsing.
                - ``token_ids`` (``list[int]``): All received token IDs.
                - ``status`` (:class:`RunStatus`): The final status.
                - ``token_count`` (``int``): Total number of tokens received.
                - ``tool_calls`` (``list[dict]``): SDK-parsed tool calls
                  (each ``{"name": str, "arguments": str}``).
        """
        ...

    def reset(self) -> RetStatus:
        """Reset the LLM context, clearing any existing history or state.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        ...

    def interrupt(self) -> RetStatus:
        """Interrupt an active inference process.

        Attempts to stop any ongoing generation immediately.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        ...

    def set_chat_template(
        self,
        system_prompt: str,
        prompt_prefix: str,
        prompt_postfix: str,
    ) -> RetStatus:
        """Configure the chat formatting template.

        Args:
            system_prompt: Initial context or role of the LLM.
            prompt_prefix: String inserted before each user input.
            prompt_postfix: String appended after each user input.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        ...

    def set_chat_template_jinja(self, jinja_template: str) -> RetStatus:
        """Register a Jinja2 chat template on the context.

        Args:
            jinja_template: Jinja2 template string.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        ...

    def set_toolcall_callback(self, enabled: bool = True, stop_output: int = 0) -> RetStatus:
        """Enable/disable the SDK tool-call callback bridge.

        When enabled, tool calls parsed by the SDK's PEG parser are collected
        during :meth:`run` and returned in the result dict as ``tool_calls``.
        ``stop_output=0`` (default) preserves the full token stream for
        text-based fallback parsing; ``stop_output=1`` suppresses tool-call
        markup tokens from the normal result callback.

        Returns:
            :class:`RetStatus`: ``SUCCESS`` or ``FAILED``.
        """
        ...
