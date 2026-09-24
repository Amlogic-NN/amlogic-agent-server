# cython: language_level=3
"""Cython bindings for libasrsdk.so (aml_asr_*), loaded via dlopen."""

import os
from pathlib import Path

cdef extern from "asrsdk_func.h":
    int load_asrsdk_func(const char* lib_path)
    int asrsdk_init(void** context, int model_type, const char* model_path,
                    const char* decoder_path, const char* tokenizer_path,
                    const char* extra_json)
    int asrsdk_transcribe_file(void* context, const char* wav_path, const char* language,
                               const char* task, char** text_out, char** language_out)
    int asrsdk_uninit(void* context)
    void asrsdk_free_cstr(char* s)

cdef int AML_ASR_SUCCESS = 0
cdef int AML_ASR_MODEL_WHISPER = 0
cdef int _asrsdk_loaded = 0


def find_asrsdk_path() -> str:
    env = os.environ.get("AML_ASRSDK_PATH")
    if env:
        return env
    bundled = Path(__file__).resolve().parent / "libasrsdk.so"
    if bundled.is_file():
        return str(bundled)
    return "libasrsdk.so"


cdef class AsrSdk:
    cdef void* _ctx
    cdef object _keep

    def __cinit__(self, lib_path=None):
        cdef bytes path_b
        cdef const char* c_path
        global _asrsdk_loaded
        self._ctx = NULL
        self._keep = []
        path = lib_path or find_asrsdk_path()
        path_b = os.fsencode(path)
        c_path = path_b
        if _asrsdk_loaded == 0:
            if load_asrsdk_func(c_path) != 0:
                raise RuntimeError(f"Cannot load libasrsdk.so ({path})")
            _asrsdk_loaded = 1

    def init(self, backend, model_path, tokenizer_path, decoder_path=None, extra_json=None):
        cdef bytes model_b, tokens_b, decoder_b, extra_b
        cdef const char* c_decoder = NULL
        cdef const char* c_extra = NULL
        cdef int model_type
        cdef int rc

        model_type = AML_ASR_MODEL_WHISPER if backend == "whisper" else 1
        model_b = os.fsencode(os.path.abspath(model_path))
        tokens_b = os.fsencode(os.path.abspath(tokenizer_path))
        self._keep = [model_b, tokens_b]
        if decoder_path:
            decoder_b = os.fsencode(os.path.abspath(decoder_path))
            self._keep.append(decoder_b)
            c_decoder = decoder_b
        if extra_json:
            extra_b = extra_json.encode("utf-8") if isinstance(extra_json, str) else extra_json
            self._keep.append(extra_b)
            c_extra = extra_b
        rc = asrsdk_init(&self._ctx, model_type, model_b, c_decoder, tokens_b, c_extra)
        if rc != AML_ASR_SUCCESS or self._ctx == NULL:
            raise RuntimeError(f"aml_asr_init failed (rc={rc})")

    def transcribe(self, wav_path, language="auto", task="transcribe"):
        cdef bytes wav_b, lang_b, task_b
        cdef const char* c_lang = NULL
        cdef const char* c_task = NULL
        cdef char* text_c = NULL
        cdef char* lang_c = NULL
        cdef int rc
        cdef str text
        cdef str lang

        wav_b = os.fsencode(os.path.abspath(wav_path))
        if language:
            lang_b = language.encode("utf-8") if isinstance(language, str) else language
            c_lang = lang_b
        if task:
            task_b = task.encode("utf-8") if isinstance(task, str) else task
            c_task = task_b
        rc = asrsdk_transcribe_file(self._ctx, wav_b, c_lang, c_task, &text_c, &lang_c)
        try:
            if rc != AML_ASR_SUCCESS:
                raise RuntimeError(f"aml_asr_transcribe failed (rc={rc})")
            text = text_c.decode("utf-8") if text_c != NULL else ""
            lang = lang_c.decode("utf-8") if lang_c != NULL else ""
            return {"text": text, "language": lang}
        finally:
            asrsdk_free_cstr(text_c)
            asrsdk_free_cstr(lang_c)

    def close(self):
        if self._ctx != NULL:
            asrsdk_uninit(self._ctx)
            self._ctx = NULL

    def __dealloc__(self):
        if self._ctx != NULL:
            asrsdk_uninit(self._ctx)
            self._ctx = NULL
