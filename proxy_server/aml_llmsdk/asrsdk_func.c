/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 * This source code is subject to the terms and conditions defined in the
 * file 'LICENSE' which is part of this source code package.
 *
 * Description: Dynamic loading (dlopen/dlsym) wrappers for aml_asr_* API.
 */

#include "asrsdk_func.h"
#include "asrsdk.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(AML_ASRInitConfig) == 1024, "AML_ASRInitConfig ABI");
_Static_assert(sizeof(AML_ASRInput) == 1024, "AML_ASRInput ABI");
_Static_assert(sizeof(AML_ASRResult) == 1024, "AML_ASRResult ABI");

typedef AML_ASRRetStatus (*PFN_aml_asr_init)(ASRContext* context, const AML_ASRInitConfig* config);
typedef AML_ASRRetStatus (*PFN_aml_asr_transcribe)(ASRContext context, const AML_ASRInput* input,
                                                   AML_ASRResult* result);
typedef AML_ASRRetStatus (*PFN_aml_asr_free_result)(AML_ASRResult* result);
typedef AML_ASRRetStatus (*PFN_aml_asr_uninit)(ASRContext context);

typedef struct {
    PFN_aml_asr_init init;
    PFN_aml_asr_transcribe transcribe;
    PFN_aml_asr_free_result free_result;
    PFN_aml_asr_uninit uninit;
} ASRSDK_FUNC_PTR;

static ASRSDK_FUNC_PTR s_func;
static void* s_handle = NULL;
static int s_loaded = 0;

#define CHECK_DLSYM(error)                                                     \
    do {                                                                       \
        if ((error = dlerror()) != NULL) {                                     \
            fprintf(stderr, "[ASRSDK_FUNC] dlsym failed: %s\n", error);        \
            goto error_exit;                                                   \
        }                                                                      \
    } while (0)

int load_asrsdk_func(const char* lib_path)
{
    char* error = NULL;

    if (s_loaded)
        return 0;

    dlerror();
    s_handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!s_handle) {
        fprintf(stderr, "[ASRSDK_FUNC] dlopen(\"%s\") failed: %s\n",
                lib_path ? lib_path : "NULL", dlerror());
        return -1;
    }

    memset(&s_func, 0, sizeof(s_func));
    s_func.init = (PFN_aml_asr_init)dlsym(s_handle, "aml_asr_init");
    CHECK_DLSYM(error);
    s_func.transcribe = (PFN_aml_asr_transcribe)dlsym(s_handle, "aml_asr_transcribe");
    CHECK_DLSYM(error);
    s_func.free_result = (PFN_aml_asr_free_result)dlsym(s_handle, "aml_asr_free_result");
    CHECK_DLSYM(error);
    s_func.uninit = (PFN_aml_asr_uninit)dlsym(s_handle, "aml_asr_uninit");
    CHECK_DLSYM(error);

    s_loaded = 1;
    return 0;

error_exit:
    if (s_handle) {
        dlclose(s_handle);
        s_handle = NULL;
    }
    memset(&s_func, 0, sizeof(s_func));
    return -1;
}

void unload_asrsdk_func(void)
{
    if (s_handle) {
        dlclose(s_handle);
        s_handle = NULL;
    }
    memset(&s_func, 0, sizeof(s_func));
    s_loaded = 0;
}

static char* DupCString(const char* src)
{
    if (!src)
        src = "";
    size_t n = strlen(src);
    char* out = (char*)malloc(n + 1);
    if (!out)
        return NULL;
    memcpy(out, src, n + 1);
    return out;
}

int asrsdk_init(void** context, int model_type, const char* model_path, const char* decoder_path,
                const char* tokenizer_path, const char* extra_json)
{
    AML_ASRInitConfig cfg;
    ASRContext ctx = NULL;
    AML_ASRRetStatus rc;

    if (!s_func.init || !context)
        return AML_ASR_Status_Failed;

    memset(&cfg, 0, sizeof(cfg));
    cfg.model_type = (AML_ASRModelType)model_type;
    cfg.model_path = model_path;
    cfg.decoder_path = decoder_path;
    cfg.tokenizer_path = tokenizer_path;
    cfg.extra_json = extra_json;

    rc = s_func.init(&ctx, &cfg);
    if (rc != AML_ASR_Status_Success || ctx == NULL)
        return AML_ASR_Status_Failed;
    *context = ctx;
    return AML_ASR_Status_Success;
}

int asrsdk_transcribe_file(void* context, const char* wav_path, const char* language,
                           const char* task, char** text_out, char** language_out)
{
    AML_ASRInput inp;
    AML_ASRResult result;
    AML_ASRRetStatus rc;

    if (!s_func.transcribe || !s_func.free_result || !context || !wav_path || !text_out ||
        !language_out)
        return AML_ASR_Status_Failed;

    *text_out = NULL;
    *language_out = NULL;
    memset(&inp, 0, sizeof(inp));
    memset(&result, 0, sizeof(result));
    inp.audio_type = AML_ASR_AUDIO_FILE;
    inp.file_path = wav_path;
    inp.language = language;
    inp.task = task;

    rc = s_func.transcribe((ASRContext)context, &inp, &result);
    if (rc == AML_ASR_Status_Success) {
        *text_out = DupCString(result.text);
        *language_out = DupCString(result.language);
        if (!*text_out || !*language_out) {
            asrsdk_free_cstr(*text_out);
            asrsdk_free_cstr(*language_out);
            *text_out = NULL;
            *language_out = NULL;
            s_func.free_result(&result);
            return AML_ASR_Status_Failed;
        }
    }
    s_func.free_result(&result);
    return rc == AML_ASR_Status_Success ? AML_ASR_Status_Success : AML_ASR_Status_Failed;
}

int asrsdk_uninit(void* context)
{
    if (!s_func.uninit)
        return AML_ASR_Status_Failed;
    return s_func.uninit((ASRContext)context);
}

void asrsdk_free_cstr(char* s)
{
    free(s);
}
