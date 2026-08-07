/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 * This source code is subject to the terms and conditions defined in the
 * file 'LICENSE' which is part of this source code package.
 *
 * Description: Dynamic loading (dlopen/dlsym) wrappers for aml_llm_* API.
 *
 * Only the symbols exported as extern "C" by libllmsdk.so are resolved here.
 * The new SDK exposes get_chat_template_jinja and the penalty-token helpers
 * only as C++ mangled symbols, so they are intentionally NOT loaded.
 */

#include "llmsdk_func.h"
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


/* Function pointer types for dynamic loading (dlopen/dlsym) of aml_llm_* APIs */
typedef AML_LLMRetStatus (*PFN_aml_llm_init)(LLMContext* context, AML_LLMInitConfig* init_config, LLMResultCallback callback);
typedef AML_LLMRetStatus (*PFN_aml_llm_uninit)(LLMContext context);
typedef AML_LLMRetStatus (*PFN_aml_llm_run)(LLMContext context, AML_LLMInput* input, AML_LLMRunConfig* run_config, void* userdata);
typedef AML_LLMRetStatus (*PFN_aml_llm_reset)(LLMContext context);
typedef AML_LLMRetStatus (*PFN_aml_llm_break)(LLMContext context);
typedef AML_LLMRetStatus (*PFN_aml_llm_set_chat_template)(LLMContext context, const char* system_prompt, const char* prompt_prefix, const char* prompt_postfix);
typedef AML_LLMRetStatus (*PFN_aml_llm_set_chat_template_jinja)(LLMContext context, const char* jinja_template);
typedef AML_LLMRetStatus (*PFN_aml_llm_set_toolcall_callback)(LLMContext context, LLM_ToolUseCallback callback, int stop_output);
typedef AML_LLMRetStatus (*PFN_aml_llm_get_vision_info)(LLMContext context, AML_LLMVisionInfo* vision_info);

/* ------------------------------------------------------------------ */
/*  Structure holding all function pointers loaded via dlsym          */
/* ------------------------------------------------------------------ */
typedef struct {
    PFN_aml_llm_init                   aml_llm_init;
    PFN_aml_llm_uninit                 aml_llm_uninit;
    PFN_aml_llm_run                    aml_llm_run;
    PFN_aml_llm_reset                  aml_llm_reset;
    PFN_aml_llm_break                  aml_llm_break;
    PFN_aml_llm_set_chat_template      aml_llm_set_chat_template;
    PFN_aml_llm_set_chat_template_jinja aml_llm_set_chat_template_jinja;
    PFN_aml_llm_set_toolcall_callback  aml_llm_set_toolcall_callback;
    PFN_aml_llm_get_vision_info        aml_llm_get_vision_info;
} LLMSDK_FUNC_PTR;

/* ------------------------------------------------------------------ */
/*  Internal state                                                    */
/* ------------------------------------------------------------------ */
static LLMSDK_FUNC_PTR  s_func;       /**< Loaded function pointers       */
static void*            s_handle = NULL; /**< dlopen handle               */
static int              s_loaded  = 0;   /**< Non-zero after success      */

/* ------------------------------------------------------------------ */
/*  Helper macro: check dlerror() after each dlsym                    */
/* ------------------------------------------------------------------ */
#define CHECK_DLSYM(error)                     do {                    \
    if ((error = dlerror()) != NULL) {                                 \
        fprintf(stderr, "[LLMSDK_FUNC] dlsym failed: %s\n", error);   \
        goto error_exit;                                               \
    }                                                                  \
} while (0)

/* ------------------------------------------------------------------ */
/*  Loader: dlopen the shared library and resolve all symbols         */
/* ------------------------------------------------------------------ */
int load_llmsdk_func(const char* lib_path)
{

#ifdef __ARM_ARCH
    char* error = NULL;

    if (s_loaded)
        return 0;

    dlerror(); /* Clear any prior error */

    s_handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!s_handle) {
        fprintf(stderr, "[LLMSDK_FUNC] dlopen(\"%s\") failed: %s\n",
                lib_path ? lib_path : "NULL", dlerror());
        return -1;
    }

    memset(&s_func, 0, sizeof(s_func));

    s_func.aml_llm_init                      = (PFN_aml_llm_init)                      dlsym(s_handle, "aml_llm_init");
    CHECK_DLSYM(error);
    s_func.aml_llm_uninit                    = (PFN_aml_llm_uninit)                    dlsym(s_handle, "aml_llm_uninit");
    CHECK_DLSYM(error);
    s_func.aml_llm_run                       = (PFN_aml_llm_run)                       dlsym(s_handle, "aml_llm_run");
    CHECK_DLSYM(error);
    s_func.aml_llm_reset                     = (PFN_aml_llm_reset)                     dlsym(s_handle, "aml_llm_reset");
    CHECK_DLSYM(error);
    s_func.aml_llm_break                     = (PFN_aml_llm_break)                     dlsym(s_handle, "aml_llm_break");
    CHECK_DLSYM(error);
    s_func.aml_llm_set_chat_template         = (PFN_aml_llm_set_chat_template)         dlsym(s_handle, "aml_llm_set_chat_template");
    CHECK_DLSYM(error);
    s_func.aml_llm_set_chat_template_jinja   = (PFN_aml_llm_set_chat_template_jinja)   dlsym(s_handle, "aml_llm_set_chat_template_jinja");
    CHECK_DLSYM(error);
    s_func.aml_llm_set_toolcall_callback    = (PFN_aml_llm_set_toolcall_callback)     dlsym(s_handle, "aml_llm_set_toolcall_callback");
    CHECK_DLSYM(error);
    s_func.aml_llm_get_vision_info           = (PFN_aml_llm_get_vision_info)           dlsym(s_handle, "aml_llm_get_vision_info");
    CHECK_DLSYM(error);

    s_loaded = 1;
    return 0;

error_exit:
    if (s_handle) {
        dlclose(s_handle);
        s_handle = NULL;
    }
    return -1;

#else

return 0;

#endif

}

/* ------------------------------------------------------------------ */
/*  Unloader: release the shared library                              */
/* ------------------------------------------------------------------ */
void unload_llmsdk_func(void)
{
    if (s_handle) {
        dlclose(s_handle);
        s_handle = NULL;
    }
    memset(&s_func, 0, sizeof(s_func));
    s_loaded = 0;
}

/* ------------------------------------------------------------------ */
/*  Proxy functions                                                   */
/*  Each calls through the corresponding function-pointer in s_func.  */
/* ------------------------------------------------------------------ */

AML_LLMRetStatus aml_llm_init(LLMContext* context, AML_LLMInitConfig* init_config, LLMResultCallback callback)
{
    if (!s_func.aml_llm_init) return AML_LLM_Status_Failed;
    return s_func.aml_llm_init(context, init_config, callback);
}

AML_LLMRetStatus aml_llm_uninit(LLMContext context)
{
    if (!s_func.aml_llm_uninit) return AML_LLM_Status_Failed;
    return s_func.aml_llm_uninit(context);
}

AML_LLMRetStatus aml_llm_run(LLMContext context, AML_LLMInput* input, AML_LLMRunConfig* run_config, void* userdata)
{
    if (!s_func.aml_llm_run) return AML_LLM_Status_Failed;
    return s_func.aml_llm_run(context, input, run_config, userdata);
}

AML_LLMRetStatus aml_llm_reset(LLMContext context)
{
    if (!s_func.aml_llm_reset) return AML_LLM_Status_Failed;
    return s_func.aml_llm_reset(context);
}

AML_LLMRetStatus aml_llm_break(LLMContext context)
{
    if (!s_func.aml_llm_break) return AML_LLM_Status_Failed;
    return s_func.aml_llm_break(context);
}

AML_LLMRetStatus aml_llm_set_chat_template(LLMContext context, const char* system_prompt, const char* prompt_prefix, const char* prompt_postfix)
{
    if (!s_func.aml_llm_set_chat_template) return AML_LLM_Status_Failed;
    return s_func.aml_llm_set_chat_template(context, system_prompt, prompt_prefix, prompt_postfix);
}

AML_LLMRetStatus aml_llm_set_chat_template_jinja(LLMContext context, const char* jinja_template)
{
    if (!s_func.aml_llm_set_chat_template_jinja) return AML_LLM_Status_Failed;
    return s_func.aml_llm_set_chat_template_jinja(context, jinja_template);
}

AML_LLMRetStatus aml_llm_set_toolcall_callback(LLMContext context, LLM_ToolUseCallback callback, int stop_output)
{
    if (!s_func.aml_llm_set_toolcall_callback) return AML_LLM_Status_Failed;
    return s_func.aml_llm_set_toolcall_callback(context, callback, stop_output);
}

AML_LLMRetStatus aml_llm_get_vision_info(LLMContext context, AML_LLMVisionInfo* vision_info)
{
    if (!s_func.aml_llm_get_vision_info) return AML_LLM_Status_Failed;
    return s_func.aml_llm_get_vision_info(context, vision_info);
}
