#ifndef __ASRSDK_FUNC_H__
#define __ASRSDK_FUNC_H__

#ifdef __cplusplus
extern "C" {
#endif

int load_asrsdk_func(const char* lib_path);
void unload_asrsdk_func(void);

int asrsdk_init(void** context, int model_type, const char* model_path,
                const char* decoder_path, const char* tokenizer_path,
                const char* extra_json);
int asrsdk_transcribe_file(void* context, const char* wav_path, const char* language,
                           const char* task, char** text_out, char** language_out);
int asrsdk_uninit(void* context);
void asrsdk_free_cstr(char* s);

#ifdef __cplusplus
}
#endif

#endif
