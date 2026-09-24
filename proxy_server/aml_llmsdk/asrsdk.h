/*
* Copyright (C) 2026 Amlogic, Inc. All rights reserved.
*
* This source code is subject to the terms and conditions defined in the
* file 'LICENSE' which is part of this source code package.
*
* Description: aml_llmsdk ASR C API (Whisper / SenseVoice)
*
* Parallel to aml_llm_*; do not reuse aml_llm_run() for audio.
* HTTP OpenAI JSON (/v1/audio/transcriptions) is assembled by the Python
* server from AML_ASRResult; this header only defines the native contract.
*/

#ifndef _ASRSDK_H_
#define _ASRSDK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef ASRContext
 * @brief Handle for one loaded ASR engine (Whisper encoder+decoder or SenseVoice).
 */
typedef void* ASRContext;

/**
 * @enum AML_ASRRetStatus
 * @brief Return status of an ASR SDK API call.
 */
typedef enum {
    AML_ASR_Status_Success = 0, /**< The API call completed successfully. */
    AML_ASR_Status_Failed  = 1, /**< The API call failed. */
} AML_ASRRetStatus;

/**
 * @enum AML_ASRModelType
 * @brief Selects which on-device ASR graph to load at init time.
 */
typedef enum {
    AML_ASR_MODEL_WHISPER    = 0, /**< Whisper encoder+decoder ADLA (e.g. large-v3-turbo). */
    AML_ASR_MODEL_SENSEVOICE = 1, /**< SenseVoice Small single ADLA (3-input w8a16). */
} AML_ASRModelType;

/**
 * @enum AML_ASRAudioType
 * @brief How audio is supplied to aml_asr_transcribe().
 */
typedef enum {
    AML_ASR_AUDIO_FILE  = 0, /**< WAV file path. SDK reads PCM; 16 kHz mono recommended. */
    AML_ASR_AUDIO_PCM16 = 1, /**< Caller-owned 16-bit PCM. Valid until transcribe() returns. */
} AML_ASRAudioType;

/**
 * @struct AML_ASRInitConfig
 * @brief Parameters for aml_asr_init(). Zero the struct before filling fields.
 *
 * Path usage:
 * - Whisper:    model_path = encoder.adla, decoder_path = decoder.adla,
 *               tokenizer_path = tokenizer dir or tokenizer_info.bin directory.
 * - SenseVoice: model_path = *.adla, decoder_path = NULL,
 *               tokenizer_path = tokens.txt.
 *
 * extra_json is optional engine knobs. Pass NULL or "" for defaults.
 * Unknown keys should be ignored by the SDK.
 * SenseVoice: {"use_itn":0,"pad_mode":"edge","max_frames":0,"decode_extra_frames":24,"swap_int_inputs":0}
 * Whisper:    {"filters_path":"/path/to/data.bin","n_threads":8}
 *             filters_path overrides tokenizer_path/data.bin when tokenizer_path is a directory.
 */
#pragma pack(push, 1)
typedef struct {
    AML_ASRModelType model_type; /**< Which ASR architecture to initialize. */
    const char* model_path;      /**< Primary ADLA path (Whisper encoder or SenseVoice model). */
    const char* decoder_path;    /**< Whisper decoder ADLA. NULL for SenseVoice. */
    const char* tokenizer_path;  /**< Whisper tokenizer assets or SenseVoice tokens.txt. */
    const char* extra_json;      /**< Optional JSON object for engine-specific init knobs. */
    uint8_t reserved[1024 - sizeof(AML_ASRModelType) - sizeof(const char*) * 4];
} AML_ASRInitConfig;
#pragma pack(pop)

/**
 * @struct AML_ASRInput
 * @brief One transcription request. Zero the struct before filling fields.
 *
 * language: ISO-639-1 such as "en" / "zh", or "auto" / NULL.
 *   Whisper multilingual: "auto"/NULL runs a decoder language-detect step; other codes map to language tokens.
 *   SenseVoice: mapped to language-id tensor (auto/zh/en/ja/ko/yue/nospeech).
 * libnnsdk.so is loaded at runtime (AML_NNSDK_PATH or libnnsdk.so).
 * task: "transcribe" (default, NULL) or "translate" (Whisper only; may fail if unsupported).
 *
 * PCM must be 16 kHz mono int16 when audio_type is AML_ASR_AUDIO_PCM16.
 * FILE input should also be 16 kHz mono WAV for v1 (no resample required of the SDK).
 */
#pragma pack(push, 1)
typedef struct {
    AML_ASRAudioType audio_type; /**< FILE or PCM16. */
    const char* file_path;       /**< Used when audio_type is AML_ASR_AUDIO_FILE. */
    const int16_t* pcm;          /**< Used when audio_type is AML_ASR_AUDIO_PCM16. */
    uint32_t num_samples;        /**< PCM sample count (int16 elements), ignored for FILE. */
    uint32_t sample_rate;        /**< PCM sample rate. 16000 expected; 0 means 16000. */
    const char* language;        /**< "en", "zh", "auto", ... NULL = engine default / auto. */
    const char* task;            /**< "transcribe" or "translate". NULL = transcribe. */
    uint8_t reserved[1024 - sizeof(AML_ASRAudioType)
                          - sizeof(const char*) * 3
                          - sizeof(const int16_t*)
                          - sizeof(uint32_t) * 2];
} AML_ASRInput;
#pragma pack(pop)

/**
 * @struct AML_ASRResult
 * @brief Transcription output filled by aml_asr_transcribe().
 *
 * text and language are SDK-owned. They remain valid until aml_asr_free_result()
 * or the next transcribe() on the same context. Copy them if they must outlive that.
 *
 * Segments / timestamps are not populated in v1; reserved is for that ABI growth.
 */
#pragma pack(push, 1)
typedef struct {
    const char* text;     /**< Transcript UTF-8. Never NULL on success (may be empty). */
    const char* language; /**< Language actually used or detected, e.g. "en". May be NULL. */
    uint8_t reserved[1024 - sizeof(const char*) * 2];
} AML_ASRResult;
#pragma pack(pop)

/**
 * @brief Load ASR models and tokenizer into a new context.
 *
 * @param context Receives the new handle on success. Caller passes a pointer to ASRContext.
 * @param config  Init configuration. Must not be NULL.
 * @return AML_ASR_Status_Success or AML_ASR_Status_Failed.
 */
AML_ASRRetStatus aml_asr_init(ASRContext* context, const AML_ASRInitConfig* config);

/**
 * @brief Run one synchronous transcription.
 *
 * Not re-entrant on the same context. Do not overlap with aml_llm_run() on the
 * same NPU without external locking.
 *
 * On success, fill result and return Success. On failure, result is zeroed.
 *
 * @param context Initialized ASR context.
 * @param input   Audio + language. Must not be NULL. PCM/file buffers stay valid until return.
 * @param result  Output. Must not be NULL. Call aml_asr_free_result() when done.
 * @return AML_ASR_Status_Success or AML_ASR_Status_Failed.
 */
AML_ASRRetStatus aml_asr_transcribe(ASRContext context, const AML_ASRInput* input, AML_ASRResult* result);

/**
 * @brief Release strings owned by a previous aml_asr_transcribe() result.
 *
 * Safe to call with a zeroed result. After return, pointers in result are invalid.
 *
 * @param result Result previously filled by aml_asr_transcribe().
 * @return AML_ASR_Status_Success or AML_ASR_Status_Failed.
 */
AML_ASRRetStatus aml_asr_free_result(AML_ASRResult* result);

/**
 * @brief Unload models and free the ASR context.
 *
 * @param context Handle from aml_asr_init(). NULL is a no-op success.
 * @return AML_ASR_Status_Success or AML_ASR_Status_Failed.
 */
AML_ASRRetStatus aml_asr_uninit(ASRContext context);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _ASRSDK_H_ */
