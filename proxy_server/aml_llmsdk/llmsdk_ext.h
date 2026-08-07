#ifndef __LLMSDK_EXTEND_H__
#define __LLMSDK_EXTEND_H__

#define LLMSDK_ENABLE_EXTSION

#ifdef __cplusplus

#include <vector>
#include <string>
#include <cstdint>

#include "byte_buffer.hpp"

struct LLMVisionInput {
    bool image_tensor;              /**< image input is tensor or bytes(RGB u8). */
    ByteBuffer image_input;         /**< Image tensor vision/projector model. */
    uint16_t image_width;           /**< Original or preprocessed image width */
    uint16_t image_height;          /**< Original or preprocessed image height */
    uint32_t image_timestamp;       /**< Optional timestamp for the image input （unit ms) . */
};

/** Internal tensor input that maps 1:1 to an ADLA vision model input descriptor. */
struct LLMVisionTensorInput
{
    int32_t index;
    const void* data;                   /**< The pointer to int32_buffer.data() or float_buffer.data() */
    int32_t size;
    std::vector<int32_t> int32_buffer;  /**< Backing storage for int32 auxiliary inputs */
    std::vector<float> float_buffer;    /**< Backing storage for float auxiliary inputs */
};

struct LLMVisionEmbeddings{
    std::vector<float> image_embeddings;  /**< Flattened float embeddings from vision model output */
    int32_t n_image_tokens;                /**< Number of image tokens (patches) in the embedding */    
    int32_t mrope_width;                     /**< Width of the M-RoPE grid (if applicable) */
    int32_t mrope_height;                    /**< Height of the M-RoPE grid (if applicable) */
    int32_t image_timestamp;
};


class VLM_VisionProjector {
public:
    virtual ~VLM_VisionProjector() = default;

    /**
    * Process the vision input (image byte array to tensors).
    * @param input           raw RGB image input (image_tensor=false means raw bytes)
    * @param vision_input    output: all tensor inputs for vision model (index 0..n-1)
    * @param out_image_width output: preprocessed image width fed to vision model
    * @param out_image_height output: preprocessed image height fed to vision model
    * @return 0 on success, -1 on error
    */
    virtual int prepare_vision_input(const LLMVisionInput& input, std::vector<LLMVisionTensorInput>& vision_input,
                                     int& out_image_width, int& out_image_height) = 0;
    virtual bool require_mrope() const = 0;
    virtual int build_mrope(std::vector<int32_t>& rope, int32_t current_token, int32_t image_width, int32_t image_height, int32_t timestamp = 0) = 0;
    virtual int postprocess_embeddings(LLMVisionEmbeddings& embeddings) = 0;
    virtual std::string append_vision_time(int32_t timestamp) { return ""; }
    virtual int patch_size() const { return 1; }
    virtual int merge_ratio() const { return 1; }
};

#else
typedef void* VLM_VisionProjector;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum AML_LLMSamplingMode
 * @brief Defines available sampling strategies for output generation.
 */
typedef enum {
    AML_LLM_ARG_Max,       /**< Argmax sampling. Always selects the token with the highest probability. Deterministic and reproducible. */
    AML_LLM_TOP_P,         /**< Top-P (nucleus) sampling. Samples from the smallest set of tokens whose cumulative probability exceeds threshold p. */
    AML_LLM_TOP_K,         /**< Top-K sampling. Samples from the top k tokens with highest probabilities. Balances diversity and control. */
    AML_LLM_CHAIN_SAMPLER, /**< Configurable sampler chain controlled by sampler_params_json. */
} AML_LLMSamplingMode;

/**
 * @struct AML_LLMInitExtend
 * @brief Additional optional parameters for LLM init.
 */
#pragma pack(push, 1)
typedef struct {
    const char* vision_model_path;                    /**< Optional vision model path. Set NULL for text-only LLM. */
    const char* sampler_params_json;                  /**< JSON string configuring AML_LLM_CHAIN_SAMPLER. Pass NULL or "" for defaults. */
    const char* model_name_override;                  /**< Temp: override model_type (e.g. "QWEN") when ADLA misdetects. NULL = use ADLA value. */
    const char* custom_jinja_template;                /**< Override the jinja template in model */
    VLM_VisionProjector* mmproj_input;                /**< User-custom VLM_VisionProjector. NULL = use built-in. */
    const char* mmproj_config;                       /**< Optional JSON config for user-custom VLM_VisionProjector. NULL = use built-in defaults. */ 
    uint8_t reserved[1024 - sizeof(const char*) * 5 - sizeof(void*)];  /**< Reserved for future extension of init configuration. */
} AML_LLMInitExtend;
#pragma pack(pop)

/**
 * @struct AML_LLMRunExtend
 * @brief Additional optional parameters for one LLM inference.
 */
#pragma pack(push, 1)
typedef struct {
    uint32_t sampling_config_valid;               /**< Set to 1 to use all sampling fields below, including sampler_params_json, for this run; set to 0 to use init-time sampling defaults. */
    AML_LLMSamplingMode sampling_mode;            /**< Runtime sampling strategy for this run. */
    int32_t top_k;                                /**< Runtime Top-K sampling parameter for this run. */
    float top_p;                                  /**< Runtime Top-P sampling parameter for this run. */
    float temperature;                            /**< Runtime temperature for this run. */
    float repeat_penalty;                         /**< Runtime repeat penalty for this run. */
    const char* sampler_params_json;              /**< Per-run chain-sampler JSON applied over init defaults when sampling_config_valid is 1. NULL or "" uses init defaults. */
    int32_t disable_chat_template;                /**< If 1, disables the chat template feature. */
    const char* extra_template_params;            /**< Optional JSON object with extra template params (e.g. {"enable_think":true}). */
    const char* tools_schemas;
    uint8_t reserved[1024 - sizeof(uint32_t)
                          - sizeof(AML_LLMSamplingMode)
                          - sizeof(int32_t) * 2
                          - sizeof(float) * 3
                          - sizeof(const char*) * 3]; /**< Reserved for future extension of run configuration. */
} AML_LLMRunExtend;
#pragma pack(pop)

/**
 * @enum AML_LLMImageType
 * @brief Specifies the image representation accepted by VLM input.
 */
typedef enum {
    AML_LLM_IMAGE_TYPE_BUFFER = 0, /**< Input image is provided as a raw buffer (u8)(e.g., RGB or BGR). */
    AML_LLM_IMAGE_TYPE_TENSOR = 1, /**< Input image is provided as a tensor (f32). */
    AML_LLM_IMAGE_TYPE_FILE   = 2, /**< Input image is provided as a file path. */
    AML_LLM_IMAGE_TYPE_BASE64 = 3  /**< Input image is a base64 encoded file. */
} AML_LLMImageType;

/**
 * @enum AML_LLMInputType
 * @brief Specifies the format of input data accepted by the LLM.
 */
typedef enum {
    AML_LLM_INPUT_PROMPT = 0,     /**< Input is a plain text prompt string. The SDK applies chat template and tokenizer before inference. */
    AML_LLM_INPUT_MULTIMODAL = 1, /**< Input contains text plus RGB888 image buffers. The SDK handles preprocessing and vision inference internally. */
    AML_LLM_INPUT_TOKEN = 2,      /**< Input is final token ids. The SDK skips chat template and tokenizer completely. */
    AML_LLM_INPUT_MESSAGES = 3,    
} AML_LLMInputType;

#include "llmsdk.h"

/**
 * @struct AML_LLMVisionInput
 * @brief One vision model input (used for internal VLM preprocessing and tests).
 */
typedef struct {
    uint16_t image_id;              /**< Unique identifier for the image input. */
    AML_LLMImageType image_type;    /**< Type of the image input. */
    const void* image_input;        /**< Pointer to image data (raw buffer or tensor). */
    uint16_t image_width;           /**< Original or preprocessed image width. */
    uint16_t image_height;          /**< Original or preprocessed image height. */
    uint16_t image_channels;        /**< Number of image channels (typically 3 for RGB). */
    uint32_t image_timestamp;       /**< Optional timestamp for the image input (unit ms). */
} AML_LLMVisionInput;

/**
 * @enum AML_LLMVisionDataType
 * @brief Data type of a vision model input tensor.
 */
typedef enum {
    AML_LLM_VISION_TYPE_INVALID = -1, /**< Unknown or unsupported tensor data type. */
    AML_LLM_VISION_TYPE_UINT8 = 0,    /**< Unsigned 8-bit integer. */
    AML_LLM_VISION_TYPE_INT8 = 1,     /**< Signed 8-bit integer. */
    AML_LLM_VISION_TYPE_UINT16 = 2,   /**< Unsigned 16-bit integer. */
    AML_LLM_VISION_TYPE_INT16 = 3,    /**< Signed 16-bit integer. */
    AML_LLM_VISION_TYPE_UINT32 = 4,   /**< Unsigned 32-bit integer. */
    AML_LLM_VISION_TYPE_INT32 = 5,    /**< Signed 32-bit integer. */
    AML_LLM_VISION_TYPE_UINT64 = 6,   /**< Unsigned 64-bit integer. */
    AML_LLM_VISION_TYPE_INT64 = 7,    /**< Signed 64-bit integer. */
    AML_LLM_VISION_TYPE_FP16 = 8,     /**< 16-bit floating point. */
    AML_LLM_VISION_TYPE_FP32 = 9,     /**< 32-bit floating point. */
} AML_LLMVisionDataType;

/**
 * @struct AML_LLMVisionTensorInfo
 * @brief Shape, type, and size of one vision model input tensor.
 */
typedef struct {
    int32_t index;              /**< Vision model input index. Use the same value in AML_LLMVisionInput::index. */
    char name[128];             /**< Vision model input tensor name. */
    AML_LLMVisionDataType type; /**< Vision tensor data type. */
    int32_t height;             /**< NHWC tensor height. */
    int32_t width;              /**< NHWC tensor width. */
    int32_t channels;           /**< NHWC tensor channel count. */
    int32_t size;               /**< Tensor buffer size in bytes. */
} AML_LLMVisionTensorInfo;

/**
 * @struct AML_LLMVisionMetadata
 * @brief Metadata used by the application to preprocess images for the vision model.
 */
typedef struct {
    char projector_type[64];    /**< Vision projector type, e.g. qwen2vl_merger or qwen2.5vl_merger. */
    float image_mean[3];        /**< Per-channel image normalization mean in RGB order. */
    float image_std[3];         /**< Per-channel image normalization std in RGB order. */
    int32_t patch_size;         /**< Vision patch size used by image preprocessing and auxiliary tensor generation. */
    int32_t spatial_merge_size; /**< Spatial merge size used by the vision projector. */
} AML_LLMVisionMetadata;

/**
 * @struct AML_LLMVisionInfo
 * @brief Vision model information queried after aml_llm_init().
 */
typedef struct {
    int32_t n_input;                      /**< Actual number of vision model inputs. */
    const AML_LLMVisionTensorInfo* input; /**< SDK-owned input tensor info array. Valid until aml_llm_uninit. */
    AML_LLMVisionMetadata metadata;       /**< Vision preprocessing metadata. Valid only when has_metadata is non-zero. */
    int32_t has_metadata;                 /**< Non-zero when metadata was found in the vision model. */
} AML_LLMVisionInfo;



/**
 * @brief Query vision model input tensors and preprocessing metadata.
 *
 * This API is available only when AML_LLMInitConfig::init_extend.vision_model_path
 * was set during aml_llm_init(). The returned pointers are owned by the SDK and
 * remain valid until aml_llm_uninit().
 *
 * @param context The LLM context.
 * @param vision_info Output structure filled with SDK-owned vision information.
 * @return Status code (0 on success; non-zero indicates an error).
 */
AML_LLMRetStatus aml_llm_get_vision_info(LLMContext context, AML_LLMVisionInfo* vision_info);

/**
 * @brief Get the Jinja template string registered on the LLM context.
 *
 * Returns a pointer to the internally-owned Jinja template (set either via
 * aml_llm_set_chat_template_jinja or loaded from the model during init).
 * The returned pointer is valid until the context is uninitialized.
 *
 * @param context  The LLM context.
 * @param jinja_template_out Output parameter receiving the template pointer (or NULL).
 * @return AML_LLM_Status_Success always (for now).
 */
AML_LLMRetStatus aml_llm_set_chat_template_jinja(LLMContext context, const char* jinja_template);

/**
 * @param tool_name The name of tool
 * @param tool_args The json string of tool arguments
 * @param userdata Pointer to user-defined data passed to the callback.
 */
typedef void(*LLM_ToolUseCallback)(const char* tool_name, const char* tool_args, void* userdata);


/**
 * 
 * @param context  The LLM context.
 * @param callback The pointer of function, set NULL to cancel this callback.
 * @param stop_output set to 1 to pause callback to LLMResultCallback (stop display tool call string)
 */
AML_LLMRetStatus aml_llm_set_toolcall_callback(LLMContext context, LLM_ToolUseCallback callback, int stop_output);

#ifdef __cplusplus
}
#endif

#endif