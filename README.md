# AML LLM OpenAI Server

An OpenAI-compatible API server for Amlogic LLM devices, with built-in tool-calling
evaluation and Agent capability benchmarking tools.

- **Server**: OpenAI-compatible `/v1/chat/completions` endpoint (ADLA & llama.cpp backends)
- **Tools Evaluation**: Measure tool-calling accuracy of local vs cloud models
- **Agent Benchmark**: Evaluate LLM agent capabilities on MMLU-Pro, TAU2-Bench, and BFCL

---

### Prerequisites

This project is using `uv` please use following command or follow the [official document](https://docs.astral.sh/uv/getting-started/installation/) to install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install the wheel

```bash
uv venv --python 3.11
uv pip install amlllm-openai-server -i <index url>
```

### Bootstrap

#### Use a bootstrap tool

_You need Internet connection to use bootstrap scripts to download the models or you can manually put files in folder_

1. List the available model

```bash
uv run -m amlllm_openai_server.bootstrap list
```

You will see the available models as following table

| \# | Name | Type | Thinking | Multimodal | Context | Tools |
|----|------|------|----------|------------|---------|-------|
| 0 | Qwen3-1.7B | Qwen3 | Yes | No | 8192 | Yes |
| 1 | Qwen3-4B-Instruct-2507 | Qwen3 | No | No | 8192 | Yes |
| 2 | Qwen3-4B | Qwen3 | Yes | No | 4096 | Yes |

2. Download the model

Replace the model name as your expectation. Here use `Qwen3-4B-Instruct-2507` for example.

```bash
uv run -m amlllm_openai_server.bootstrap download Qwen3-4B-Instruct-2507
```

---

## Server Configuration

### `config/server.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  api_key: ""          # optional, set for authentication
  log_level: "info"
  title: "AMLLLM OneAPI Proxy"
  version: "0.2.0"

models:
  root_dir: "../models"
  enabled:
    - default           # model directories under root_dir

# Optional: upstream proxy to a cloud API
# upstream:
#   base_url: "https://api.openai.com"
#   api_key: "sk-..."
#   model: "gpt-4o"
#   force_upstream: false
#   skill_injection: false
```

### `models/default/model.json`

Example for Qwen3 Seriers:

```json
{
  "id": "amlogic-default",
  "weights": "Qwen3-1.7B-Q4AM_PB32.adla",
  "model_type": "qwen",
  "sampling_mode": "top_p",
  "sampler_params": {"temp": 0.7, "top_p": 0.85, "top_k": 40, "penalty_repeat": 1.15},
  "system_prompt": "You are a helpful assistant.",
  "prompt_prefix": "",
  "prompt_postfix": "",
  "retain_history": true,
  "loglevel": "ERROR",
  "metadata": {
    "family": "default",
    "device": "amlogic"
  },
  "chat_format": "{%- if tools %}\n    ... (Qwen3 tool-aware Jinja2 template) ...\n"
}
```

> The example above uses legacy top-level sampling keys for brevity. The
> preferred form is a single `sampler_params` dict, e.g.
> `"sampler_params": {"temp": 0.7, "top_p": 0.85, "top_k": 40, "penalty_repeat": 1.15}`.

### Model Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | `string` | Yes | — | Unique model identifier, used as `model` in OpenAI API requests |
| `weights` | `string` | Yes | — | Path to the model weights file (`.adla` for ADLA, `.gguf` for llama.cpp) |
| `backend` | `string` | No | `"adla"` | Runtime backend: `"adla"` (Amlogic ADLA, AARCH64 only) or `"llama_cpp"` |
| `model_type` | `string` | Yes | — | Model architecture type: `"qwen"`, `"llama"`, `"gemma"`, etc. Determines chat template and tokenization |
| `sampling_mode` | `string` | No | `"chain_sampler"` | Sampling strategy: `"arg_max"` (a.k.a. `"max"` / greedy), `"top_p"`, `"top_k"`, or `"chain_sampler"` (a.k.a. `"chain"`, configurable via `sampler_params`) |
| `sampler_params` | `dict` | No | `{}` | Preferred sampler configuration dict (e.g. `{"temp": 0.7, "top_p": 0.85, "top_k": 40, "penalty_repeat": 1.15}`). Serialized to JSON and passed as `init_extend.sampler_params_json`; per-run overrides are merged at request time. See [Sampler parameters](#sampler-parameters) |
| `top_k` | `int` | No | `40` | Legacy Top-K (auto-converted to `sampler_params.top_k` when `sampler_params` is absent) |
| `top_p` | `float` | No | `0.85` | Legacy Top-P (auto-converted to `sampler_params.top_p` when `sampler_params` is absent) |
| `temperature` | `float` | No | `0.7` | Legacy temperature (auto-converted to `sampler_params.temp` when `sampler_params` is absent) |
| `repeat_penalty` | `float` | No | `1.15` | Legacy repeat penalty (auto-converted to `sampler_params.penalty_repeat`) |
| `presence_penalty` | `float` | No | `0` | Legacy presence penalty (auto-converted to `sampler_params.penalty_present`) |
| `system_prompt` | `string` | No | `""` | Default system prompt prepended to every conversation |
| `prompt_prefix` | `string` | No | `""` | Custom text prepended before the prompt |
| `prompt_postfix` | `string` | No | `""` | Custom text appended after the prompt |
| `stop` | `list[string]` | No | `[]` | Stop sequences that halt generation |
| `retain_history` | `bool` | No | `true` | Whether to keep conversation history across turns (enables KV-cache continuation) |
| `loglevel` | `string` | No | `"ERROR"` | ADLA runtime log level: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |
| `context_size` | `int` | No | `4096` | Model context window size (tokens) |
| `metadata` | `dict` | No | `{}` | Arbitrary key-value metadata exposed in `/v1/models` response |
| `token_pair` | `dict` | No | `{}` | Special token ID mapping, e.g. `{"151657": 151658}` maps `<tool_call>` → `</tool_call>`, the token MUST appear in pair, as  151657 ->xxx -> 151658, otherwise, the generation will be stopped. |
| `chat_format` | `string`  | No | `""` | Jinja2 chat template string **or** path to `tokenizer_config.json`. When set, the template is rendered in Python (`disable_chat_template=1`). When empty, the raw OpenAI messages JSON is handed to the SDK (`AML_LLM_INPUT_MESSAGES`, `disable_chat_template=0`) and the SDK renders its built-in template. See [Jinja2 Chat Templates](#jinja2-chat-templates) |
| `template_kwargs` | `dict` | No | `{}` | Extra parameters passed to the Jinja2 chat template (e.g. `{"enable_thinking": true}`) |
| `mmproj` | `string` | No | `""` | Path to the mmproj (vision/projector) model for VLM (multimodal) support |
| `image_pad` | `string` | No | `"<\|image_pad\|>"` | Placeholder in the prompt marking where image content is inserted |

### Sampler parameters

The `sampler_params` dict is the preferred way to configure sampling. It is
serialized to JSON and passed to the SDK as `init_extend.sampler_params_json`;
per-request overrides (`temperature`, `top_p`, and other recognized keys) are
merged into a per-run `sampler_params_json` with `sampling_config_valid=1`.
Recognized keys include (among others): `temp`, `top_p`, `top_k`, `min_p`,
`typ_p`, `penalty_repeat`, `penalty_present`, `penalty_freq`, `penalty_last_n`,
`mirostat`, `mirostat_tau`, `mirostat_eta`, `seed`, `samplers`,
`grammar_gbnf`. Legacy top-level keys (`top_k`, `top_p`, `temperature`,
`repeat_penalty`, `presence_penalty`) are auto-converted into this dict only
when `sampler_params` is absent.

### Jinja2 Chat Templates

The `chat_format` field uses Jinja2 syntax or a tokenizer config to define how
messages are formatted into the model's native prompt structure. Two rendering
modes are supported, selected automatically from `chat_format`:

- **`chat_format` set** (Jinja template string **or** path to
  `tokenizer_config.json`): the template is rendered in **Python**
  (`disable_chat_template=1`). Tool schemas are merged into the template by
  the renderer, and only the rendered prompt is sent to the SDK. KV-cache
  continuation sends only the delta (new) messages.
- **`chat_format` empty**: the raw OpenAI messages JSON is passed to the SDK
  with `AML_LLM_INPUT_MESSAGES` and `disable_chat_template=0`; the SDK renders
  its built-in Jinja template. Tool schemas are passed separately as
  `run_extend.tools_schemas`.

1. Prepare the model `tokenizer_config.json`
2. For Qwen series, it is possible to put the Jinja template string directly
3. For other series, put the path to `tokenizer_config.json` 

### Tool calling

Tool calls are detected by the SDK's PEG tool-call parser and delivered through
the `on_tool_call` callback (registered automatically at init, mirroring
`on_token`). Parsed calls are the primary source; if the SDK yields none, the
server falls back to parsing tool-call markup from the generated text. No
extra configuration is required beyond passing `tools` in the OpenAI request —
the schemas are forwarded to the SDK as `run_extend.tools_schemas`.

---

## Running the Server

```bash
# Default: reads config/server.yaml
uv run python -m amlllm_openai_server

# Custom config path
uv run python -m amlllm_openai_server --config /path/to/server.yaml

# Override host/port/log-level
uv run python -m amlllm_openai_server --host 0.0.0.0 --port 8000 --log-level debug
```

Test the endpoint:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/healthz
```

### If you want to install PicoClaw

```bash
dpkg -i picoclaw_modified_aarch64.deb
```

### Run demo with PicoClaw

1. Start the AMLLLM server (see above)
2. Start PicoClaw:
   ```bash
   picoclaw-launcher &
   ```
3. Configure PicoClaw model:
   - **Provider**: openai
   - **Model identifier**: `amlogic-default`
   - **API Base URL**: `http://localhost:8000/v1`

**Tips**: The agent work folder is under `$HOME/.picoclaw/workspace`. If you are using
local models with context limitations, it's better to remove `AGENTS.md` and `SOUL.md`.

---

## Chat Completion API Usage Examples

### Python (using the `openai` package)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # optional, set only if configured in server.yaml
)

# Basic text completion
response = client.chat.completions.create(
    model="amlogic-default",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)


# Streaming completion
stream = client.chat.completions.create(
    model="amlogic-default",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about AI."},
    ],
    temperature=0.8,
    max_tokens=256,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()


# Tool / function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Beijing'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="amlogic-default",
    messages=[
        {"role": "user", "content": "What is the weather like in Shanghai?"},
    ],
    tools=tools,
    temperature=0.1,
    max_tokens=512,
)
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
else:
    print(message.content)
```

### JavaScript / TypeScript (using the `openai` npm package)

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed",
});

// Basic text completion
async function basicCompletion() {
  const response = await client.chat.completions.create({
    model: "amlogic-default",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello! What is the capital of France?" },
    ],
    temperature: 0.7,
    max_tokens: 512,
  });
  console.log(response.choices[0].message.content);
}

// Streaming completion
async function streamCompletion() {
  const stream = await client.chat.completions.create({
    model: "amlogic-default",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Write a short poem about AI." },
    ],
    temperature: 0.8,
    max_tokens: 256,
    stream: true,
  });
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
  }
  console.log();
}

// Tool / function calling
async function toolCalling() {
  const response = await client.chat.completions.create({
    model: "amlogic-default",
    messages: [
      { role: "user", content: "What is the weather like in Shanghai?" },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get the current weather for a city",
          parameters: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "City name, e.g. 'Beijing'",
              },
              unit: {
                type: "string",
                enum: ["celsius", "fahrenheit"],
              },
            },
            required: ["location"],
          },
        },
      },
    ],
    temperature: 0.1,
    max_tokens: 512,
  });
  const message = response.choices[0].message;
  if (message.tool_calls) {
    for (const toolCall of message.tool_calls) {
      console.log(
        `Call: ${toolCall.function.name}(${toolCall.function.arguments})`,
      );
    }
  } else {
    console.log(message.content);
  }
}

// Run all examples
basicCompletion();
streamCompletion();
toolCalling();
```

