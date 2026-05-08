# vllm_model

Responses API model server backed by a remote vLLM endpoint.

The server exposes Gym's `/v1/responses` and `/v1/chat/completions` and forwards
to vLLM. Two upstream backends are supported, selected by
`upstream_backend` in the config:

| `upstream_backend`  | vLLM endpoint hit         | Primary use case                                                  |
|---------------------|---------------------------|-------------------------------------------------------------------|
| `chat_completions`  | `POST /v1/chat/completions` | Instruct models. vLLM applies the chat template and runs reasoning- / tool-call parsers server-side. |
| `completions`       | `POST /v1/completions`      | Base (non-instruct) models. Caller renders the full prompt upstream and forwards bytes verbatim. |

## `upstream_backend: chat_completions` (default)

Standard path. Configure with `configs/vllm_model.yaml` (rollouts) or
`configs/vllm_model_for_training.yaml` (RL training: `return_token_id_information: true`).

Supports the full schema: tool-calling, multi-turn assistant/tool messages,
images via `image_url` content blocks, and the audio sidechannel
(`metadata.audio_path` / `audio_paths` / `audio_data`).

## `upstream_backend: completions`

Use `configs/vllm_model_completions.yaml`. Drives vLLM's text-completions
endpoint instead of chat completions. The bytes the caller puts in their
message content are the bytes vLLM sees in `prompt`.

**`render_mode: raw` (default and the only currently supported mode):**

- The messages list must contain at most one optional system message followed
  by exactly one user message. Their content is concatenated with `\n\n` and
  forwarded as `prompt`.
- For `/v1/responses`, a string `input` is accepted directly and forwarded as
  `prompt` after the converter's text-block round-trip.

The following are **rejected** (400-style `ValueError`) on this path:

- `tools` — `/v1/completions` doesn't run vLLM's tool-call parser.
- `metadata.audio_*` — `/v1/completions` is text-only.
- multi-turn / assistant / tool messages — render upstream.
- non-text content blocks (images, audio).

`<think>...</think>` blocks come back inline in the completion text and are
extracted by `VLLMConverter._extract_reasoning_from_content` when results are
converted back to a Response.

When `return_token_id_information: true`, generation token IDs are read from
the native `logprobs.tokens` (`"token_id:<int>"`) entries and prompt token IDs
from `prompt_logprobs` — no separate `/tokenize` round-trip is needed.
