# vllm_model

Responses API model server backed by a remote vLLM endpoint.

The server exposes Gym's `/v1/responses` and `/v1/chat/completions` and forwards
to vLLM. The `use_completions_api` config flag picks which vLLM endpoint
receives the upstream call:

| `use_completions_api`         | vLLM endpoint hit            | Primary use case                                                  |
|-------------------------------|------------------------------|-------------------------------------------------------------------|
| `false` (default)             | `POST /v1/chat/completions`  | Instruct models. vLLM applies the chat template and runs reasoning- / tool-call parsers server-side. |
| `true`                        | `POST /v1/completions`       | Base (non-instruct) models. Caller renders the full prompt upstream and forwards bytes verbatim. |

## `use_completions_api: false` (default)

Standard path. Configure with `configs/vllm_model.yaml` (rollouts) or
`configs/vllm_model_for_training.yaml` (RL training: `return_token_id_information: true`).

Supports the full schema: tool-calling, multi-turn assistant/tool messages,
images via `image_url` content blocks, and the audio sidechannel
(`metadata.audio_path` / `audio_paths` / `audio_data`).

## `use_completions_api: true`

Use `configs/vllm_model_completions.yaml`. Drives vLLM's text-completions
endpoint instead of chat completions. Two render modes select how the
caller's `messages` list becomes the `prompt` string.

### `render_chat_template: false` (default — raw)

The bytes the caller puts in their message content are the bytes vLLM
sees in `prompt`.

- The messages list must contain at most one optional system message followed
  by exactly one user message. Their content is concatenated with `\n\n` and
  forwarded as `prompt`.
- For `/v1/responses`, a string `input` is accepted directly and forwarded as
  `prompt` after the converter's text-block round-trip.

Rejected on the raw path (with a `ValueError`):

- `tools` — set `render_chat_template: true` (so the chat template can render
  tool definitions into the prompt) or `use_completions_api: false`.
- `metadata.audio_*` — `/v1/completions` is text-only (rejected in both modes).
- multi-turn / assistant / tool messages — render upstream.
- non-text content blocks (images, audio).

### `render_chat_template: true`

Renders the messages list to a prompt string client-side via HF
`AutoTokenizer.apply_chat_template(messages, tokenize=False,
add_generation_prompt=True, tools=..., **chat_template_kwargs)`. The
multi-turn restriction is lifted; tools are passed through to the template
and rendered into the prompt.

| Constraint                | `render_chat_template: false` | `render_chat_template: true` |
|---------------------------|-------------------------------|------------------------------|
| Single user / [system, user] only | required             | **lifted** — full multi-turn ok |
| Assistant / tool turns    | rejected                      | **allowed**                   |
| Tools                     | rejected                      | **allowed** (rendered into prompt; **tool-call output text is NOT parsed by Gym** — `/v1/completions` doesn't run vLLM's tool-call parser, so the caller is on their own to parse tool calls out of the assistant text) |
| Audio / image content     | rejected                      | rejected                      |
| `chat_template_kwargs`    | unused                        | passed through                |

The HF tokenizer is loaded once at server startup. By default it's loaded
from the same identifier as `model`. Override with the `tokenizer:` config
field — useful for base-model setups where the model checkpoint has no
chat template in its tokenizer config: point `tokenizer:` at a sibling
instruct model whose template you want to inherit.

```yaml
model: Qwen/Qwen2.5-7B           # base model, no chat template
use_completions_api: true
render_chat_template: true
tokenizer: Qwen/Qwen2.5-7B-Instruct  # borrow the instruct chat template
```

If the loaded tokenizer has no `chat_template`, the server fails at startup
with a clear error — silent fallback to raw mode would mask a config bug.

### Common to both render modes

`<think>...</think>` blocks come back inline in the completion text and are
extracted by `VLLMConverter._extract_reasoning_from_content` when results are
converted back to a Response.

When `return_token_id_information: true`, generation token IDs are read from
the native `logprobs.tokens` (`"token_id:<int>"`) entries and prompt token IDs
from `prompt_logprobs` — no separate `/tokenize` round-trip is needed.
