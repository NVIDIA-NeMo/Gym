# OpenCode visual sandboxed agent

`opencode_sandboxed_agent` with image input. The harness, sandbox lifecycle and verification flow
are the same. Only two things change:

- **Images the model reads reach the model.** When the model reads a PNG or JPEG (for example a
  screenshot of its own page, or a reference design), the image is sent to the model server as
  an `image_url` part.
- **Task prompts may carry images.** `input_image` parts in `responses_create_params.input` are
  written into the sandbox and attached to the first turn with `opencode run --file`.

Used by `resources_servers/visual_agent`.

## Why the base agent never sends images

We checked this against OpenCode 1.17.11 (the pinned version)
using a request-logging mock of the model endpoint:

| setup | what the model receives when it reads `red.png` |
|---|---|
| base agent config (no `modalities`) | a user message `Attached media from tool result:` + the text `ERROR: Cannot read image (this model does not support image input). Inform the user.` |
| `modalities.input = [text, image]`, `attachment: true` | a user message `Attached media from tool result:` + an `image_url` part (`data:image/png;base64,...`) |
| `--file red.png`, image modality declared | the first user message gets synthetic `Called the Read tool...` text + an `image_url` part |
| `--file red.png`, no image modality | the same, with the error string instead of the image |

`@ai-sdk/openai-compatible` cannot put media inside a `role: tool` message, so OpenCode moves images into
a user message that follows the tool result. Gym's chat-completions schema already accepts
`image_url` parts in user messages (`NeMoGymChatCompletionUserMessageParam`), and `vllm_model`
passes them through to vLLM unchanged. No change to Gym core is needed. What the model serving
needs:

- A vision model served with its vision tower. For example, drop `--language-model-only` for
  Qwen3.8-Flash-Next (see `benchmarks/nemotron_3.5_super/vllm_configs/qwen_3.8_flash_next_vision.sh`).
- A `--limit-mm-per-prompt` large enough for a session's screenshots.

A text-only model such as GLM-5.3 cannot use these images: its chat template replaces every image
part with a "you are unable to process this image" reminder.

Two gaps in the base agent that this subclass fixes:

- `responses()` asserts the user message has exactly one content part, so a prompt with an
  `input_image` crashes it.
- `_opencode_export_to_output_items` drops user `file` parts and tool-result `attachments`, so the
  rebuilt rollout does not show the images the model saw.

The only change to the base agent is a no-op hook, `_opencode_run_extra_args`, which lets a
subclass add `opencode run` flags.

## Config

| key | default | meaning |
|---|---|---|
| `enable_image_input` | `true` | declare `modalities.input = [text, image]` and `attachment: true` for the model |
| `prompt_image_dir` | `/tmp/nemo_gym_prompt_images` | where prompt images are written in the sandbox |
| `max_prompt_image_bytes` | 20 MiB | per-image cap; larger images fail the request |
| `rollout_image_mode` | `reference` | `inline` keeps the data URL in rollout output items; `reference` stores `sha256:<digest>` to keep rollouts small |

Prompt images must be base64 `data:image/...` URLs. Remote URLs are rejected, so a rollout never
depends on fetching from the network.

## Tests

```bash
pytest responses_api_agents/opencode_visual_sandboxed_agent/tests
```
