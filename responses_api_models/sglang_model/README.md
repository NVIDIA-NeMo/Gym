# Description

A Responses-API **model server** for policies served by [SGLang](https://github.com/sgl-project/sglang).

RL trainers (GRPO etc.) need the *exact* token ids the policy emitted and their logprobs, not a
re-tokenized decode of the text. `vllm_model` recovers those by parsing `token_id:NNN` logprob
tokens out of `/v1/chat/completions` — a vLLM-specific encoding SGLang does not produce. This
server closes that gap.

It subclasses `vllm_model`'s `VLLMModel`. Everything else — Responses<->ChatCompletions
conversion, `responses()`, and the assistant-message training-class upgrade — is inherited
unchanged.

## Transports

### `transport: chat` (default) — requires **sglang >= 0.5.13**

Drives SGLang's OpenAI-compatible `/v1/chat/completions`, requesting the training metadata via
SGLang's native `return_meta_info` / `return_prompt_token_ids` extensions. Token ids and logprobs
come back on each choice as `meta_info.output_token_logprobs` and `prompt_token_ids`.

These extensions landed with the sglang-miles TITO sync series and are in the tree as of the
0.5.13 release — **no patched build or fork is required**. (ProRL ships a
`patch_sglang_0513_token_metadata.sh`, but that patch only makes `logprobs=true` *imply* those
flags for clients that don't set them; this server sets them explicitly, so the patch is
unnecessary here.)

This is the path to use whenever you can. Only the token extraction is overridden, so prompt
templating, **tool-call parsing**, sampling parameters, auth, and context-overflow handling are
all done server-side by SGLang — there is no local tokenizer that can drift from the server's.

### `transport: generate` — fallback for older builds

Drives SGLang's native `/generate` with `return_logprob=true`, tokenizing the prompt locally.
Use only for SGLang builds/forks predating chat-side TITO. Inherent limitations of this path:

- **Tool calls are not parsed** out of the generated text (tool *schemas* are rendered into the
  prompt only if the local chat template does so).
- The prompt is tokenized by a **local** copy of the chat template, so `model` must be the exact
  path/revision the SGLang server was launched with — otherwise generation is silently
  conditioned on ids from the wrong template.
- Sampling params `/generate` does not accept (`n`, `seed`, `response_format`, `logit_bias`) are
  logged and dropped rather than honored.
- A prompt that overflows `context_length` yields an empty completion with
  `finish_reason="length"` (matching the inherited vLLM behavior) so it is filterable, rather
  than a head-truncated prompt that would generate from a malformed context.

In both transports, a generation SGLang reports as `finish_reason="abort"` raises instead of
being emitted, so a server-cancelled partial rollout cannot enter a training batch looking like
a normal completion.

## On-policy scope

Single-turn rollouts are exactly on-policy: the ids attached to the assistant message are the
ids the policy emitted.

**Multi-turn is not yet on-policy.** Turn N re-renders the whole history through the chat
template, which re-tokenizes prior assistant spans; those generally differ from the
`generation_token_ids` the policy actually produced. The fix is to splice the carried-forward
token ids and tokenize only the newly inserted environment/user messages — tracked as follow-up
work, not yet implemented here.

## Config

- `transport`: `chat` (default) or `generate`.
- `base_url`: **must end in `/v1` for `transport: chat`** (like `vllm_model`), and must be the
  **bare** server URL for `transport: generate` (the server appends `/generate`). Migrating a
  bare URL to the chat transport 404s.
- `context_length` (generate only): SGLang context window; keep in sync with the trainer's max
  sequence length.
- `default_max_new_tokens` (generate only): used only when a request carries no
  `max_(completion_)tokens`.
- `add_generation_prompt`, `trust_remote_code` (generate only): forwarded to the local
  tokenizer / chat template. `trust_remote_code` defaults to `false`.

See `configs/sglang_model_for_training.yaml` and
`configs/sglang_model_for_training_generate.yaml`.

## Licensing information

Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- transformers: Apache 2.0
