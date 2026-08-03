# Description

A Responses-API **model server** backed by an [SGLang](https://github.com/sgl-project/sglang)
server's native `/generate` endpoint.

It subclasses `vllm_model`'s `VLLMModel` and overrides **only** `chat_completions`. Unlike the
OpenAI-compatible `/v1/chat/completions` path (which can only surface logprobs as `token_id:NNN`
string tokens, and not at all for some serving stacks), this server talks to SGLang's native
`/generate` with `return_logprob=true`, so it recovers the **exact generated token ids and their
logprobs**. That makes it suitable as the policy model server for RL training (GRPO etc.), where
the trainer needs the precise token sequence + logprobs the policy emitted, not a re-tokenized
decode of the text.

What it does per request:

1. Renders the prompt to token ids with the model's own HF chat template (local tokenizer).
2. Caps the prompt to the context window and shrinks `max_new_tokens` so `input + gen < context`.
3. POSTs `{base_url}/generate` with those `input_ids` and parses
   `meta_info.output_token_logprobs` into `(generation_token_ids, generation_log_probs)`.
4. Attaches `prompt_token_ids` / `generation_token_ids` / `generation_log_probs` to the assistant
   message (when `return_token_id_information: true`), exactly as the vLLM path does. The graded
   `content` is decoded with `skip_special_tokens=True`; the raw token ids are preserved for training.

Everything else — Responses<->ChatCompletions conversion, `responses()`, and the assistant-message
training-class upgrade — is inherited unchanged from `vllm_model`.

The pure request/response transforms live in `_logic.py` and are unit-tested in `tests/`.

## Config

See `configs/sglang_model_for_training.yaml`. Key fields (beyond `vllm_model`'s):

- `base_url`: bare SGLang server URL(s); the server calls `{base_url}/generate` (not `/v1/...`).
- `context_length`: SGLang context window; keep in sync with the trainer's max sequence length.
- `default_max_new_tokens`: used only when a request carries no `max_(completion_)tokens`.
- `add_generation_prompt`, `trust_remote_code`: forwarded to the local tokenizer / chat template.

## Licensing information

Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- transformers: Apache 2.0
