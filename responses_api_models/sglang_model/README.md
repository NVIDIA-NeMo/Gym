# SGLang model server

This Responses-API model server connects NeMo Gym to an SGLang server managed
outside Gym. It preserves the exact prompt IDs, sampled token IDs, and sampled
token logprobs required by token-level RL training.

The adapter currently renders the chat template locally and calls SGLang's
native `/generate` endpoint with `return_logprob=true`. Gym's public
Chat-Completions response contract does not guarantee exact sampled integer
token IDs, so moving this transport to `/v1/chat/completions` would currently
lose a required training invariant. If that endpoint gains a stable
token-ID/logprob contract, the transport can change without changing the
session-splice or context-overflow rules below.

For a multi-turn session, the adapter caches the token sequence and splices
each prior assistant turn's exact sampled IDs into the next prompt. It never
re-tokenizes those sampled turns. If a prompt already fills `context_length`,
the adapter returns a terminal response with `finish_reason="length"` instead
of truncating the prefix. Both behaviors preserve the trainer's prefix
contiguity invariant.

The session cache is process-local. Run one Gym worker per model-server
instance, or provide sticky routing that keeps every turn of a session on the
same worker.

## Configuration

See `configs/sglang_model_for_training.yaml`.

- `base_url`: the SGLang URL; either a bare server URL or one ending in `/v1`.
- `model`: a tokenizer/model identifier available to the Gym server.
- `context_length`: the SGLang server's total context limit.
- `sglang_chat_template`: optional inline copy of the server/training template.
- `sglang_chat_template_path`: optional path to the exact server chat template.
- `sglang_tool_format`: `hermes` or `qwen3_coder`.
- `trust_remote_code`: forwarded to the local tokenizer loader; defaults to
  `false`.

`transformers` is intentionally not pinned at this leaf package. Use the
environment lockfile to pin a validated release for reproducible deployment.
The example config leaves `context_length` mandatory (`???`) so a mismatched
server limit cannot be selected silently.

## CPU-only tests

From the Gym checkout:

```bash
uv run --extra dev pytest responses_api_models/sglang_model/tests
```

The direct tests mock the tokenizer and SGLang HTTP client; they do not load
weights or require a GPU.
