# claude_messages_model

An Anthropic **Messages** (`/v1/messages`) front-end for a vLLM-served open model.

It lets a closed-source, Anthropic-protocol harness — notably the `claude` CLI used by
`claude_code_agent`, whose binary we cannot edit — drive an open model served by vLLM,
**while NeMo Gym captures the exact token IDs the harness never surfaces**. This is what
makes such harnesses trainable.

## Why it exists

Token IDs (`prompt_token_ids` / `generation_token_ids` / `generation_log_probs`) are a
generation-time artifact of the serving engine; they cannot be reliably recovered after
the fact, and the real Anthropic API never returns them. Harnesses that call the model
behind their own HTTP client drop these fields when deserializing the response. Rather
than patch every harness (impossible for a closed binary), the **model server buffers**
each generation's token IDs as it serves requests, keyed by a per-run correlation token in
the request path, and the **agent reconciles** them back onto its trajectory afterward.
This is a shared capability, not specific to this server: buffering lives in
`nemo_gym.base_responses_api_model.TokenIDBufferingMixin` (used by `VLLMModel`) and
reconciliation in `nemo_gym.base_responses_api_agent` (used by any agent).

## How it works

```
claude CLI ──POST /runs/<token>/v1/messages (stream:true)──▶ claude_messages_model
                                                             │  Anthropic → chat-completions
                                                             ▼
                                                   VLLMModel.chat_completions ──▶ vLLM
                                                   (captures token IDs + logprobs)
                          ◀── Anthropic SSE stream ──────────┤  + buffer under <token>
agent ── GET /runs/<token>/buffered_generations ────────────┘  → reconcile → ...ForTraining
```

The server subclasses `VLLMModel`, so token-ID capture and the OpenAI-native
`/v1/chat/completions` and `/v1/responses` endpoints come for free. It adds:

- `HEAD /` — liveness (the CLI probes this).
- `POST /runs/{token}/v1/messages` — run-scoped; buffers token IDs under `{token}`.
- `POST /v1/messages` — uncorrelated variant for smoke tests (no buffering).
- `GET /runs/{token}/buffered_generations` — pop the run's buffered generations.

## Usage

Point `claude_code_agent.model_server` at this server and run the open model under vLLM:

```bash
ng_run "+config_paths=[\
responses_api_models/claude_messages_model/configs/claude_messages_model.yaml,\
responses_api_agents/claude_code_agent/configs/claude_code_agent.yaml,\
<resources_server>.yaml]" \
  +policy_base_url=http://127.0.0.1:8000/v1 \
  +policy_api_key=EMPTY \
  +policy_model_name=<served-model-name>
```

## Limitations

The buffer is in-process, so a run's `/v1/messages` calls and the agent's pop must reach
the same server process. With path-token correlation this is naturally sticky for a single
replica; multi-replica deployments need token-hash routing or a shared store (follow-up).

Reasoning is kept inline as `<think>...</think>` text rather than synthesized as separate
Anthropic `thinking` blocks (which would require signatures the open model can't produce).
