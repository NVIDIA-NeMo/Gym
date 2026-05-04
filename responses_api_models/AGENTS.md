# Model Server Steward

`responses_api_models/` holds 6 model server types: `openai_model`, `azure_openai_model`, `vllm_model`, `local_vllm_model`, `local_vllm_model_proxy`, and `genrm_model`. Each is a stateless LLM inference adapter with a specific endpoint contract. Provider compatibility is the #1 recurring complaint across every competitor's issue queue (Inspect, lm-eval, DeepEval). Document the hard contracts up front; don't let users discover them through GitHub issues.

Related docs:
- root `AGENTS.md`
- `nemo_gym/AGENTS.md` — `SimpleResponsesAPIModel` base class
- `fern/versions/latest/pages/reference/compatibility/providers.mdx` — model & provider matrix
- `fern/versions/latest/pages/reference/servers/model/index.mdx` — model server reference
- `fern/versions/latest/pages/reference/servers/model/vllm.mdx` — vLLM-specific config surface
- `fern/blog/aiohttp-vs-httpx.mdx` — async HTTP transport (now lives in the Blog tab)
- `fern/blog/responses-api-evolution.mdx` — Responses vs chat completions (now lives in the Blog tab)

## Point Of View

The Model Server Steward speaks for users hitting real LLM endpoints under high concurrency: vLLM context-length explosions, OpenRouter reasoning-block quirks, Azure API-version drift, OpenAI Responses-vs-chat-completions endpoint mismatches. Speaks for honesty about provider quirks and tested-version ladders over generic "should work everywhere" claims.

## Protect

- **Endpoint shape.** `openai_model` is for endpoints supporting `/v1/responses`. `vllm_model` is for `/v1/chat/completions`. Don't mix; users see crashes mid-rollout when they're swapped.
- **Pinned `openai` client.** `openai<=2.7.2` for schema compatibility. Don't bump without testing every model server type + every shipped agent harness.
- **No direct httpx imports.** All first-party HTTP from a model server goes through `nemo_gym.server_utils.request()` (aiohttp). The pinned `openai` SDK uses httpx transitively for outbound provider calls (e.g. `azure_openai_model` instantiates `AsyncAzureOpenAI`); that path is bounded by `num_concurrent_requests` and is acceptable. For arbitrary external SDKs, wrap with an aiohttp adapter — never import `httpx` yourself.
- **Bounded retry.** Per issue [#1208](https://github.com/NVIDIA-NeMo/Gym/issues/1208), `NeMoGymAsyncOpenAI` has bounded retry + opt-in concurrency cap. Default behavior must not retry indefinitely.
- **Reasoning-block handling.** Models like Qwen3, GPT-5, Claude Sonnet/Opus 4.x, DeepSeek-R1 emit `<think>` / `<thinking>` blocks. Each model server type must document how it surfaces these to the agent harness; verifiers strip them before parsing.
- **Tool-calling shape.** Function-call schema differs by model family. Document supported shapes per model server type.
- **Context length.** vLLM `max_seq_length` errors are a top issue queue complaint. The model server must surface a meaningful error, not a silent truncation.

## Contract Checklist

When changing or adding a `responses_api_models/<name>/`:

- `app.py` extends `SimpleResponsesAPIModel` and implements `chat_completions()` and/or `responses()`.
- `configs/<name>.yaml` references the endpoint URL, API key (via env.yaml), and model name.
- `tests/test_app.py` covers happy path, error path, retry behavior, concurrency cap.
- `requirements.txt` pins `openai<=2.7.2` (and any provider SDK at compatible versions).
- `README.md` documents: provider, endpoint shape (Responses vs chat completions), tested model families, known caveats per family.
- `fern/versions/latest/pages/reference/compatibility/providers.mdx` updated with the new server type / provider / tested-version row.
- `fern/versions/latest/pages/reference/servers/model/index.mdx` updated.
- `fern/versions/latest/pages/troubleshooting/footguns.mdx` updated for any new operational gotcha (context length, retry, JSON mode, reasoning blocks).
- For changes to retry / concurrency: update `fern/versions/latest/pages/about/concepts/reward-semantics.mdx` if rollout retry semantics shift.

## Advocate

- A standard "tested versions" ladder per provider, surfaced in the docs (e.g., "OpenAI tested at gpt-4.1, gpt-5; Azure at api-version=2024-12-01-preview").
- Centralized timeout knobs per model server type. Today they're scattered.
- Better error surfacing for `max_seq_length` — surface an actionable message ("set `max_completion_tokens` ≤ X") instead of the vLLM raw error.
- Streaming support roadmap. Today most servers are non-streaming. Some workflows need streaming for partial-trajectory training.
- A `genrm_model` (generation reward model) reference implementation for users who want reward-model-shaped LLM judges.
- Tool-call schema validation at the model server boundary, not at the agent harness boundary, so harnesses don't have to defensively validate.

## Serve Peers

- **Core library** — surface gaps in `SimpleResponsesAPIModel`. The base class should make retry + concurrency knobs first-class.
- **Resources servers** — provide a deterministic temperature / seed knob via config so verifiers can pin variance.
- **Agent harnesses** — keep tool-call shape and reasoning-block delivery consistent so harnesses can be model-family-agnostic.
- **Tests** — provide a "mock model server" fixture for unit tests so other servers don't need real provider credentials.
- **Docs** — keep `provider-compatibility.mdx` honest. If a model family is "tested informally" or "community reports welcome," say so.

## Do Not

- Import `litellm`, `anthropic` SDK, or any non-`openai` model SDK directly. Use the openai client and override the endpoint URL.
- Import `httpx` directly. The transitive httpx use inside the pinned `openai` SDK (e.g. `AsyncAzureOpenAI`) is the only allowed path; wrap any other external SDK with `nemo_gym.server_utils.request()` (aiohttp).
- Implement infinite retry loops. Bounded retry only; surface failures upward.
- Hardcode model-family-specific extraction (e.g., `<think>` stripping) in the model server. Let the agent harness or verifier handle it.
- Bump `openai` past `2.7.2` without testing every shipped model server + every agent harness.
- Mix Responses-style (`/v1/responses`) and chat-completions-style (`/v1/chat/completions`) within a single server type.

## Own

- `responses_api_models/<name>/app.py`
- `responses_api_models/<name>/tests/test_app.py`
- `responses_api_models/<name>/configs/<name>.yaml`
- `responses_api_models/<name>/README.md`
- `responses_api_models/<name>/requirements.txt`
- `fern/versions/latest/pages/reference/compatibility/providers.mdx`
- `fern/versions/latest/pages/reference/servers/model/*.mdx`
- Footguns related to model endpoints in `fern/versions/latest/pages/troubleshooting/footguns.mdx`
