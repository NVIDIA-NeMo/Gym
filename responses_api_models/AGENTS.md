# Model servers

Each directory here is a model server built on `nemo_gym.models`. The contract they implement, and the tests and
dependency sources to check, are in [`nemo_gym/models/AGENTS.md`](../nemo_gym/models/AGENTS.md). User docs are under
[Model server](../fern/versions/latest/pages/model-server/index.mdx); each server's own README covers its settings.

What model server changes commonly get wrong:

- Serving generation from a route of their own instead of `chat_completions()` or `responses()`, which skips capture
  and the synthesized streaming, Anthropic Messages and embeddings routes.
- Overriding `setup_webserver` without starting from `super().setup_webserver()`.
- Catching a provider's HTTP error and answering 500, or raising it without `raise_for_status`. Agents then lose the
  provider's status and body, including context-length and rate-limit errors.
- Calling providers with an SDK or `httpx` client instead of `NeMoGymAsyncOpenAI` or `nemo_gym.client.request`.
- Adding a route that a rollout's training capture prefix can reach without declaring it in
  `non_generating_model_routes`, so the rollout is marked incomplete; or declaring a route there whose response can
  carry generated text.
- Pinning a different vLLM version from the `vllm` extra in the root `pyproject.toml`.
