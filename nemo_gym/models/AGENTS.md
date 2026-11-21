# Models: contract note

This note covers `nemo_gym/models`. It is for anyone who changes the package, writes a model server in
`responses_api_models/`, or reviews either. The server runtime contract it builds on is in
[`../server_runtime/AGENTS.md`](../server_runtime/AGENTS.md). User-facing docs are under
[Model server](../../fern/versions/latest/pages/model-server/index.mdx) and
[Model call capture](../../fern/versions/latest/pages/model-server/model-call-capture.mdx).

## Assumptions

- A model server implements two non-streaming methods, `chat_completions()` and `responses()`. The base class owns the
  routes: `/v1/chat/completions`, `/v1/responses`, `/v1/messages` and `/v1/embeddings`.
- Streaming is synthesized. A `stream: true` request is sanitized, served by the same non-streaming method, and the
  complete result is replayed as SSE. The request sent to the backend is never streaming.
- `/v1/messages` is mapped to `responses()` through `AnthropicConverter`, so every model server speaks the Anthropic
  Messages API without its own code. `/v1/embeddings` answers 501 unless a server overrides `embeddings()`.
- Provider calls go through `NeMoGymAsyncOpenAI` in `nemo_gym.client`, which wraps the shared aiohttp client. The
  OpenAI and Anthropic SDKs are used for their types, not for HTTP.
- `models` sits above `server_runtime` and below `agents` in the import-linter layers: it must not import `agents`.

## State and ownership

| State | Owner and lifetime | How ownership moves |
| --- | --- | --- |
| Request and response | The model server, for one call. `_invoke_responses` and `_invoke_chat_completions` resolve the capture parent before the call and record tokens after it. | The base dispatch converts the result to the client's dialect after capture. |
| Model call capture record | The capture middleware, per call prefixed with `/ng-rollout/<id>`. Off by default; it never changes the response. | Written to the rollout's `CaptureStore`; rollout collection reads it. |
| Training token capture | The token sink for the rollout, under the `/ng-rollout/<id>/training-token-capture` prefix. | The training framework freezes and retires the snapshot. Each step is in the `nemo_gym.models.token_capture` docstring. |
| Policy checkpoint gate | Only an instance with `checkpoint.policy: true`. With several workers, the main process runs the coordinator and each worker links to it. | The checkpoint coordinator drains and resumes the gate; refused calls never register capture intent. |

## Public surface and outside callers

- `_PUBLIC_API` in [`__init__.py`](__init__.py) is the surface for model servers. Token capture has its own public
  packages, `nemo_gym.models.token_capture` and `nemo_gym.models.token_capture.staging`, which import without the
  server stack so training frameworks can implement their protocols.
- Every server in `responses_api_models/` subclasses `SimpleResponsesAPIModel`, directly or through `vllm_model`'s
  `VLLMModel` or `openai_model`'s `SimpleModelServer`.
- NeMo RL (upstream `main`) imports token capture through the old path `nemo_gym.token_id_capture`, which
  [`_moved_paths.py`](../_moved_paths.py) keeps working until its removal release. It uses the staging records,
  capture, digest, rebuild and terminal modules, `adapters.vllm`, `lineage`, `fingerprint`, `completion`,
  `external_capture`, `config` and `replay`. Changes to those names or record shapes land together with NeMo RL.

## Pitfalls

- Do not register a server's own route for generation. Calls outside the base dispatch skip capture. Under the
  training capture prefix, a call to any other route that does not answer 404 or 405 marks the rollout incomplete,
  unless the server lists it in `non_generating_model_routes` or the config's
  `token_id_capture_non_generating_requests`. List a route there only if its response can never contain
  model-generated content.
- An override of `setup_webserver` calls `super().setup_webserver()` and adds to the app it returns. Building a new
  app drops capture, the default routes and the checkpoint gate.
- Raise provider HTTP errors with `raise_for_status`. The base handler forwards the provider's body and status to the
  client, which agents need for context-length and rate-limit handling. Override
  `_should_propagate_upstream_http_error` to narrow that, as `openai_model` does.
- `stream_keepalive_s` names the response before it exists, so a server whose response IDs come from an upstream
  Responses API returns `False` from `_mints_response_ids` and the setting is refused at startup.
- An engine refusal propagated on purpose is marked with `PROPAGATED_REFUSAL_CODE_ATTRIBUTE`.
  `context_overflow_error_fields` parses both the nested vLLM error body and the older flat one.
- `drop_custom_tools`, `drop_hosted_tools` and `custom_tool_strategy` apply only on dispatches that sanitize a
  streaming body, and `function_bridge` only on servers that convert Responses to Chat Completions. The field
  descriptions in [`base_responses_api_model.py`](base_responses_api_model.py) list where each applies.

## Tests

```bash
pytest tests/unit_tests/models tests/unit_tests/client/test_openai_client.py -x -n 16
pytest responses_api_models/<server>/tests -x
```

Changes to capture or token staging also run `tests/unit_tests/models/test_token_capture_conformance.py` and
`test_token_capture_golden_vectors.py`, which pin record shapes that NeMo RL also reads.

## Dependency sources

Check behavior in the source of the version `uv.lock` pins, not the latest release:

```bash
grep -A1 -E '^name = "(openai|anthropic|vllm)"$' uv.lock
```

- OpenAI SDK: the request and response types `nemo_gym.core.responses_api_types` and the converters build on.
- Anthropic SDK: `MessageCreateParams` and the message types in `anthropic_converter.py` and `anthropic_utils.py`.
- vLLM: the OpenAI-compatible server's error body, token-ID and log-probability fields read by
  `token_capture/adapters/vllm.py`, and tool and reasoning parsers. The `vllm` extra in `pyproject.toml` and the
  `local_vllm_model` server pin the same version.
