# Description

LiteLLM model server for use with LiteLLM proxy endpoints that expose the OpenAI Responses API (`/v1/responses`).

LiteLLM proxies may return non-standard response formats (e.g. `object="chat.completion"` instead of `"response"`, or `reasoning.effort = "none"` as a string instead of `null`). This server normalizes those responses so downstream NeMo Gym validation succeeds.

Extends `openai_model`'s `SimpleModelServer`.

On startup the server verifies the proxy runs LiteLLM `>= 1.83.0` via `GET /health/readiness/details`. When that route answers 401/403 (for example a virtual key scoped to `llm_api_routes`), the version is read from the `x-litellm-version` header of a one-token `POST /v1/responses` (`{"model": <configured model>, "input": "ping", "max_output_tokens": 1}`) instead. `GET /v1/models` cannot be used for this: real deployments answer it without the header, which the proxy attaches to LLM-call responses only. `POST /v1/chat/completions` is not used either: in LiteLLM 1.105.x its exception path omits the version argument, so a model-side 4xx (for example a thinking model rejecting a one-token budget) answers without the header and would block startup on a safe proxy; the Responses API exception path keeps it. The header is therefore read whatever the probe's HTTP status is; startup still fails if the header is missing, the probe cannot be reached, or the version is too old. The probe is a billed one-token call and is logged as a warning; operators who object to it can instead grant the key the health route, in which case no probe is sent.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
