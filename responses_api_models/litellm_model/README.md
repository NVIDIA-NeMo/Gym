# Description

LiteLLM model server for use with LiteLLM proxy endpoints that expose the OpenAI Responses API (`/v1/responses`).

LiteLLM proxies may return non-standard response formats (e.g. `object="chat.completion"` instead of `"response"`, or `reasoning.effort = "none"` as a string instead of `null`). This server normalizes those responses so downstream NeMo Gym validation succeeds.

Extends `openai_model`'s `SimpleModelServer`.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
