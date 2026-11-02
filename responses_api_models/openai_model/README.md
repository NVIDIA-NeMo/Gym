# Description

OpenAI-compatible model server using Gym's shared HTTP client.

## HTTP retries

HTTP 404 and 408 are retryable by default, alongside the existing transient
server and rate-limit errors. This applies to every dataset and to policy and
judge clients. A permanent 404 still fails after the bounded attempt budget.
Retries resend the same request and preserve the terminal error body.
Connection retries wait 0.5 seconds between attempts.

`max_http_attempts` defaults to three total attempts. Set it per model server:

```yaml
judge_model:
  responses_api_models:
    openai_model:
      max_http_attempts: 5
```

A value of one disables retries. Connection-error retries use the same
`max_http_attempts` budget. Values of one or two also lower request-timeout
attempts from three to one or two; the default remains three. Internal Gym
clients retain their existing unbounded extension for rate-limit statuses;
404 and 408 never trigger that extension.

## Streaming Responses passthrough

`responses_stream_passthrough: true` relays streaming `/v1/responses` requests to
`{openai_base_url}/responses` as the client sent them, except that `model` becomes
`openai_model` (`extra_body` fills only absent fields), and streams the upstream SSE back
unchanged. Use it for blackbox harnesses whose own requests the typed path cannot
carry, for example the Codex CLI running a model from its bundled catalog: code-mode
tools in `additional_tools` items, or a reasoning effort newer than the pinned OpenAI
SDK. The upstream must serve the OpenAI Responses API. Non-streaming requests keep the
typed path.

The relay is for evaluation: it skips Gym's request sanitizer and rejects token-ID
capture. Model-call capture still records each exchange, so check the effective request
fields (for example `reasoning.effort`) in the captured calls. A multi-attempt
`upstream_retry_policy` and `drop_input_reasoning_items` are rejected with it; the
client's HTTP retries before the stream starts and `propagate_upstream_http_status_codes`
still apply.

A relayed stream lasts as long as the model's answer, which can exceed the default
30-minute `upstream_request_timeout_seconds`. For relayed streams that setting therefore
bounds the wait for the response headers and for each read of the body (a stalled stream),
not the whole stream. Set it to `null` to leave every time limit to the client, as when the
harness calls the endpoint directly; long agent runs, such as 8-hour Codex sessions, do this.


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
