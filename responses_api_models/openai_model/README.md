# Description

OpenAI-compatible model server using Gym's shared HTTP client.

## HTTP retries

HTTP 404 and 408 are retryable by default, alongside the existing transient
server and rate-limit errors. This applies to every dataset and to policy and
judge clients. A permanent 404 still fails after the bounded attempt budget.
Retries resend the same request with the existing fixed 0.5-second delay
between attempts and preserve the terminal error body.

`max_http_attempts` defaults to three total attempts. Set it per model server:

```yaml
judge_model:
  responses_api_models:
    openai_model:
      max_http_attempts: 5
```

A value of one disables HTTP retries. Connection-error retries are controlled
separately. Internal Gym clients retain their existing unbounded extension for
rate-limit statuses; 404 and 408 never trigger that extension.

## Retrying empty completions

Some always-thinking reasoning models occasionally stop after emitting only
reasoning: the chat completion has `finish_reason: "stop"` but neither
`content` nor `tool_calls` (on `/v1/responses`, the output holds only
`reasoning` items). Agent harnesses usually treat that empty assistant turn as
fatal and end the episode. `retry_empty_completions: N` re-issues the same
upstream request up to N extra times when this happens; the first response with
visible output (or the last attempt) is returned. Responses cut off by the
token limit (`finish_reason: "length"`, `status: "incomplete"`) are returned as
is. The default, 0, disables the retry.

Each empty-completion retry uses the configured upstream retry policy and
releases its provider slot between attempts. With `upstream_max_num_tries: 1`,
the maximum provider attempts are `(N + 1) * upstream_retry_policy.max_attempts`.

```yaml
policy_model:
  responses_api_models:
    openai_model:
      retry_empty_completions: 2
```


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
