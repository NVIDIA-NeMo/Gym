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


## Provider reasoning content

Responses-compatible providers sometimes use `output_text` for a reasoning
item's content. Before validating provider responses, this server changes that
part's discriminator to `reasoning_text` only when it contains a string `text`
and no other fields except optional empty `annotations: []`. Text, encrypted
content, item order and other response fields are preserved without modifying
the provider's response object. Unknown or malformed shapes still go through
native validation unchanged. Requests and continuation replay are unchanged.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
