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


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
