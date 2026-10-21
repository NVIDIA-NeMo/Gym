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


## Session-affinity routing

Endpoints behind a router that pins a session to one replica (to reuse its KV cache) need every
request of a rollout to carry the same session id. `rollout_id_headers` sends the rollout id in the
named headers on every upstream request made for a rollout:

```bash
++rollout_correlation=true \
++policy_model.responses_api_models.openai_model.rollout_id_headers='[X-Session-Id]'
```

The model server learns the rollout from the `/ng-rollout/<id>` prefix that agents add to their
model calls. `rollout_correlation=true` makes agents add it without turning on model-call capture;
`observability_enabled` and token capture add it too. Requests without a rollout get no header.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
