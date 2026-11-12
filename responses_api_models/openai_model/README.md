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


## Session-affinity routing

Endpoints behind a router that pins a session to one replica (to reuse its KV cache) need every
request of a session to carry the same session id. `session_key_headers` sends the session key in the
named headers on every upstream request made for a rollout:

```bash
++rollout_correlation=true \
++policy_model.responses_api_models.openai_model.session_key_headers='[X-Session-Id]'
```

The model server learns the rollout from the `/ng-rollout/<id>` prefix that agents add to their
model calls. `rollout_correlation=true` makes agents add it without turning on model-call capture;
`observability_enabled` and token capture add it too. Requests without a rollout get no header.

A session is a rollout attempt: the header carries the capture key (`<rollout id>`, or `<rollout id>-a<N>` for
a retry), and `nemo_gym.core.current_session_key()` is the one function that builds it. Agents that make model calls
inside the same attempt, such as a policy and a user simulator, or a judge that calls through the same model server,
share one session and therefore one replica. The vLLM model server's endpoint selection uses the same key.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
