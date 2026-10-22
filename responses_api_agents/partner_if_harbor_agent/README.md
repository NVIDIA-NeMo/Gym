# Partner IF Harbor agent

Runs a native Harbor task (OpenHands 0.62 on OpenSandbox) and sends what the run captured
to its `resources_server`'s `/verify` as a `verification_input`: the input the actor was
given, the task bundle's documents and the verifier's artifact receipt. `partner_if`,
with `accept_verification_input: true`, grades that with the response and returns the
score fields; `/aggregate_metrics` returns its metrics.

OpenHands calls the configured `model_server` from inside the sandbox at the URL set as
both `api_base` and `LLM_BASE_URL`, so set `use_absolute_ip: true`. With capture
correlation on (observability, token capture or checkpointing), that URL carries the
run's rollout prefix.
The agent fails at startup on a loopback model server and, with `ArtifactEnvironment`,
without `OPENSANDBOX_API_KEY` in Gym's environment or `harbor_worker_env`. The Harbor
worker reads the key from its own environment, with `harbor_worker_env` on top: Gym's
local Ray workers inherit the environment Gym was started with, and with an external Ray
cluster (`ray_head_node_address`) the key has to be in that cluster's environment or in
`harbor_worker_env`. A run that reaches OpenHands' iteration cap is graded; a missing or
unreadable artifact receipt leaves only the items that need the artifact unscorable. A
Harbor or OpenHands failure, or a row whose task name or task bundle fails the checks
before Harbor runs, goes to the failures file masked, with
`_ng_failure_class: harbor_failed` and `failure_kind: agent_run_error`, and a resume runs
it again. A failed `/verify` call, or a `/verify` reply that is not a valid response,
goes there masked with both set to `verifier_error`. The sandbox run had finished, so
that row is terminal: a resume does not run it again.

`configs/partner_if_harbor_agent.yaml` validates, but it runs only with an OpenSandbox
deployment and Harbor tasks with `live_agent` rows, which Gym does not ship.
