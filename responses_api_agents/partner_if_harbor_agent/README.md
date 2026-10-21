# Partner IF Harbor agent

Runs a native Harbor task (OpenHands 0.62 on OpenSandbox) and sends what the run captured
to its `resources_server`'s `/verify` as a `verification_input`: the input the actor was
given, the task bundle's documents and the verifier's artifact receipt. `partner_if`
grades that with the response and returns the score fields; `/aggregate_metrics` returns
its metrics.

OpenHands calls the configured `model_server` from inside the sandbox at the URL set as
both `api_base` and `LLM_BASE_URL`, so set `use_absolute_ip: true`. With capture
correlation on (observability, token capture or checkpointing), that URL carries the
run's rollout prefix.
The agent fails at startup on a loopback model server and, with `ArtifactEnvironment`,
without `OPENSANDBOX_API_KEY`. A run that reaches OpenHands' iteration cap is graded; a
missing or unreadable artifact receipt leaves only the items that need the artifact
unscorable. A Harbor or OpenHands failure goes to the failures file with
`failure_kind: agent_run_error`.

`configs/partner_if_harbor_agent.yaml` validates, but it runs only with an OpenSandbox
deployment and Harbor tasks with `live_agent` rows, which Gym does not ship.
