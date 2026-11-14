# SaaS Bench Environment Server

This Environment Server owns episode ordering, reward projection, failure policy and benchmark metrics for both SaaS harnesses. It accepts the existing flat prepared rows at `/run` and typed episode requests at the same endpoint. The same bounded BaseEnvironmentServer lifecycle seeds, activates and closes a native SaaS agent session in either case. The configured episode deadline cancels the activation, and close waits for the worker's process and app cleanup before confirming release. Closed session fences and saved close replies use Gym's shared session implementation.

The Environment Server and agent receive the same hash-verified prepared bundle path. Model sampling and token-capture routing follow Gym's episode settings. Tasks, native verifier checks and partial-score policy are unchanged. Masked infrastructure/judge failures remain distinct from genuine model endings.

Use `benchmarks/saas_bench_opencode/config.yaml` or `benchmarks/saas_bench_browser_use/config.yaml`; set the data, cache, run-root and model/judge runtime references documented in [the SaaS guide](../../responses_api_agents/saas_bench_agent/README.md). Re-prepare the bundle and use a fresh campaign after changing pins. Existing campaigns retain their frozen source.

The native worker still couples application provisioning, the external harness and native final-state verification in one trusted subprocess. Moving those native app operations to a reusable Resources Server and using the stock OpenCode agent remains separate work. Session compliance does not imply that larger migration is complete.
