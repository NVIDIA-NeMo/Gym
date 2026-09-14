# TB4 resources server

This server owns pinned task resolution, environments, sidecars, official budgets,
verification, artifacts, and cleanup. Agent workers own the evaluated harness.
Runtime: Harbor 0.23.0 with Gym's existing Harbor sandbox environment adapter.

## Contract

1. `/seed_session` accepts `task_name`, `task_ref`, `dataset_ref`, and `rollout_id`.
   The task must match the configured manifest. Dataset rows cannot select local
   paths or override grading instructions. The response includes a session ID,
   main-sandbox descriptor, instruction, user, working directory, setup/agent
   budgets, MCP declarations, and a task skills directory.
2. The agent attaches with `owns_sandbox=False`, installs/configures its harness,
   and calls `/start_session`. Resources return the remaining official agent
   budget. Repeating this call does not restart the clock.
3. `/verify` accepts the session ID, Gym response/usage, termination reason, and
   artifact references. Resources stop tracked harness process groups, collect
   task-declared state using Harbor, run the official verifier, and destroy the
   collection. Official zero and nonzero rewards survive agent failure/timeout.
4. `/cancel_session` ends an abandoned setup or running episode. A running
   cancellation still permits official grading. Closing a borrowed client never
   destroys the resources-owned sandbox.

The states are preparing, ready, agent running, verifying, and closed. Session
cookies bind access to the originating resources session. The agent supplies its
stable `client_session_id` so retries before the initial cookie response also reuse
the same episode. Its `execution_id` stays fixed across resources HTTP retries;
another worker invocation for the same rollout is rejected before attachment.
Run one resources worker per artifact directory. Concurrent duplicate
seeds share one attempt; conflicting identities fail. Concurrent verification
requests share one lifecycle and the first accepted agent result. Completed
verification retries return the recorded response, including after restart.

The server retains the Compose creator and its relay/volume ownership for the
whole lifecycle. Shutdown cancels owner tasks, allowing Harbor cleanup. Client
HTTP disconnection does not cancel the resources task. Abandoned setup expires
at the setup deadline; abandoned execution expires at the official agent deadline.
Abrupt process death stops renewal; provider TTL is the cleanup fallback. Active
persisted episodes are rejected after restart and cannot be resumed safely. Use a
new rollout identity for a new attempt, not a stale descriptor.

`evaluation_completed` means an official reward was retrieved. A scored negative
has `reward=0` with no infrastructure error. A missing verifier result or an agent
infrastructure failure adds `infrastructure_error` and `_ng_failure_class`, even
when a reward was retrieved. Artifacts include the Harbor trial directory and
worker trajectory references. Failure diagnostics are written before teardown.

This server is not marked verified: deployment smoke evidence and category gaps
are recorded in the [benchmark migration notes](../../benchmarks/terminal_bench_4/migration.md).
