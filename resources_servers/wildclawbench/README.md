# WildClawBench Resources Server

Owns one native task-container lease per Gym Resources session and runs the pinned original
WildClawBench grader only after confirmed agent close. Typed seed validates the task, source,
prompt and workspace hashes. Failed cleanup retains ownership for a later close; shutdown attempts
every remaining container. Verification retries reuse retained results and never rerun a judge
after the task container has been removed.

Paired with `wildclawbench_agent` and the `wildclawbench` Environment Server on one trusted host
and shared workspace. No external tool routes or borrowed SandboxAccess are exposed. The native
container is created lazily by the agent worker on this owner's behalf. This is an unverified
development integration; see [setup and validation](../../benchmarks/wildclawbench/README.md).
