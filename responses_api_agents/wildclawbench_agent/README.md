# WildClawBench agent

Evaluation adapter for original OpenClaw at the source revision in
[`source-pin.json`](../../benchmarks/wildclawbench/source-pin.json), using the provisioned
`wildclawbench-ubuntu:v1.3` image. Supports Gym agent-session seed, one activation and close.
The paired `wildclawbench` Resources Server owns the native container, grading and suite aggregation.
The shipped `single_agent_turn_legacy` Environment Server orders the episode and adds performance metrics.

Identical seed/activation retries reuse their state; changed bindings conflict. Close cancels
active work and returns captured observations or an explicit observation gap. Borrowed sandbox
access, required external tool grants, modified prompts and per-row sampling are rejected.
This native bridge uses trusted-host Docker or Enroot, inherited networking and one worker.
It does not capture token IDs or support activation continuation. Its supported benchmark pairing
is WildClawBench; it is not a general-purpose OpenClaw adapter.

See the [benchmark setup, protocol, limitations and validation](../../benchmarks/wildclawbench/README.md).
