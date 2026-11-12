# Native Legal Agent Bench agent

LAB's Responses tool loop runs in a sandbox owned by the Harbor Environment
Server. Select `benchmarks/legal_agent_bench/config_native.yaml`; `config.yaml`
selects the same LAB Native agent.

The agent borrows the sandbox connection. Closing its session stops tool
processes and disconnects; the Environment Server collects outputs, grades in
a separate sandbox, and cleans up both sandboxes. Identical activation retries
join the same execution. Continuations and required external tool grants are
unsupported.

The loop implements LAB's workspace tools and `finish`. Turn, context, and
agent-budget exhaustion are scoreable incomplete outcomes after confirmed
cleanup. Model, provider, and cleanup failures remain operational failures.
The budget includes connection and activation setup. Responses and tool
observations are retained; no training-grade observation support is claimed.

See the [LAB evaluation guide](../../fern/versions/latest/pages/evaluation/legal-agent-bench.mdx).
