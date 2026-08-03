# Legal Agent Bench Native Agent

This agent is the direct Gym-native implementation of Harvey LAB's model/tool
loop. It runs only as the inner agent of the hardened
[`legal_agent_bench_agent`](../legal_agent_bench_agent/README.md) sandbox
runner.

For each turn it sends the accumulated Responses API trajectory and LAB's
canonical function-tool definitions to the configured Gym policy model. It
executes returned `bash`, `read`, `write`, `write_docx`, `edit`, `glob`, and
`grep` calls inside the task sandbox, appends their results, and continues until
the model returns a final assistant message or reaches the configured turn
limit.

The outer runner owns task resolution, prompt and skill composition, sandbox
staging, result collection, verifier isolation, and artifact publication. A
model error, timeout, repeated empty response, failed runtime preflight, or
turn-limit exhaustion returns an explicit failed response while preserving any
partial trajectory. The outer runner masks that sample and skips verification.

Use `--benchmark legal_agent_bench` for this default. See the
[benchmark README](../../benchmarks/legal_agent_bench/README.md) for setup,
smoke-test, and artifact-inspection commands.
