# Native Legal Agent Bench agent

Evaluation-level Responses tool loop for LAB. Use `benchmarks/legal_agent_bench/config_native.yaml` to pair it with the LAB resources server through `single_agent_turn_legacy`. Harbor remains the benchmark default.

The LAB resources server creates the task sandbox, prepares the task, collects outputs, and stops the sandbox. It also creates and cleans up a separate verifier sandbox for grading. This agent borrows a connection to the task sandbox. Closing the agent session stops its tool processes and disconnects; it leaves sandbox teardown to the resources server. It supports one activation per session; duplicate identical activation requests return the existing result. Required external tool grants and continuations are rejected. Operator rollout stops are unsupported; use the Environment Server episode deadline or session close to cancel the host loop.

The native loop implements LAB's six workspace tools and `finish`, including two missing-deliverable rejections before soft acceptance. Max-turn and overall agent-budget exhaustion remain incomplete and can be graded. The default agent budget is 10800 seconds; individual model calls and shell commands retain their own smaller limits. Model and sandbox failures propagate through Gym's failure handling. Responses and full tool observations are retained; normalized invocation evidence is currently an explicit observation gap, so no training-grade or TB21 support is claimed.

See the LAB evaluation guide in `fern/versions/latest/pages/evaluation/legal-agent-bench.mdx` for task preparation, judge configuration, provider requirements, and score comparability.
