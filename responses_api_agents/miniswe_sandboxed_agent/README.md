# Sandboxed mini-SWE

Generic mini-SWE 2.1.0 `DefaultAgent` execution on an environment owned by a Gym
resources server. It uses the sandbox seed/start/verify contract. It has no SWE-bench image, patch, dataset, or grading assumptions.

The existing synchronous mini-SWE loop uses an explicit bridge to Gym's async
Responses model client and sandbox operations. Cancellation closes pending I/O
and joins the loop before requesting verification. The worker releases its
connection; resources retain destruction ownership and the authoritative budget.

Profile `tb4-miniswe-text-v1` uses one text-form bash action per model response.
The generic prompt defines completion as a successful command whose first output
line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, matching mini-SWE's convention.
`step_limit=0` and `cost_limit=0` leave those caps disabled; resources bound time.
Every step persists the native mini-SWE trajectory, including observations.

Task skills are exposed by their supplied directory. For MCP tasks, setup installs
`mcp==1.29.0` into a task-local virtual environment, discovers the declared tools,
and adds their schemas and invocation command to the prompt. The CLI supports
stdio, SSE, and streamable HTTP; calls execute inside the main sandbox so service
names retain their task-network meaning. A persistent MCP session preserves state
across calls. Image tool results become multimodal model inputs. This changes the evaluation profile
relative to native mini-SWE and must be disclosed in score comparisons.

Use [the TB4 mini-SWE profile](../../benchmarks/terminal_bench_4/miniswe.yaml) with
[TB4 resources](../../resources_servers/terminal_bench_4/README.md). Existing
`mini_swe_agent_2` SWE-bench behavior remains unchanged. Coverage is a validation
claim, not implied by selecting this configuration.
