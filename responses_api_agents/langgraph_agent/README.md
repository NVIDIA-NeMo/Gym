# LangGraph Agent

The adapter runs four in-process LangGraph workflows: iterative reflection,
subagent orchestration, parallel thinking, and ReWOO. The workflows use only the
Gym Model Server; they do not execute environment tools or use task sandboxes.
The framework dependencies are declared in `requirements.txt` (currently unpinned).
The session migration was tested with LangGraph 1.2.18 and langchain-core 1.7.3
on Python 3.13.14.

## Sessions and capabilities

**Capability level: Evaluation.** The adapter serves agent sessions for
`single_agent_turn` and flat-row `single_agent_turn_legacy` environments. Pairings
are provided for `reasoning_gym` and `math_with_judge` in `environments/langgraph_*`.
Each activation runs the complete graph. Identical retries join or replay that
activation; changed requests, wrong rollout identities, and closed sessions return
409. Use the seeded episode's `/ng-rollout/<capture_key>/v1/responses` route.
Close cancels and drains active graph execution before verification.

Required HTTP/MCP grants and any sandbox access return a final 422. Optional grants
are ignored. Set `resources_tool_transports: []` and `num_workers: 1` in the pairing.
An unbound agent config has `resources_server: null`; the Environment Server owns
resource setup, verification and cleanup. Select a workflow with
`--agent langgraph_agent/<flavor>_agent`.

All graph model calls preserve evaluation and training-capture URL prefixes.
Agent/resource cookies are never forwarded to the Model Server. Response usage is
the sum of every graph model call only when all calls report usage; otherwise it is
unknown. Close always returns an observation bundle with an explicit gap: this
adapter does not record native invocation or turn evidence. Rollout health can be
`unobserved`, and the strict evidence inspector is not expected to pass. Shell-tool
harness conformance probes do not apply because these graphs have no shell tools.

Checkpoint recovery remains `restart_only`. Non-monotonic trajectories, including
parallel reasoning and synthesis, are not directly compatible with NeMo RL's
monotonically increasing trajectory assumption. Preserving the capture route does
not establish training qualification. Sampling fields are forwarded to each model
call; `max_output_tokens` is a per-call limit, not a whole-graph budget. The graph
uses its own prompts and does not execute model-requested function calls.

## Quick start

```bash
gym env start \
    --resources-server reasoning_gym/reflection_agent \
    --model-type vllm_model

gym eval run --no-serve \
    --agent reasoning_gym_langgraph_agent \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output example_rollouts.jsonl \
    --limit 1
```

Configure the model endpoint as described in the model server documentation. The
existing twelve pairing configs now use `single_agent_turn_legacy` while retaining
their dataset formats and agent names. Legacy callers can still post to the
agent's `/run` when `resources_server` is configured; it preserves resource cookies
and verifier metadata. Sessionless `/v1/responses` also remains available.
