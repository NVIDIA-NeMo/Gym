# Agent-X GoalAcc

605 non-generative tasks: 494 image tasks and 111 video tasks. [AgentX.json](../manifests/AgentX.json) pins the dataset revision, file hashes, selected tasks and exclusions. Video preparation samples approximately one frame per second, up to 128 frames.

## Task and score

Prepared rows contain the question and image inputs in
`responses_create_params.input`. Reference answers and category labels stay in
`verifier_metadata`; they are not model inputs. No external task tools or sandbox
are required by the verifier. The catalog configuration uses `simple_agent` for
image-question answering; a tool-using harness is a separate configuration choice.

Use the public Goal Accuracy rubric. Objective answers score 0 or 1; subjective answers retain fractional scores in [0, 1]. This entry does not measure image-generation quality or tool-selection accuracy. The caller chooses the judge; its identity must accompany the score.

Private reasoning is removed before grading. Empty final answers score zero.
Unresolved judge errors remain errors, not incorrect model answers. The reward
range is [0, 1], higher is better. Reports include benchmark-specific breakdowns;
record both the metric key and its scale when comparing results.

## Prepare and check

Run from the Gym repository root after obtaining the dataset:

```bash
gym eval prepare --benchmark opencode_multimodal_benchmarks/agentx \
  +prepare_script_args.source_root=/data/agentx
gym env validate opencode_multimodal_benchmarks/agentx
gym env test opencode_multimodal_benchmarks/agentx
gym env publish opencode_multimodal_benchmarks/agentx
```

Preparation writes `benchmarks/opencode_multimodal_benchmarks/data/agentx.jsonl`.
Collation wraps the prepared rows in typed `task_id`/`task_input` tasks.
The `visual_tasks` route selects `single_agent_turn`, which owns the agent session
and verification lifecycle; the source JSONL is unchanged.
The offline verifier fixtures use synthetic answers and scripted judge replies,
not a hosted judge or benchmark images. Publication checks are local validation,
not certification or an upload.

## Evaluate

Configure `policy_model` and, when judging is required, `judge_model` plus a
`judge_server` or `judge_endpoint`. Keep credentials in the runtime environment.
See [run configuration](../README.md#run-configuration) before running:

```bash
gym eval run --benchmark opencode_multimodal_benchmarks/agentx \
  --model-type openai_model --split benchmark --limit 5 --num-repeats 1 \
  --output results/agentx_smoke.jsonl
```

Remove the smoke limit for a complete score. Save the resolved configuration,
judge identity, sampling, budgets, timeouts, repeats and supported random seeds.
The prepared requests use temperature 0.6 and top-p 0.95; the catalog sets one
repeat. Agent/model limits are supplied by the run configuration.

The integration code is Apache-2.0; source dataset terms remain those of its
publisher. See [implementation provenance](../../../resources_servers/opencode_multimodal_benchmarks/third-party-notices.md)
for adapted evaluation code. No benchmark data is included in this entry.

## Harness pairings

| Harness | Configuration | Status |
| --- | --- | --- |
| `simple_agent` | This catalog entry, with a vision-capable model | Experimental; no task tools |
| `opencode_multimodal_agent` | [`opencode.yaml`](opencode.yaml) plus model and sandbox-provider settings | Experimental; OpenCode 1.17.11, WebSearch and WebFetch enabled, one activation |

Configure `webfetch` and `websearch` independently under the pairing's
`visual_agent.responses_api_agents.opencode_multimodal_agent.opencode_config.tools`.
Both tools are enabled in this pairing, with `network_access: inherit`.
Bash and Python retain the sandbox's existing network access as well. Disabling a
web tool removes its advertisement; it does not change sandbox egress. Record
both the network policy and tool settings with the score.

The OpenCode session path requires explicit sampling, embeds tool-produced images
before cleanup, and rejects continuation and training-capture requests. The
Environment Server owns activation, close and verification. See the
[agent README](../../../responses_api_agents/opencode_multimodal_agent/README.md)
for runtime requirements and configurable limits.

See the shared [grading, deadlines and recovery contract](../README.md#grading-and-recovery)
for observed judging, failure outcomes and resumable-run requirements.
