# HRBench8K

800 test rows, including the option-order cycles. The source TSV and image identities are pinned in [HRBench8K.json](../manifests/HRBench8K.json).

## Task and score

Prepared rows contain the question and image inputs in
`responses_create_params.input`. Reference answers and category labels stay in
`verifier_metadata`; they are not model inputs. No external task tools or sandbox
are required by the verifier. The catalog configuration uses `simple_agent` for
image-question answering; a tool-using harness is a separate configuration choice.

Mean accuracy across option-order cycles, with the public multiple-choice extractor and reference-blind judge fallback. Preserve cycles when aggregating.

Private reasoning is removed before grading. Empty final answers score zero.
Unresolved judge errors remain errors, not incorrect model answers. The reward
range is [0, 1], higher is better. Reports include benchmark-specific breakdowns;
record both the metric key and its scale when comparing results.

## Prepare and check

Run from the Gym repository root after obtaining the dataset:

```bash
gym eval prepare --benchmark opencode_multimodal_benchmarks/hrbench8k \
  +prepare_script_args.benchmark=hrbench8k \
  +prepare_script_args.tsv=/data/hrbench8k.tsv \
  +prepare_script_args.sha256=<source-sha256>
gym env validate opencode_multimodal_benchmarks/hrbench8k
gym env test opencode_multimodal_benchmarks/hrbench8k
gym env publish opencode_multimodal_benchmarks/hrbench8k
```

Preparation writes `benchmarks/opencode_multimodal_benchmarks/data/hrbench8k.jsonl`.
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
gym eval run --benchmark opencode_multimodal_benchmarks/hrbench8k \
  --model-type openai_model --split benchmark --limit 5 --num-repeats 1 \
  --output results/hrbench8k_smoke.jsonl
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
| `opencode_multimodal_agent` | This entry plus [`opencode_sessions.yaml`](../opencode_sessions.yaml) and a sandbox provider | Experimental; OpenCode 1.17.11, initial images and text, one activation |

The OpenCode session path uses configured sampling defaults or request overrides
and supports both evaluation and training capture. Shared media handling embeds
tool-produced images before cleanup. Each episode permits one activation and
rejects continuation. The Environment Server owns activation, close, verification
and optional training tool rewards; the evaluation profile disables those rewards. See the
[agent README](../../../responses_api_agents/opencode_multimodal_agent/README.md)
for runtime requirements and configurable limits.

See the shared [grading, deadlines and recovery contract](../README.md#grading-and-recovery)
for observed judging, failure outcomes and resumable-run requirements.
