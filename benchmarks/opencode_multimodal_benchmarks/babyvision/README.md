# BabyVision

The caller-supplied BabyVision test TSV, with choices already included in each question. Supply its SHA256 and record its release and row count; this entry does not enforce a canonical dataset size. The scoring prompt follows [BabyVision](https://github.com/UniPat-AI/BabyVision/tree/7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a).

## Task and score

Prepared rows contain the question and image inputs in
`responses_create_params.input`. Reference answers and category labels stay in
`verifier_metadata`; they are not model inputs. No external task tools or sandbox
are required by the verifier. The catalog configuration uses `simple_agent` for
image-question answering; a tool-using harness is a separate configuration choice.

Judge the boxed answer, or the full visible response when no box exists. This fallback is an adapter choice.

Private reasoning is removed before grading. Empty final answers score zero.
Unresolved judge errors remain errors, not incorrect model answers. The reward
range is [0, 1], higher is better. Reports include benchmark-specific breakdowns;
record both the metric key and its scale when comparing results.

## Prepare and check

Run from the Gym repository root after obtaining the dataset:

```bash
gym eval prepare --benchmark opencode_multimodal_benchmarks/babyvision \
  +prepare_script_args.benchmark=babyvision \
  +prepare_script_args.tsv=/data/babyvision.tsv \
  +prepare_script_args.sha256=<source-sha256>
gym env validate opencode_multimodal_benchmarks/babyvision
gym env test opencode_multimodal_benchmarks/babyvision
gym env publish opencode_multimodal_benchmarks/babyvision
```

Preparation writes `benchmarks/opencode_multimodal_benchmarks/data/babyvision.jsonl`.
The catalog uses already-prepared flat rows through `single_agent_turn_legacy`;
that Environment Server owns the agent session and verification lifecycle.
The offline verifier fixtures use synthetic answers and scripted judge replies,
not a hosted judge or benchmark images. Publication checks are local validation,
not certification or an upload.

## Evaluate

Configure `policy_model` and, when judging is required, `judge_model` plus a
`judge_server` or `judge_endpoint`. Keep credentials in the runtime environment.
See [run configuration](../README.md#run-configuration) before running:

```bash
gym eval run --benchmark opencode_multimodal_benchmarks/babyvision \
  --model-type openai_model --split benchmark --limit 5 --num-repeats 1 \
  --output results/babyvision_smoke.jsonl
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

The OpenCode session path requires explicit sampling, embeds tool-produced images
before cleanup, and rejects continuation and training-capture requests. The
Environment Server owns activation, close and verification. See the
[agent README](../../../../responses_api_agents/opencode_multimodal_agent/README.md)
for runtime requirements and configurable limits.
