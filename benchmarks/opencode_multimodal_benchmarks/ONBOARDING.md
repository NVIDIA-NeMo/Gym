# Visual benchmarks with OpenCode

Evaluate visual tasks with `opencode_multimodal_agent`. The benchmark configurations
prepare Gym task rows, run the agent, and score its answers through the
`opencode_multimodal_benchmarks` resources server.

## Supported benchmarks

| Configuration | Data | Scoring |
|---|---|---|
| `vstar.yaml` | VStar TSV and images | Multiple-choice accuracy |
| `hrbench8k.yaml` | HRBench8K TSV and images | Mean accuracy across cycles |
| `vtcbench.yaml` | VTCBench TSV and images | Rule matching with LLM fallback |
| `agentx.yaml` | Agent-X source data or prepared bundle | GoalAcc on 605 non-generative tasks |
| `babyvision.yaml` | BabyVision TSV and images | LLM answer verification |

## Prepare data

Datasets are downloaded separately. Preparation writes task JSONL and image assets
under `benchmarks/opencode_multimodal_benchmarks/data/`. Reference answers are
stored in `verifier_metadata`, separately from the agent's inputs.

### TSV datasets

Supply these preparation arguments through Gym's `prepare_kwargs`:

- `benchmark`: `vstar`, `hrbench8k`, `vtcbench`, or `babyvision`.
- `tsv`: path to the dataset file.
- `sha256`: SHA256 of that file.
- `image_root`: directory for externally referenced images.
- `limit`: optional number of tasks for a smoke run.

TSV files can contain embedded images or image paths. Use relative image paths
for portable datasets. BabyVision questions must already include their answer
choices. VStar and HRBench8K inputs are checked against the manifests in
[`manifests/`](manifests/) before applying `limit`; keep the original image bytes.

### Agent-X

Download the release identified in [`manifests/AgentX.json`](manifests/AgentX.json),
including `data.json`, `dataset.json`, and `files/`. Set `prepare_kwargs.source_root`
to its directory. Preparation validates the source files and produces 494 image
tasks and 111 video tasks. Videos are sampled at approximately one frame per second,
up to 128 frames. The manifest records the selected tasks and exclusions.

To package prepared tasks for another machine:

```bash
python -m benchmarks.opencode_multimodal_benchmarks.bundle_agentx \
  benchmarks/opencode_multimodal_benchmarks/data/agentx.jsonl \
  /data/agentx/bundle --sha256 <prepared-jsonl-sha256>
```

Use a new destination directory. The bundle contains `agentx.jsonl`, `media/`, and
`provenance.json`. To load it, supply `prepare_kwargs.dataset`, `sha256`, and
`media_root` instead of `source_root`.

## Configure a run

Compose the selected benchmark YAML with [`opencode_eval.yaml`](opencode_eval.yaml)
and your policy model configuration. See the
[agent setup guide](../../responses_api_agents/opencode_multimodal_agent/README.md)
for installing OpenCode, Bubblewrap, the Python image libraries, and plugin dependencies.
Update the runtime paths in the evaluation profile for your installation.

The example profile uses OpenCode 1.17.11, a 131072-token context/output ceiling,
and 55 regular turns plus 10 grace turns. It retains reasoning history, disables
compaction, and allows at most two tool calls during the grace period. These
settings can be overridden. Match the serving model's context and image capacity
to your run, and configure sampling and concurrency explicitly.

### Configure the judge

The evaluation profile routes judging through the `visual_judge` Gym model server.
Provide `GYM_JUDGE_BASE_URL` and `GYM_JUDGE_API_KEY` in the runtime environment and
set the judge model in your configuration:

```yaml
opencode_multimodal_benchmarks:
  resources_servers:
    opencode_multimodal_benchmarks:
      judge_model: <provider-model-id>
```

Judge calls are captured with the rollout. For direct HTTP judging, unset
`judge_server` and configure `judge_endpoint` with the chat-completions URL and
`judge_api_key_env` with the credential environment-variable name.

## Scoring

Scoring uses the final response after removing reasoning blocks. Empty answers
score zero.

- **VStar and HRBench8K:** extract the selected option; use a judge when extraction
  is ambiguous. HRBench8K averages the per-cycle accuracies.
- **VTCBench:** apply the rule matcher, then judge answers that did not match.
  Set `vtc_llm_fallback: false` for rule-only scoring.
- **Agent-X:** use the GoalAcc rubric. Objective answers receive 0 or 1;
  subjective answers receive a score from 0 to 1. Image-generation tasks are excluded.
- **BabyVision:** judge the boxed answer, or the full final response when no box
  is present.

Unresolved judge failures need rejudging before reporting a complete benchmark
score. Recognized agent crashes are recorded with zero reward and failure metadata.

Source implementations are linked beside the scoring code in the
[resources server](../../resources_servers/opencode_multimodal_benchmarks/).

## Validate a run

Start with a small `limit`. Inspect the resolved configuration, input images,
tool results, final answers, and verifier output. Check model-call evidence and
`quality_summary.json`, then run the complete dataset with the same settings.

To check the harness itself, provision its runtime using the
[conformance setup](../../scripts/harness_conformance/README.md#run-diagnostic-probes)
and run:

```bash
python scripts/run_harness_conformance.py --harness opencode_multimodal \
  --output /tmp/visual-conformance
```

Use a fresh output directory and inspect `conformance_report.md` for the results.
