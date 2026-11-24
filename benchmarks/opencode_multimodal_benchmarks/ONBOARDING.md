# OpenCode benchmark onboarding

All recipes use the shared OpenCode session path and owned OpenSandbox. See the
[catalog guide](README.md) and [agent setup](../../responses_api_agents/opencode_agent/README.md).
Do not change runtime revisions while resuming historical runs.

## Supported benchmarks

| Configuration | Data | Scoring |
|---|---|---|
| `vstar/opencode.yaml` | VStar TSV and images | Multiple-choice accuracy |
| `hrbench8k/opencode.yaml` | HRBench8K TSV and images | Mean accuracy across cycles |
| `vtcbench/opencode.yaml` | VTCBench TSV and images | Rule matching with LLM fallback |
| `zerobench/opencode.yaml` | Official main-split Parquet with embedded images | Case-insensitive exact answer accuracy |
| `agentx/opencode.yaml` | Agent-X source data or prepared bundle | Independent reference grading on 605 non-generative tasks |

## Prepare data

Datasets are downloaded separately. Preparation writes task JSONL and image assets
under `benchmarks/opencode_multimodal_benchmarks/data/`. Reference answers are
stored in `verifier_metadata`, separately from the agent's inputs.

### TSV datasets

Supply these preparation arguments through Gym's `prepare_kwargs`:

- `benchmark`: `vstar`, `hrbench8k`, or `vtcbench`.
- `tsv`: path to the dataset file.
- `sha256`: SHA256 of that file.
- `image_root`: directory for externally referenced images.
- `limit`: optional number of tasks for a smoke run.

TSV files can contain embedded images or image paths. Use relative image paths
for portable datasets. VStar and HRBench8K inputs are checked against the manifests in
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

### ZeroBench

Accept the access conditions on the [official dataset page](https://huggingface.co/datasets/jonathan-roberts1/zerobench).
Download `data/zerobench-00000-of-00001.parquet` at revision
`8fcdfd9138c802cac52b774b36946e70f738f500` and supply its path as
`prepare_kwargs.parquet`. Preparation validates the file checksum and all 100
question IDs, preserves image bytes, and keeps answers outside policy input.
Do not check the gated dataset or its answers into source control.

Compose `zerobench/opencode.yaml` with `opencode_eval_agent.yaml` for evaluation without a
judge. The public question suffix requests a final answer in curly braces. The
scorer follows the [public example](https://github.com/jonathan-roberts1/zerobench#code),
including last-brace extraction and whitespace behavior. The default is five repeats;
`ZeroBench/pass@5` is the primary metric. Mean single-attempt accuracy is labeled
`ZeroBench/pass@1`. Incomplete attempts cannot certify a complete pass@5 score.

## Configure a run

Compose the selected benchmark YAML with [`opencode_eval.yaml`](opencode_eval.yaml)
and your policy model configuration. See the
[agent setup guide](../../responses_api_agents/opencode_agent/README.md)
for the task image, pinned offline runtime assets, and provider configuration.

The example profile uses OpenCode 1.17.11, a 131072-token context/output ceiling,
and the stock 65-step limit. It retains reasoning history, disables
compaction, and uses OpenCode’s native final-answer note at the step limit. These
settings can be overridden. Match the serving model's context and image capacity
to your run, and configure sampling and concurrency explicitly.

### Policy adapter selection

The common OpenCode recipes do not select a policy implementation. Supply
`--model-type openai_model` for a hosted adapter, or `--model-type vllm_model`
for a vLLM-compatible adapter. With `vllm_model`, also include
`benchmarks/opencode_multimodal_benchmarks/opencode_eval_vllm.yaml` in
`config_paths`. This evaluation-only overlay preserves reasoning history,
uses interleaved reasoning, disables token-ID capture, clears per-call output
token overrides, and propagates context-overflow errors. Do not include it
with `openai_model`.

Training profiles continue to select
`responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`,
which enables `return_token_id_information`; they must not import the
evaluation-only vLLM overlay.


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
- **Agent-X:** use the independent reference-answer judge. Objective answers receive 0 or 1;
  subjective answers receive a score from 0 to 1. Image-generation tasks are excluded.

Unresolved judge failures need rejudging before reporting a complete benchmark
score. For agent sessions, the Environment Server's `agent_outcome_policy`
determines whether a failed attempt is verified, retried or left ungraded.
Configure that typed outcome policy explicitly when changing model-failure scoring.
Provider and runtime failures must retain the retry/failure-sidecar path.

Source implementations are linked beside the scoring code in the
[resources server](../../resources_servers/opencode_multimodal_benchmarks/).

## Validate a run

Start with a small `limit`. Inspect the resolved configuration, input images,
tool results, final answers, and verifier output. Check model-call evidence and
`quality_summary.json`, then run the complete dataset with the same settings.

The shared OpenCode conformance suite covers the common harness. Run the visual
adapter's unit tests and qualify real OpenSandbox rollouts separately, including
image fidelity, network enforcement, cleanup, and training capture.
