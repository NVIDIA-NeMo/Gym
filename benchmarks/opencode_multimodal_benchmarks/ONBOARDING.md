# Local compatibility runs with OpenCode

For new evaluations, use the provider-backed `<benchmark>/opencode.yaml` recipes
in the [catalog guide](README.md). This page documents the existing local
Bubblewrap relay path for historical runs and training. Bubblewrap and Enroot
use private PID/procfs and network namespaces, with the validating proxy
protecting Gym control services. Owned OpenSandbox sessions use the provider's
process isolation and egress policy; Gym control services must be separately
inaccessible from task sandboxes. Deployment qualification is still required; see
the [network policy](../../responses_api_agents/opencode_multimodal_agent/network-policy.md).
Do not change an existing run's runtime while resuming it.

Evaluate visual tasks with `opencode_multimodal_agent`. The benchmark configurations
prepare Gym task rows, run the agent, and score its answers through the
`opencode_multimodal_benchmarks` resources server.

## Supported benchmarks

| Configuration | Data | Scoring |
|---|---|---|
| `vstar/opencode_relay.yaml` | VStar TSV and images | Multiple-choice accuracy |
| `hrbench8k/opencode_relay.yaml` | HRBench8K TSV and images | Mean accuracy across cycles |
| `vtcbench/opencode_relay.yaml` | VTCBench TSV and images | Rule matching with LLM fallback |
| `zerobench/opencode_relay.yaml` | Official main-split Parquet with embedded images | Case-insensitive exact answer accuracy |

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

### ZeroBench

Accept the access conditions on the [official dataset page](https://huggingface.co/datasets/jonathan-roberts1/zerobench).
Download `data/zerobench-00000-of-00001.parquet` at revision
`8fcdfd9138c802cac52b774b36946e70f738f500` and supply its path as
`prepare_kwargs.parquet`. Preparation validates the file checksum and all 100
question IDs, preserves image bytes, and keeps answers outside policy input.
Do not check the gated dataset or its answers into source control.

Compose `zerobench/opencode_relay.yaml` with `opencode_eval_agent.yaml` for evaluation without a
judge. The public question suffix requests a final answer in curly braces. The
scorer follows the [public example](https://github.com/jonathan-roberts1/zerobench#code),
including last-brace extraction and whitespace behavior. The default is five repeats;
`ZeroBench/pass@5` is the primary metric. Mean single-attempt accuracy is labeled
`ZeroBench/pass@1`. Incomplete attempts cannot certify a complete pass@5 score.

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

Unresolved judge failures need rejudging before reporting a complete benchmark
score. For agent sessions, the Environment Server's `agent_outcome_policy`
determines whether a failed attempt is verified, retried or left ungraded.
The OpenCode compatibility evaluation `/run` path scores established model-caused
failures as zero. Provider 429/5xx, transport, runtime setup and unclassified process
failures retain the retry/failure-sidecar path rather than settling tasks as wrong.

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
