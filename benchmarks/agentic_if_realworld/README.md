# Real-world Agentic Instruction Following

This is a private, inspected, failure-conditioned evaluation benchmark: 969 tasks
and 2,803 frozen atomic checks. The original view has 912 tasks / 1,805 checks;
the supplemental view has 57 tasks / 998 checks. Subtypes overlap.
All tasks remain development-exposed and evaluation-only. Original split names
do not authorize training, prompt examples, threshold selection or rubric tuning.

## Private data

Gym contains runtime code, the portable validator, source hashes and independently
constructed synthetic fixtures. Real inputs, per-item criteria, provenance,
historical answers, original rubric and judge prompt remain in an authenticated
private GitLab Model Registry. None of those contents may be committed or uploaded
publicly. Registry preparation verifies the original artifact hashes; it has no
public dataset fallback and requires no separate benchmark-repository checkout.

Obtain access and the approved tracking URI through your organization's private
access process. Supply it through `MLFLOW_TRACKING_URI`; there is no public
endpoint default. Before making a registry request, the loader checks the URI
against its fixed SHA-256 allowlist pin. Supply
`MLFLOW_TRACKING_TOKEN` securely in the process environment. For questions about
the public implementation, use [Gym maintainer issues](https://github.com/NVIDIA-NeMo/Gym/issues/new/choose).

Install Gym with `uv sync --extra dev --extra mlflow`, then install the frozen
validator dependency with
`uv pip install -r resources_servers/agentic_if_realworld/benchmark/policy/requirements.txt`.
Set `AGENTIC_IF_DATA_DIR` to an existing private release/cache, or use the default
`~/.cache/nemo-gym/agentic_if_realworld/1.0.0`. Preparation checks every original
artifact hash and makes the entire ignored `data/` directory a symlink to the
private prepared-file directory. Gym's adjacent collation and metrics files
therefore stay outside the checkout as well. Set `AGENTIC_IF_PREPARED_PATH` for
another private output, including a custom basename; an external `prepared.jsonl`
alias preserves the configured dataset path. Collation sidecar names are reserved.
Existing real directories, different directory targets and conflicting private
`prepared.jsonl` files are rejected before writing prepared data.

For an older file-only symlink layout, stop writers and inspect the ignored
`benchmarks/agentic_if_realworld/data/` directory. Preserve its generated files
outside Git, unlink only its old `prepared.jsonl` symlink, and remove the empty
directory with `rmdir` before preparing again. To switch a current directory
symlink, inspect and unlink that directory symlink itself; do not remove files
through it. Preparation never deletes or migrates existing artifacts.

Registry credentials, prepared inputs, model-call captures, rollouts, judge
receipts and analysis all remain private. Keep capture/output directories outside
Git; they contain the input traces. Leave `upload_rollouts` disabled. Published
code or review evidence may contain synthetic cases and aggregate statistics only.

## Measurement contract

The policy receives the exact frozen single-user-message wrapper containing the
complete captured prefix and embedded tool definitions. There are 817 nonempty
and 152 empty registries. Generate one assistant turn without native tool
execution. Do not truncate history, change roles, regenerate criteria, or send
historical answers or criteria to the policy model.

The Resources Server preserves strict-v2.2 validation and frozen criteria.
Real runs use the original hash-verified private judge prompt. The public
synthetic prompt is separately identified and is only for plumbing checks;
it does not replace the benchmark's prompt or establish judge quality. Original
source hashes remain distinct from public packaging hashes.
All checks passing yields reward 1; an established violation yields reward 0.
Exhausted delivery attempts or unresolved delivered reviews are unscored with
`mask_sample: true`, a failure kind and a reason. The numeric placeholder is
excluded from quality metrics. Set an approved judge model and endpoint through
`AGENTIC_IF_JUDGE_MODEL` and `AGENTIC_IF_JUDGE_ENDPOINT`; credentials come from
`AGENTIC_IF_JUDGE_API_KEY`. The reasoning setting is medium. A model or settings
change defines a separate run.

The default `adaptive_first_complete_v1` protocol uses temperature 0.7, streaming,
and these (seed, maximum-output-token) attempts:
`(17,32768), (17,65536), (17,131072), (29,131072), (43,131072)`.
Advance only after incomplete delivery. Keep the first complete answer regardless
of its grade. Judge recovery never regenerates that answer. Two HTTP 2xx judge
deliveries are allowed across resumes, with bounded transport retries and the
existing timeouts. Malformed reviews consume that budget; failed connections and
non-2xx judge responses do not. Those infrastructure failures stop the run so the
operator can fix the service or credentials and explicitly resume the saved
answer. After two delivered reviews, unresolved scoring stays unscored. Judge
endpoint, model and credential presence are checked before policy generation.

For a vLLM-backed model requiring maximum reasoning effort, set
`policy_model.responses_api_models.agentic_if_text.reasoning_effort=max` in the
run configuration. The adapter sends `chat_template_kwargs.reasoning_effort: max`,
using the same wire field as Gym's vLLM adapter. This selects the distinct
`adaptive_first_complete_max_effort_v1` profile. The default `null` omits that
field and retains the original request. Temperature 0.7, the delivery schedule,
wrapper, criteria and judge settings stay unchanged. Use a new run directory;
resuming across these effort settings is rejected. This parameter requires an
endpoint that honors vLLM chat-template kwargs; an endpoint accepting the JSON
alone does not establish that it applied maximum effort.

Every run fingerprints model settings, source bytes and dependencies. Resume
requires matching settings and rejects ambiguous in-flight receipts. A rejected
resume preserves the saved output and reports. Use a new run directory for changed
settings. Historical grades and recovery waves
remain historical; this adaptive protocol does not reproduce them.
The runtime identity records the running Gym revision and hashes its agent/core
source separately from the historical source pin. Older journals are rejected
after this recovery fix; start a new run directory rather than migrating them.
Each delivery attempt has its own Gym capture identity. The main rollout retains
the selected attempt's actual request settings and trajectory; all earlier
attempts remain in the private journal and capture sidecars.
Completed resume also revalidates the saved policy and judge HTTP bytes and the
derived grade. It cannot generate a missing cached judgement. Both execution
profiles reject judge receipt directories inside Git.

The adaptive driver saves the actual resolved configuration beside the output
as `resolved-config.yaml`, with secrets masked and private permissions. Resume
preserves that original snapshot, including when changed settings are rejected.
Keep it with `.run.json`: Gym's generic redactor also masks token-named budget
fields, whose fixed schedule remains in the run fingerprint and this contract.

Before dispatch, the driver atomically saves the selected logical task inventory
as `rollouts_materialized_inputs.jsonl` beside the output, with private
permissions. Adaptive delivery attempts do not add inventory rows. Health reports
use this inventory for expected and missing counts; the benchmark summary still
uses the full declared dataset denominator. Resume requires the identical saved
inventory and never replaces it. Older runs without one require a new run directory.

`rollouts.summary.json` reports both views, subtype overlap, expected tasks,
pass/fail/unscored counts and coverage. `rollouts_aggregate_metrics.json` exposes
`composite/complete_accuracy`, which stays null until every declared task has a
valid grade. The driver calls the Resources Server's standard aggregation
endpoint and saves Gym's task/repeat metrics, baseline statistics and environment
identity alongside the benchmark metrics. Native collection and reverification
use that same endpoint, including masked and unattempted tasks. Never substitute
resolved-only mean reward for complete accuracy.

## Run

Replace these placeholders with approved values supplied through private channels:

```bash
export MLFLOW_TRACKING_URI=https://your-approved-registry.example/api/v4/projects/PROJECT_ID/ml/mlflow
export AGENTIC_IF_POLICY_ENDPOINT=https://your-approved-provider/v1/chat/completions
export AGENTIC_IF_POLICY_MODEL=your-policy-model
export AGENTIC_IF_JUDGE_ENDPOINT=https://your-approved-judge.example/v1/chat/completions
export AGENTIC_IF_JUDGE_MODEL=your-approved-judge-model
# Supply AGENTIC_IF_POLICY_API_KEY, AGENTIC_IF_JUDGE_API_KEY and registry token securely.
export AGENTIC_IF_RECEIPT_DIR=/private/agentic-if/run-01/receipts

gym eval prepare --benchmark agentic_if_realworld
gym env validate agentic_if_realworld
gym env test agentic_if_realworld
gym eval run --benchmark agentic_if_realworld --split benchmark --num-repeats 1 \
  --concurrency 1 --output /private/agentic-if/run-01/rollouts.jsonl \
  ++observability_enabled=true ++model_call_capture_dir=/private/agentic-if/run-01/captures
```

Repeat the final command with `--resume` and the same paths to resume generation
and judging independently. Preserve saved response hashes and receipts.
`--concurrency N` runs up to N tasks at once, each with its own journal. A policy
transport error (429, 5xx, timeout) moves a task to its next attempt, as an
incomplete delivery does, only when a response establishes the delivery outcome.
A connection refusal or connect timeout stops the run; explicit resume retries
the same seed and token cap while retaining the failed receipt. A complete saved
SSE answer remains selected even if the connection fails after its final event.
Uncertain dispatch or a run stopped mid-generation requires receipt reconciliation
before resume. Keep N within the policy server's capacity.

### Private YAML configuration

Model endpoints, model names and the receipt directory are ordinary Gym/Hydra
configuration fields. Environment variables above are optional defaults for those
fields. The adaptive profile reads `AGENTIC_IF_POLICY_ENDPOINT` and
`AGENTIC_IF_POLICY_MODEL` directly; unrelated quickstart `policy_base_url` and
`policy_model_name` values do not override them. To keep a reusable run configuration, save this YAML outside Git, for
example as `/private/agentic-if/run-01/config.yaml`:

```yaml
policy_model:
  responses_api_models:
    agentic_if_text:
      endpoint: https://your-approved-provider/v1/chat/completions
      model: your-policy-model
agentic_if_realworld_resources:
  resources_servers:
    agentic_if_realworld:
      judge_endpoint: https://your-approved-judge.example/v1/chat/completions
      judge_model: your-approved-judge-model
      receipt_dir: /private/agentic-if/run-01/receipts
```

Compose it after the benchmark configuration:

```bash
gym eval run --config benchmarks/agentic_if_realworld/config.yaml \
  --config /private/agentic-if/run-01/config.yaml --split benchmark \
  --num-repeats 1 --concurrency 1 --output /private/agentic-if/run-01/rollouts.jsonl \
  ++observability_enabled=true ++model_call_capture_dir=/private/agentic-if/run-01/captures
```

Keep credentials in the process environment. Preparation still reads
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_TOKEN`, `AGENTIC_IF_DATA_DIR`,
`AGENTIC_IF_PREPARED_PATH` and `AGENTIC_IF_SYNTHETIC` from the environment;
the run override file does not configure the preparation subprocess. Retain the
same data paths for preparation and execution. Resume with the same YAML and
paths plus `--resume`; changing settings requires a new run directory.

For a platform deployment, pin the approved merged Gym commit and keep model-call
capture on. Resolve credentials securely on the compute node. Keep exact platform
bindings and access instructions in private configuration. Agree the reference
score, tolerance, runtime and coverage/health gates with the platform owner;
synthetic tests are not certification evidence.

The `agentic_if_realworld/native` profile retains the stock Gym model server,
agent, collector and response/reward interface. It is a distinct run protocol,
without the adaptive recovery schedule. No trainer or training-data authorization
is supplied. Native saved responses can use
`resources_servers/agentic_if_realworld/configs/reverify_native.yaml` for Gym
reverification without policy generation.
For a local native run, also save a redacted config using `gym env resolve` with
the same config and overrides, or use `gym eval submit`, which saves the config
and job record. Keep all native outputs, captures and judge receipts outside Git.

To recompute metrics from saved adaptive rollouts without policy or judge calls,
start the native profile's servers with the same private dataset configuration,
then aggregate through its standard environment route:

```bash
gym env start --config benchmarks/agentic_if_realworld/native.yaml --model-type vllm_model
# In another terminal with the same private data/judge environment:
gym eval aggregate --config benchmarks/agentic_if_realworld/native.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --input-glob /private/agentic-if/run-01/rollouts.jsonl \
  --output /private/agentic-if/run-01/rollouts.jsonl \
  ++policy_model_name=policy ++policy_base_url=http://localhost:8000/v1
```

Aggregation reads saved results only. Keep the judge configuration available for
server startup validation. The stock model config satisfies Gym's reference
validation; aggregation never calls its placeholder endpoint. The adaptive profile's legacy relay remains unchanged;
use the native profile above for the standalone aggregation command.

## Architecture review

The [Gym benchmark guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments/adding-a-benchmark)
supports external rollout drivers. The documented `gym eval run` command already
uses this supported hook and the standard SimpleAgent for generation. The hook
handles adaptive delivery selection, independent judging and saved-run accounting;
it does not replace the CLI entrypoint. This contribution needs an explicit
Gym maintainer decision on its recorded streaming model adapter and thin adaptive
driver: stock repeat/resume does not preserve the frozen first-complete schedule
and delivery receipts. The stock native profile is a separate protocol.
`app.py` delegates scoring to the verifier module. Verifier semantic metadata is
checked against the frozen dataset; extra Gym trajectory/row fields pass through
the transport model. Maintainers should confirm that metadata contract and the
applicable conformance registry/probes before platform acceptance.

The two Python files in `benchmark/policy/` are frozen source artifacts:
`validate_response_judge.py` and `test_contract.py`. Only those files are excluded
from automatic lint/format edits, which would change their recorded hashes.
`test_frozen_assets_are_unchanged` checks their original bodies; the surrounding
runtime and other policy-directory files use the normal repository checks.
Their headers record the [frozen source provenance](validation.md#frozen-code-provenance),
original filenames and Apache-2.0 license. Packaging headers are checked exactly
and removed only for the original-body hash comparison.

## Synthetic smoke without registry credentials or paid inference

Use a new private directory outside Git. If the ignored `data/` directory symlink
already points to another run, inspect and deliberately unlink that symlink first.

```bash
export AGENTIC_IF_DATA_DIR=/private/agentic-if/smoke/data
export AGENTIC_IF_PREPARED_PATH=/private/agentic-if/smoke/prepared.jsonl
export AGENTIC_IF_RECEIPT_DIR=/private/agentic-if/smoke/receipts
export AGENTIC_IF_SYNTHETIC=1
export AGENTIC_IF_POLICY_MODEL=synthetic-policy
export AGENTIC_IF_JUDGE_MODEL=synthetic-judge
export AGENTIC_IF_POLICY_ENDPOINT=http://127.0.0.1:19291/v1/chat/completions
export AGENTIC_IF_JUDGE_ENDPOINT=$AGENTIC_IF_POLICY_ENDPOINT
export AGENTIC_IF_POLICY_API_KEY=synthetic
export AGENTIC_IF_JUDGE_API_KEY=synthetic
python -c 'import os; from pathlib import Path; from resources_servers.agentic_if_realworld.benchmark.synthetic import create; create(Path(os.environ["AGENTIC_IF_DATA_DIR"]))'
python -m resources_servers.agentic_if_realworld.benchmark.mock_endpoint &
MOCK_PID=$!
trap 'kill "$MOCK_PID"' EXIT

gym eval prepare --benchmark agentic_if_realworld
gym env validate agentic_if_realworld
gym env test agentic_if_realworld
gym eval run --benchmark agentic_if_realworld --split benchmark --limit 1 \
  --num-repeats 1 --concurrency 1 --output /private/agentic-if/smoke/rollouts.jsonl \
  ++observability_enabled=true ++model_call_capture_dir=/private/agentic-if/smoke/captures \
  ++overwrite_metrics_conflicts=true
gym eval health-check /private/agentic-if/smoke
python scripts/inspect_harness_conformance.py --bundle /private/agentic-if/smoke/rollouts.jsonl \
  --no-tools --output /private/agentic-if/smoke/evidence
```

Run component tests with `python -m pytest resources_servers/agentic_if_realworld/tests`.
The synthetic command refreshes Gym's ignored input-metrics cache when switching
between the one-task and five-task fixtures. Frozen real-data hashes are still
checked independently; this flag does not change the dataset or scoring policy.
These exercise the production verifier, cache concurrency, recovery, frozen policy,
private registry integrity and blocked external destinations. They use synthetic
fixtures only. Inspect `quality_summary.json` and trajectory evidence as well as
the exit status. A full paid benchmark and a new reference score are separate
validation steps; no such result is claimed by the synthetic smoke.
The evidence check uses `--no-tools` because this benchmark never executes tools;
embedded historical tool definitions remain plain input text.

Public CI also runs the Resources Server's legacy data validator. Its five
independent synthetic tasks and recorded rollouts are documented in the
[server README](../../resources_servers/agentic_if_realworld/README.md#public-ci-examples).
They are separate from the 969 private tasks and need no registry or inference
credentials. Only the three named example artifacts are allowed through Git's
data ignore rules.
