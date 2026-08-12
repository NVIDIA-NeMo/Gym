# BioMysteryBench

Native NeMo Gym integration for Anthropic's gated
[BioMysteryBench-full](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full).
It uses `anyterminal_agent` with a native Gym sandbox and resources-server verifier.

Each policy-agent sandbox receives one extracted task directory read-only at `/data` and a
writable `/workspace`. The answer rubric remains on the Gym host. After the sandbox exits, the
captured final response is sent to `biomysterybench_judge`, which performs binary rubric grading
and checks captured tool calls for prohibited active GEO/SRA/ENA/BioProject lookups. Each task's
network allowlist is appended to the agent prompt, and explicit off-list URLs in captured tool calls
receive zero credit.

## Usage restrictions

BioMysteryBench is evaluation-only. Its access terms prohibit using its questions, rubrics, or
task formulation to train, fine-tune, reinforce, or distill a model. Do not register this data as
a Gym train/validation dataset and do not use these rollouts for optimization.

The default `biomysterybench` benchmark is pinned to the exact pre-audit revision used for
Anthropic's published result:
`a066d4135d087934f1c5399f45ca7f2cd4bd0675` (99 tasks: 76 human-solvable and 23 human-hard).
The separate `biomysterybench_v11` benchmark pins the corrected release at
`b5a889c4757214ec9a6ade876b734f920a7799db` (90 tasks: 73 human-solvable and 17 human-hard).
The gated archives are 147.85 GiB compressed for the official 99-task pin and 144.70 GiB for v11.
The Hugging Face cache, extracted data, and results coexist, so budget substantially more space.

## Build the shared runtime

```bash
docker build \
  -t biomysterybench-runtime:v12 \
  -f benchmarks/biomysterybench/docker/Dockerfile \
  benchmarks/biomysterybench
```

Task data is mounted at runtime, so the shared image never contains gated data or answer rubrics.

## Credentials and model configuration

Accept the dataset gate on Hugging Face, then export credentials without committing them:

```bash
export HF_TOKEN=...
export NVIDIA_API_KEY=...
```

For the Anthropic-reported Opus 4.6 comparison, configure the NVIDIA inference gateway in the
repository-root `env.yaml`:

```yaml
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: ${oc.env:NVIDIA_API_KEY}
policy_model_name: azure/anthropic/claude-opus-4-6
```

The judge reuses the policy model on the same gateway. Its chat-completions request sends
`temperature` only because NVIDIA-hosted Anthropic models reject requests that specify both
`temperature` and `top_p`.

## Prepare

Prepare one task for an initial test run of the official release:

```bash
uv run gym eval prepare --benchmark biomysterybench_test
```

Prepare all 99 tasks used in the official result:

```bash
uv run gym eval prepare --benchmark biomysterybench
```

Prepare corrected v11 instead with
`uv run gym eval prepare --benchmark biomysterybench_v11`.

Preparation downloads only selected archives, safely extracts them under the gitignored
`benchmarks/biomysterybench/data/cache/`, and writes an absolute-path benchmark JSONL. Re-running
preparation reuses archives from the Hugging Face cache and hash-validated extracted directories.

## Run

Run the one-task, one-repeat test benchmark first:

```bash
uv run gym eval run \
  --benchmark biomysterybench_test \
  --model-type vllm_model \
  --model azure/anthropic/claude-opus-4-6 \
  --model-url https://inference-api.nvidia.com/v1 \
  --model-api-key "$NVIDIA_API_KEY" \
  --split benchmark \
  --output results/biomysterybench/opus_4_6_test.jsonl \
  --concurrency 1
```

Run the complete published-release benchmark:

```bash
uv run gym eval run \
  --benchmark biomysterybench \
  --model-type vllm_model \
  --model azure/anthropic/claude-opus-4-6 \
  --model-url https://inference-api.nvidia.com/v1 \
  --model-api-key "$NVIDIA_API_KEY" \
  --split benchmark \
  --output results/biomysterybench/opus_4_6_official_99.jsonl \
  --concurrency 2
```

The rollout collector supports resumable execution. Add `--resume` when restarting the same output
path after an interruption.

The benchmark config requests five repeats per problem, matching Anthropic's reported evaluation
protocol. Headline metrics are average accuracy over repeats, plus human-solvable/human-hard subset
accuracy and per-problem consistency. Infrastructure failures are marked with `mask_sample` and
must not be silently scored as biological failures.

## Fidelity notes

Anthropic's original article reports results on the pre-audit 99-task release. That exact gated
revision is the default here; corrected v11 is intentionally a different benchmark name. The
runtime follows Anthropic's subsequently disclosed bash-in-Docker tool categories (alignment,
sequence processing, Python 3.11, R 4.3/Bioconductor, FastQC, and Nextflow), while pinning an
independently reproducible image. The exact original image versions, resource limits, and all judge
details are still not public, so disclose this harness when comparing scores with Anthropic's figure.

## Licensing

Integration code: Apache 2.0.

Problem statements, rubrics, and task formulation: CC BY 4.0 plus the upstream evaluation-only
access condition. Data archives are anonymized derivatives of public biological archives and remain
subject to their original repositories' data-use policies.
