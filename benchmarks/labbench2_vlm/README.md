# LabBench2 VLM

[LabBench2](https://huggingface.co/datasets/EdisonScientific/labbench2) is a
scientific figure and table question answering benchmark for vision-language
models, plus protocol troubleshooting (`protocolqa2`). This config bundles all
subtasks (`figqa2-img`, `figqa2-pdf`, `tableqa2-img`, `tableqa2-pdf`,
`protocolqa2`) into a single benchmark run. Per-tag metrics are emitted
alongside the overall score via `verifier_metadata.tag`.

## Configuration

Uses the `labbench2_vlm` resource server with the `labbench2_vlm_agent` custom
agent. Scoring is LLM-as-judge (`[[A=B]]` / `[[A!=B]]`). Media files (images
and PDFs) are referenced by path in the JSONL and embedded at rollout time by
the agent.

### Judge setup

The benchmark chains in `resources_servers/labbench2_vlm/configs/judge_model_openai.yaml`,
which targets an OpenAI-compatible hosted endpoint. To use a different judge
(e.g. a local vLLM), drop that chain and supply your own `responses_api_models`
instance named `judge_model` — see `responses_api_models/vllm_model/configs/vllm_model.yaml`
for the vLLM form.

Credentials go in `env.yaml` at the **repository root** (the parser loads
`$CWD/env.yaml` first, then falls back to `$PARENT_DIR/env.yaml`). The file
is gitignored — create it if missing.

```yaml
# env.yaml — policy, judge credentials, HF token
hf_token: <your-hf-token>      # required: dataset is gated
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: <your-api-key>
policy_model_name: openai/openai/gpt-5.2
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: <your-api-key>
judge_model_name: openai/openai/gpt-5-mini
```

`judge_model_openai.yaml` reads the three `judge_*` keys via
`${oc.select:judge_*,…}` — you can override them on the CLI
(`+judge_base_url=… +judge_api_key=… +judge_model_name=…`) instead of
putting them in `env.yaml`.

## Prepare data

The source dataset [EdisonScientific/labbench2](https://huggingface.co/datasets/EdisonScientific/labbench2)
is **gated** — accept the terms on the HF page, generate a token at
https://huggingface.co/settings/tokens, and set `hf_token` in `env.yaml`
(see above).

```bash
gym eval prepare --benchmark labbench2_vlm
```

Downloads the subtask splits from HuggingFace and media files (images,
PDFs, protocol PDFs) from a public GCS bucket into `resources_servers/labbench2_vlm/data/media/`
(gitignored), then writes a single combined JSONL to
`benchmarks/labbench2_vlm/data/labbench2_vlm_benchmark.jsonl` (gitignored).
First run is slow — hundreds of media files plus the dataset download.
Re-runs are fast (HF cache + GCS "skip if exists").

### Example smoke data

The resource server also keeps a small committed smoke set at
`resources_servers/labbench2_vlm/data/example.jsonl` with media copied under
`resources_servers/labbench2_vlm/data/test_media/`. Regenerate it from the full
LABBench2 source with:

```bash
.venv/bin/python resources_servers/labbench2_vlm/prepare_data.py \
  --tags protocolqa2 figqa2-img figqa2-pdf tableqa2-img tableqa2-pdf \
  --example
```

`--example` writes at most five rows. It takes two rows from the first selected
tag, then one row from each subsequent tag until the five-row cap is reached. So
with the tag order above, the smoke set contains two `protocolqa2` rows and one
row each for `figqa2-img`, `figqa2-pdf`, and `tableqa2-img`; `tableqa2-pdf` is
prepared in its validation JSONL but is not included in `example.jsonl`.

After changing `example.jsonl`, regenerate its static validation metrics:

```bash
.venv/bin/gym dataset collate \
  --resources-server labbench2_vlm \
  --config resources_servers/labbench2_vlm/configs/judge_model_openai.yaml \
  --config responses_api_models/openai_model/configs/openai_model.yaml \
  --mode example_validation \
  --output-dir /tmp/labbench2_vlm_example_validation \
  +overwrite_metrics_conflicts=true
```

This updates `resources_servers/labbench2_vlm/data/example_metrics.json`.
Use a temporary `+output_dirpath` so the collated validation artifacts do not
overwrite source data. The full config chain is required because
`labbench2_vlm.yaml` references both `policy_model` and `judge_model`.

## Usage

```bash
# Start servers
gym env start \
    --benchmark labbench2_vlm \
    --model-type openai_model

# Collect rollouts
gym eval run --no-serve \
    --benchmark labbench2_vlm \
    --model-type openai_model \
    --agent labbench2_vlm_benchmark_simple_agent \
    --input benchmarks/labbench2_vlm/data/labbench2_vlm_benchmark.jsonl \
    --output results/labbench2_vlm.jsonl
```

`+agent_name` and `+input_jsonl_fpath` are both required — rows in the
prepared JSONL don't carry an `agent_ref`, and `gym eval run --no-serve` doesn't
read the path from the benchmark config.

For **protocolqa2** as **extracted text** with a text-capable policy model, pass
`+media_mode=text`. That setting applies **only** to rows whose tag is
`protocolqa2`; figure/table rows in the same JSONL still use images. So you can
use `+media_mode=text` with the combined benchmark JSONL or `example.jsonl` without
breaking figqa2/tableqa2. Omit it (default `media_mode=image`) to render protocol
PDFs as pages like other PDF tasks.

(See `resources_servers/labbench2_vlm/run.sh` for a protocol-only rollout using
`+media_mode=text`.)

### One-shot alternative

`gym eval run` starts the server stack, preprocesses, and
collects rollouts in a single command (don't run `gym env start` separately).
Input path and agent ref are auto-derived from the `type: benchmark`
dataset entry in the chained config:

```bash
gym eval run \
    --benchmark labbench2_vlm \
    --model-type openai_model \
    --split benchmark \
    --output results/labbench2_vlm.jsonl \
    --concurrency 16
```

For a fast smoke test, add `--limit 10 --num-repeats 1`.

`num_repeats` defaults to 3. Bump higher for tighter variance on the
judge-based reward.

## Throttling

Each in-flight sample fans out to one policy call + one judge call, so the
endpoints see roughly `2 × concurrency` concurrent requests.
On a hosted endpoint you'll likely hit rate limits or socket errors
(`Hit N global ClientOSError`) well before saturating your machine.

Cap concurrency with `--concurrency <N>`:

```bash
gym eval run --no-serve ... --concurrency 16
```

Start around 16 and bump up if it holds.
