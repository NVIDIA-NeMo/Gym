# Nemotron 3.5 Lightning 30B A3B Base — v0.2 evaluation recipes

> **Interim location.** Before this merges to `main` the `more/` folder moves to
> **`nemotron_recipes/nano-3.5/`**. Nothing in the configs depends on their path, so the
> move is a plain `git mv`.

These are `nemo-evaluator-launcher` configs run with `nel run`, not Gym recipes — see
[`../reproducibility.md`](../reproducibility.md) for how they sit alongside the instruct
recipes.

Evaluation configs for the **base (pretraining)** benchmarks of
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16)
on `nemo-evaluator-launcher` 0.2.6.

There are two configs:

- **Base suite** — [`base-suite.yaml`](./base-suite.yaml),
  the 21 short-context tasks across knowledge, math, code, commonsense, reading
  comprehension and multilingual. Runs against an OpenAI-compatible endpoint **you
  provide** (`deployment: none`) — see
  [Running against an existing endpoint](#running-against-an-existing-endpoint). The
  endpoint must support `echo` + `logprobs` for 9 of the tasks.
- **Long context (RULER)** — [`ruler.yaml`](./ruler.yaml),
  RULER at 64k / 128k / 256k / 512k / 1M. Also endpoint-based, but it needs a
  **separately served, long-context endpoint** — a normal deployment will not do. Also
  needs a pre-generated dataset. **Results are provisional** — see [RULER](#ruler).

Neither config deploys anything: you serve the model and point them at it. See
[Serving the model](#serving-the-model) for the exact commands.

**No judge is required.** Unlike several of the instruct recipes, nothing in the base
suite is LLM-graded — every task is scored deterministically by `lm-evaluation-harness`
(exact match, log-likelihood, or code execution) and RULER by `nemo-skills`. There is no
`JUDGE_API_KEY` and no judge endpoint to configure. What you need is:

- one OpenAI-compatible **`/v1/completions`** endpoint serving the base model, which
  **must honour `echo` together with `logprobs`** (9 of the 21 tasks are scored by
  log-likelihood), and
- for RULER only, a **second** endpoint of the same model served for long context.

### Reproducing the reference numbers

Reproduction depends on the *serving* configuration as much as on the eval configuration:
vLLM version, tensor parallelism, cache dtype, prefix caching and mamba cache mode all
move scores. So to compare against published numbers, **serve the model with the exact
command in [Serving the model](#serving-the-model)** and point the config at it.

If you instead point the config at an endpoint someone else operates, the suite still
runs, but a difference from the reference becomes ambiguous — you cannot tell whether it
came from that endpoint's serving stack or from the model. That is fine for measuring a
deployment you already have; it is not a basis for confirming published numbers.

For the instruct-model recipes, see [`../instruct/`](../instruct/).


---

## Common setup

### 0. Prerequisites

- **Docker**, running and usable by your user. `execution: local` runs each benchmark in
  a container, so the launcher shells out to Docker.
- **Python 3.10+** for the launcher itself.
- Outbound network access to `huggingface.co` (datasets, tokenizer) and `nvcr.io`
  (benchmark containers).
- No NGC login is required — the `nvcr.io/nvidia/eval-factory/*` containers used here
  pull anonymously.
- GPUs are **not** needed to run these two configs: the model is served elsewhere. You
  need GPUs only to serve it — see [Serving the model](#serving-the-model).

### 1. Install `nemo-evaluator-launcher`

```bash
python3 -m venv .venv
.venv/bin/pip install "nemo-evaluator-launcher[all]==0.2.6"
```

### 2. Credentials

Two environment variables are used:

| variable | needed for |
|---|---|
| `HF_TOKEN` | benchmark datasets and the tokenizer, both fetched from the Hub. Some datasets are gated — accept their terms on the Hub first, and make sure the token has access. |
| `POLICY_API_KEY` | the bearer token your endpoint expects. A self-hosted vLLM usually ignores it, so any placeholder works there. |

Export them, or put them in a `.env` file and pass `--env-file .env` — every example
below does the latter:

```bash
# .env
HF_TOKEN=hf_...
POLICY_API_KEY=...
```

The variable name for the endpoint key is set by `target.api_endpoint.api_key_name` in
the config; rename it there if you prefer a different one.

### 3. Bind mounts (optional but recommended)

Caching datasets and wheels across runs:

```yaml
execution:
  extra_docker_args: >-
    -v /home/you/.cache/huggingface:/cache/huggingface
    -v /home/you/.cache/uv:/cache/uv
```

---

## Serving the model

Both configs run against an endpoint you provide (`deployment: none`), so serve the model
first and point them at it. **They need different serving configurations** — see
[Long context](#long-context-serving) below for RULER.

Model card:
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16)
— check it for the authoritative serving guidance; the command below is what this suite
was validated with.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16 \
  --tensor-parallel-size=4 --pipeline-parallel-size=1 --data-parallel-size=1 \
  --port 8000 --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16 \
  --trust-remote-code --gpu-memory-utilization 0.9 --max-num-seqs 512 \
  --no-enable-prefix-caching \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
```

Container: `vllm/vllm-openai:v0.19.1-cu130` (multi-arch, amd64 + arm64).

**Sizing.** ~31.6B parameters in BF16, so roughly **62 GB of weights** plus KV cache.
TP=4 is what was validated; smaller TP works wherever the weights and cache fit. The
checkpoint declares `max_position_embeddings: 262144`.

**The endpoint must expose `/v1/completions`, and must honour `echo` together with
`logprobs`** — see [Running against an existing endpoint](#running-against-an-existing-endpoint)
for why, and for a one-command check. vLLM supports both out of the box.

### Long context serving

RULER will not run against the endpoint above. The checkpoint declares
`max_position_embeddings: 262144`, so anything past 256k is refused unless the server was
started to allow it. Serve a **second** endpoint for RULER:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16 \
  --tensor-parallel-size=4 --port 8000 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16 \
  --trust-remote-code --gpu-memory-utilization 0.9 --max-num-seqs 512 \
  --max-model-len 1100000 --mamba-cache-mode align \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
```

A 1.1M-token context costs a great deal of KV cache, which is why you would not serve the
short-context suite this way.

---

## Running

### Base suite

21 pretraining tasks against your endpoint:

```bash
nel run --config base-suite.yaml --env-file .env
nel run --config base-suite.yaml --env-file .env --dry-run    # preview
nel run --config base-suite.yaml --env-file .env -t adlr_mmlu # one benchmark
```

| Category | Tasks |
|---|---|
| General knowledge | `adlr_mmlu`, `adlr_mmlu_pro_5_shot_base`, `adlr_agieval_en_cot`, `adlr_gpqa_diamond_cot_5_shot` |
| Math | `adlr_gsm8k_cot_8_shot`, `adlr_minerva_math_nemo_4_shot`, `adlr_math_500_4_shot_sampled` |
| Code | `adlr_humaneval_greedy`, `adlr_humaneval_sampled`, `adlr_mbpp_sanitized_3_shot_greedy`, `adlr_mbpp_sanitized_3_shot_sampled` |
| Commonsense | `adlr_commonsense_qa_7_shot`, `adlr_arc_challenge_llama_25_shot`, `hellaswag`, `openbookqa`, `piqa`, `social_iqa`, `adlr_winogrande_5_shot` |
| Reading comprehension | `adlr_race` |
| Multilingual | `adlr_global_mmlu_lite_5_shot`, `adlr_mgsm_native_cot_8_shot` |

The `*_sampled` tasks repeat requests with the same seed, so response caching is
disabled for them in the config (lm-eval does its own caching).

> ⚠️ **Known issue — `adlr_minerva_math_nemo_4_shot` fails on
> `lm-evaluation-harness:26.03`.** The task's `doc_to_target` does
> `parse(doc["solution"])[1]`, and for exactly one document in the dataset
> (`EleutherAI/hendrycks_math`, `algebra` config, `test` index 670) `math_verify.parse()`
> returns a single-element list, so the run dies with `IndexError: list index out of
> range` during scoring:
>
> ```
> lm_eval/tasks/custom/adlr_minerva_math_nemo/nemo_utils.py:41, in doc_to_target
>     return parse(doc["solution"])[1] or extract_answer(doc["solution"])
> IndexError: list index out of range
> ```
>
> This is independent of the model — it is dataset/container-side, and it reproduces for
> any model. The other 20 tasks are unaffected. Until a container ships with the guard
> (`p = parse(...); (p[1] if len(p) > 1 else None) or extract_answer(...)`), either skip
> this task or run it with a container where it is fixed. Note that
> `lm-evaluation-harness:26.01` is *not* a workaround: its older `nemo-evaluator` rejects
> this config's `required_capabilities` field outright.

### Running against an existing endpoint

If you already have the model served somewhere, use
[`base-suite.yaml`](./base-suite.yaml)
— it deploys nothing. Point `target.api_endpoint.url` at your endpoint's
`/v1/completions` route, set `model_id` to the served name, and export
`POLICY_API_KEY` alongside `HF_TOKEN` (the tokenizer is still fetched from the Hub):

```bash
nel run --config base-suite.yaml --env-file .env
```

**This is a base model, so the suite uses the completions API, not chat.** A chat
endpoint is not a substitute — the base model has no chat template, and wrapping the
prompts in one changes what is measured.

**Your endpoint must support `echo: true` together with `logprobs`.** Nine of the 21
tasks are multiple-choice, scored by log-likelihood, and send:

```json
{"prompt": "...", "max_tokens": 1, "logprobs": 1, "echo": true, "temperature": 0, "seed": 1234}
```

| | tasks |
|---|---|
| **Need `echo` + `logprobs`** | `adlr_arc_challenge_llama_25_shot`, `adlr_commonsense_qa_7_shot`, `adlr_global_mmlu_lite_5_shot`, `adlr_race`, `adlr_winogrande_5_shot`, `hellaswag`, `openbookqa`, `piqa`, `social_iqa` |
| **Plain completions** | `adlr_agieval_en_cot`, `adlr_gpqa_diamond_cot_5_shot`, `adlr_gsm8k_cot_8_shot`, `adlr_humaneval_greedy`, `adlr_humaneval_sampled`, `adlr_math_500_4_shot_sampled`, `adlr_mbpp_sanitized_3_shot_greedy`, `adlr_mbpp_sanitized_3_shot_sampled`, `adlr_mgsm_native_cot_8_shot`, `adlr_minerva_math_nemo_4_shot`, `adlr_mmlu`, `adlr_mmlu_pro_5_shot_base` |

Some hosted services accept the request but silently ignore `echo`, which produces
**wrong scores rather than an error** — check before trusting the multiple-choice
numbers. Self-hosted vLLM supports both; the exact serve command this suite was
validated with is in the header of the endpoint config.

A quick way to check your endpoint before running anything:

```bash
curl -s "$URL" -H "Authorization: Bearer $POLICY_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"<served-name>","prompt":"Question: how to boil water\nAnswer: Put water in a pot",
       "max_tokens":1,"logprobs":1,"echo":true,"temperature":0}' \
  | python3 -c "import json,sys; c=json.load(sys.stdin)['choices'][0]; lp=c.get('logprobs') or {}; \
      print('echo:', c['text'][:40]); print('logprobs:', len(lp.get('tokens') or []))"
```

The response must echo the prompt back and return one logprob per prompt token. If
`logprobs` is empty or the text does not start with your prompt, the nine tasks above
will mis-score.

**Cross-checked at full scale.** All nine log-likelihood tasks were run at full sample
counts twice — once against a self-hosted vLLM 0.19.1 (TP=4) and once against a separate
OpenAI-compatible endpoint serving the same weights with vLLM 0.26.0 (TP=2):

| task | endpoint B (0.26.0) | endpoint A (0.19.1) | Δ |
|---|---|---|---|
| `adlr_commonsense_qa_7_shot` | 0.8026 | 0.8026 | 0.0000 |
| `hellaswag` | 0.6553 | 0.6560 | −0.0007 |
| `piqa` | 0.8319 | 0.8308 | +0.0011 |
| `openbookqa` | 0.3720 | 0.3700 | +0.0020 |
| `social_iqa` | 0.4770 | 0.4790 | −0.0020 |
| `adlr_global_mmlu_lite_5_shot` | 0.7600 | 0.7575 | +0.0025 |
| `adlr_arc_challenge_llama_25_shot` | 0.9241 | 0.9275 | −0.0034 |
| `adlr_race` | 0.8727 | 0.8689 | +0.0038 |
| `adlr_winogrande_5_shot` | 0.7901 | 0.7972 | −0.0071 |

Maximum deviation **0.0071**, across two different vLLM versions and TP layouts. This is
expected for log-likelihood scoring: it is an argmax over prompt-token logprobs at
temperature 0, so small numerical differences between builds rarely change the selected
choice. It is also the evidence that `echo` is genuinely honoured — an endpoint ignoring
it would drift toward chance, not track the reference to three decimals.

Generative tasks have no such guarantee: they sample, so expect ordinary run-to-run
variation there. The reference table under [Expected results](#expected-results) is from
a self-hosted deployment started with the serve command in
[Serving the model](#serving-the-model).

### RULER

```bash
nel run --config ruler.yaml --env-file .env
```

**Prepare the dataset — required, and not included.** RULER runs from a pre-generated
dataset rather than building it per request (generating in-run is slow and gives fewer
configuration options). You must generate it yourself with the RULER preparation
tooling in [nemo-skills](https://github.com/NVIDIA/NeMo-Skills), then bind-mount it:

```yaml
execution:
  extra_docker_args: >-
    -v /path/to/your/ruler_data:/ruler_data
```

Each task selects its slice through `extra.ruler.setup`, which the configs set to
`nanonext_<length>_100` (100 samples per length). Rename these to match whatever your
generated data provides — the names are just keys into your dataset directory, and the
run will fail if they do not resolve.

**Endpoint.** Point this config at a long-context endpoint served as described in
[Long context serving](#long-context-serving) — not at the one used for the base suite.
Expect the 512k and 1M tasks to be substantially slower and more memory-hungry than
everything else; budget time accordingly.

> ⚠️ **Long-context numbers from this recipe are provisional.** Our 512k measurement
> came out materially below the published long-context figure, and the discrepancy is
> not yet explained. Treat these scores as not-yet-validated, and prefer the model card
> for long-context claims.
>
> The published checkpoint declares `max_position_embeddings: 262144` with
> `rope_theta: 10000` and **no `rope_scaling`**, so the 512k and 1M tasks run 2× and 4×
> beyond the model's trained positional range — which is why
> `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is required to serve them at all. Scores at those two
> lengths should be read with that in mind. The 64k / 128k / 256k tasks are within
> range and are not affected.

### Output

Each run creates a timestamped invocation directory under `execution.output_dir`, with
one subdirectory per task:

```
<output_dir>/<timestamp>-<invocation-id>/<task>.<n>/
├── artifacts/
│   ├── results.yml                     # headline scores + the exact command that ran
│   ├── report.json / report.html       # rendered summary
│   ├── eval_factory_metrics.json       # request counts, latencies, HTTP status codes
│   └── <served-model-name>/
│       └── results_<timestamp>.json    # raw lm-eval output
└── logs/                               # client + server logs
```

Start with `results.yml` for scores, and **check `eval_factory_metrics.json` before
trusting them** — `response_stats.status_codes` should be `{'200': N}` with
`count == successful_count`. Anything else means some requests failed and the score was
computed on partial data.

Override the location with `-o "execution.output_dir=/absolute/path"`. For
`resume_from_cache` to survive retries, `output_dir` must be an absolute path on stable
storage (not tmpfs).

---

## Monitoring & customization

```bash
nel status ; nel logs <job-id>                                                              # status / logs
nel run --config <cfg> -o execution.output_dir=/abs/path                                    # override output
nel run --config <cfg> -o evaluation.nemo_evaluator_config.config.params.limit_samples=10   # smoke test only — never for reported numbers
```

## Troubleshooting

- **Gated HF datasets** — accept the terms on the Hub; `HF_TOKEN` must have access.
- **Timeouts** — `request_timeout` is 3600s; on rate limits lower `parallelism`.
- **Out of memory on RULER** — lower `--max-num-seqs`, or raise
  `tensor_parallel_size`; the 512k/1M tasks are the ones that will hit it first.
- **`limit_samples` results** — never report them; they exist to validate setup only.

## Expected results

Comparable to the model card; expect slight variation on the sampled tasks
(`temperature > 0`). Never report `limit_samples` runs.

This recipe has been run end to end against the published weights and reproduces the
reference numbers. Across the 15 tasks with a reference value to compare against, every
one agreed to within 0.02 absolute, and 12 of 15 to within 0.006 — for example
`adlr_mmlu` 0.7859 vs 0.7859, `adlr_gsm8k_cot_8_shot` 0.9158 vs 0.9128, `hellaswag`
0.6560 vs 0.6557, `piqa` 0.8308 vs 0.8335.

## License

Apache 2.0 — see the repository `LICENSE`.
