# HLE-Verified Benchmark

Benchmark wrapper for [HLE-Verified](https://huggingface.co/datasets/skylenage/HLE-Verified),
a 1600-question (text-only subset) exam covering graduate-level STEM and humanities knowledge.

- **Tasks**: 1600 text-only questions from the Gold and Revision subsets
- **Reward**: binary; LLM judge checks whether the model's response matches the ground-truth answer
- **Metrics**: `mean/reward` — fraction of questions judged correct

The judge uses the official HLE evaluation prompt adapted from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle), which extracts the model's
final answer and checks it against the expected answer with a yes/no verdict. The policy model
serves as the judge — no separate judge server is needed.

## Dataset access

If dataset access requires authentication, log in to HuggingFace:

```bash
huggingface-cli login
```

Gym also reads a token from `hf_token` in `env.yaml` if one is set.

## Prepare benchmark data

```bash
gym eval prepare --benchmark hle_verified
```

Downloads `skylenage/HLE-Verified`, filters to text-only questions from the Gold and Revision
subsets, and writes `benchmarks/hle_verified/data/hle_verified_benchmark.jsonl`.

### Vision (multimodal) subset

The vision variant includes image questions and is configured in `config_vision.yaml`:

```bash
gym eval prepare --benchmark hle_verified/config_vision
```

This prepares 1811 questions (1600 text questions and 211 image questions) and writes
`benchmarks/hle_verified/data/hle_verified_benchmark_vision.jsonl`. These rows include
the prompt in `responses_create_params.input`, with an `input_image` block for image
questions. The dataset uses `prompt_config: null` because its inputs are already populated.

The preparation script also accepts the vision flag directly:

```bash
python benchmarks/hle_verified/prepare.py --include-vision
```

Evaluating the vision variant requires a vision-capable policy model.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark hle_verified
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

For the vision variant:

```bash
gym env start --model-type vllm_model --benchmark hle_verified/config_vision
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent hle_verified_equivalence_llm_judge_simple_agent \
    --output results/hle_verified_rollouts.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

The dataset and prompt are loaded from the benchmark config.

For the vision variant:

```bash
gym eval run --no-serve \
    --agent hle_verified_vision_equivalence_llm_judge_simple_agent \
    --output results/hle_verified_vision_rollouts.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

## Metrics

`mean/reward` is the headline metric. `judgement_parsing_issue_rate` reports the
fraction of judged rollouts with parsing issues.

## Judge recovery and answer length

Both variants score final assistant answers longer than 100,000 characters zero
without sending them to the judge. The full response is retained, with
`failure_reason` starting with `final_answer_too_long`. The check excludes
reasoning, counts the last assistant message before regex extraction, and does
not stop generation early. This is a benchmark scoring guard; set the resources
server's `max_answer_chars` to `null` to disable it when reproducing an uncapped
grading protocol. Other equivalence-judge benchmarks have no limit by default.

HTTP 408 is retried automatically by the OpenAI-compatible model client. For a
judge endpoint known to return transient HTTP 404 routing errors, load
`--config benchmarks/hle_verified/judge_recovery.yaml` instead of the plain
judge-model config, and point the benchmark resources server's
`judge_model_server.name` at `judge_model`. Supply `judge_base_url`, `judge_api_key`,
and `judge_model_name` as usual. This opts only that judge client into 404 retries,
with five total HTTP attempts and exponential backoff. Unconfigured clients do
not retry 404. Judge retries preserve the request and do not rerun the policy.

If attempts are exhausted, the shared judge failsafe saves the policy response
in the failures sidecar as `judge_failed`, excluded from scored aggregates.
Recover these rows with `gym eval reverify --judge-failed-only` and the same
verifier/judge configuration. Successful grades are preserved. This recovery path
applies to new sidecar-tagged failures; older in-band `JUDGE_ERROR` rows require
migration before using it. A run with pending grades is incomplete.
