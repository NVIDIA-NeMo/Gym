# HLE Benchmark

Benchmark wrapper for [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle), a
2158-question (text-only subset) exam covering graduate-level STEM and humanities knowledge.

- **Tasks**: 2158 text-only questions (image questions filtered at prepare time)
- **Reward**: binary; LLM judge checks whether the model's response matches the ground-truth answer
- **Metrics**: `pass@1/judge_accuracy` — fraction of questions judged correct

The judge uses the official HLE evaluation prompt adapted from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle), which extracts the model's
final answer and checks it against the expected answer with a yes/no verdict. The policy model
serves as the judge — no separate judge server is needed.

## Dataset access

`cais/hle` is a gated HuggingFace dataset. Request access at
[https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle), then
authenticate:

```bash
huggingface-cli login
```

## Prepare benchmark data

```bash
gym eval prepare --benchmark hle
```

Downloads `cais/hle`, filters to text-only questions, and writes
`benchmarks/hle/data/hle_benchmark.jsonl`.

### Vision (multimodal) subset

HLE's image questions are exposed as a variant config in this directory,
`config_vision.yaml`, i.e. the benchmark `hle/config_vision`:

```bash
gym eval prepare --benchmark hle/config_vision
```

This downloads the full `cais/hle` split (2500 questions: the same 2158 text
questions plus 342 image questions) and writes
`benchmarks/hle/data/hle_benchmark_vision.jsonl`. Unlike the text-only file,
these rows are **fully materialized** — the prompt template is baked into
`responses_create_params.input`, and image questions carry an `input_image`
block (base64 data URI). Because the input is pre-populated, this dataset uses
`prompt_config: null` (the two are mutually exclusive; a prompt template can
only produce string content, so an image block has to be built at prepare
time).

The `include_vision` flag lives on `prepare.py`; `prepare_vision.py` is a thin
wrapper that calls `prepare(include_vision=True)`. Running the text-only
prepare directly with the flag also works:

```bash
python benchmarks/hle/prepare.py --include-vision
```

Evaluating the vision variant requires a vision-capable policy model:

```bash
gym env start --model-type vllm_model --benchmark hle/config_vision
```

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark hle
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

For the vision variant use `--benchmark hle/config_vision` (agent
`hle_vision_equivalence_llm_judge_simple_agent`) with a vision-capable policy
model.

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent hle_equivalence_llm_judge_simple_agent \
    --input benchmarks/hle/data/hle_benchmark.jsonl \
    --output results/hle_rollouts.jsonl \
    --prompt-config benchmarks/hle/prompts/default.yaml \
    --num-repeats 1 \
    --temperature 0.0
```

Use `temperature: 0.0` to match the nemo-skills evaluation setup and ensure reproducible scores.

For the vision variant, drop `--prompt-config`: those rows already carry
`responses_create_params.input`, and passing a prompt config would overwrite it
(the run fails with "Some rows have responses_create_params.input but
prompt_config is also specified").

```bash
gym eval run --no-serve \
    --agent hle_vision_equivalence_llm_judge_simple_agent \
    --input benchmarks/hle/data/hle_benchmark_vision.jsonl \
    --output results/hle_vision_rollouts.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

## Metrics

`pass@1/judge_accuracy` is the headline metric.

## Sandboxed agents

Use `--benchmark hle/opencode` or `hle/pi` for Python tools without search, or
`hle/opencode_search` or `hle/pi_search` to enable Tavily search as well.
These text-only presets preserve standard preparation and grading, with the
Explanation/Answer/Confidence instructions in a single user message and one repeat.
Keep collection repeats at one.

See the shared [OpenCode and Pi setup guide](../../responses_api_agents/opencode_sandboxed_agent/README.md#offline-scientific-evaluation-with-opencode-or-pi)
for images, credentials, network policies, token limits, and failure handling.

The search presets supply the public HLE [exclusion list](search_exclusions.json).
Deployments requiring an additional policy must supply their combined policy file
through the Tavily resource's `exclude_domains_file_path`; keep private policies
outside the public repository. This list reduces answer exposure but cannot
guarantee that every answer-bearing page is excluded.

### Domain metrics

Text and vision preparation retain `category`, `raw_subject` and explicit
`has_image` metadata. The HLE and HLE-Verified judge configurations enable `report_hle_domains`; existing cached
HLE datasets need to be prepared again to retain these labels.

Aggregation emits flat `domains/<field>/<URL-encoded-label>/...` keys in
`agent_metrics`: question/rollout counts, `mean/reward` (accuracy as a fraction),
`mean/output_tokens`, `ci_low_95_across_repeats/<metric>` /
`ci_high_95_across_repeats/<metric>`.
Modality slices use `domains/modality/text/...` and `domains/modality/vision/...`
with the same metrics. Missing `has_image` metadata is counted and never assumed
to be text. The overall headline includes the full evaluated set; its score and
grading are unchanged.
Missing labels are counted; incomplete token usage suppresses the token mean and
interval rather than silently averaging a subset. Masked outcomes are excluded
by the normal aggregation pipeline.

Each slice reuses Gym's public `compute_aggregate_metrics` function for means and
Student-t intervals across repeat-level means, with at least two repeats covering
the same questions. `ci_available/<metric>` indicates
availability; single-repeat evaluations have no default interval. Bounds are not
clipped. These intervals measure generation variability on a fixed benchmark.
