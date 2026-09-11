# HLE-Verified Benchmark

Benchmark wrapper for [HLE-Verified](https://huggingface.co/datasets/skylenage/HLE-Verified), an
expert re-verification of [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle).
Every HLE question was re-checked by domain experts and sorted into three classes:

| Class       | Rows | Meaning                                                                     |
| ----------- | ---- | --------------------------------------------------------------------------- |
| `gold`      |  668 | Question and reference answer confirmed correct as-is.                       |
| `revision`  | 1143 | The upstream reference answer was wrong or under-specified; corrected here.  |
| `uncertain` |  689 | Annotators could not confirm the answer — unreliable to grade against.       |

- **Tasks**: Gold + Revision, image-free — 1600 rows (575 gold, 1025 revision). Same split
  NeMo Skills uses via `EVAL_SPLIT = "text"`. Image questions are dropped at prepare time;
  see [Vision variant](#vision-variant) to keep them.
- **Reward**: binary; an LLM judge checks whether the model's response matches the
  (re-verified) ground-truth answer.
- **Metrics**: `mean/reward` — fraction of questions judged correct — plus
  `judgement_parsing_issue_rate`. See [Metrics](#metrics).

The point of this benchmark over plain `hle` is the answer key: HLE has a non-trivial rate of
incorrect reference answers, which caps measurable accuracy and penalizes models that are
actually right. That Revision outnumbers Gold nearly 2:1 is the size of the problem — of the
questions the annotators could confirm at all, most needed their answer corrected. Scores here
are therefore **not** comparable to `hle` scores.

Verification is otherwise identical to `benchmarks/hle`: the official HLE judge prompt from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) extracts the model's final
answer and returns a yes/no verdict. The policy model serves as the judge — no separate judge
server is needed (see [Separate judge model](#separate-judge-model)).

The judge prompt is not copied here — `config.yaml` points at `benchmarks/hle/prompts/judge.txt`,
the very file `benchmarks/hle` uses. The two benchmarks exist to be compared, so they must grade
identically; a duplicated prompt would drift the first time one of them was edited.

## Dataset access

`skylenage/HLE-Verified` embeds questions from the gated `cais/hle` dataset. If the download
403s, request access at [https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle),
then authenticate:

```bash
huggingface-cli login
```

Gym also reads a token from `hf_token` in `env.yaml` if one is set.

## Prepare benchmark data

```bash
gym eval prepare --benchmark hle_verified
```

Downloads `skylenage/HLE-Verified`, keeps the image-free Gold + Revision rows (1600), and
writes `benchmarks/hle_verified/data/hle_verified_benchmark.jsonl`.

`--subset` selects verified classes only — `text` (the default: Gold + Revision), `gold`,
`revision`, `uncertain`, or `all`. The name `text` is inherited from Skills' `EVAL_SPLIT`;
modality is a separate axis, controlled by `--include-vision`.

To prepare a single class (a Gold-only run, or to inspect the Uncertain rows), call the script
directly and give it its own output file — the default path is the one `config.yaml` points at,
so writing a different subset there would silently change what `--benchmark hle_verified`
evaluates:

```bash
python benchmarks/hle_verified/prepare.py \
    --subset gold \
    --output benchmarks/hle_verified/data/hle_verified_gold.jsonl
```

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark hle_verified
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in `env.yaml` (or passed
as CLI overrides).

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent hle_verified_equivalence_llm_judge_simple_agent \
    --output results/hle_verified_rollouts.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

Use `temperature: 0.0` to match the NeMo Skills evaluation setup and keep scores reproducible.

The dataset and prompt come from the config, not the command line: `gym eval run` re-derives
its input from the prepared dataset and overwrites `input_jsonl_fpath`, so passing `--input`
does not subset the run. To evaluate a different row set, prepare it as its own dataset (see
above) and point a config at it.

## Vision variant

`--subset` selects verified classes; `--include-vision` decides modality. The two are
independent, so including images does not pull in the Uncertain rows.

```bash
gym eval prepare --benchmark hle_verified/config_vision
```

Keeps the Gold + Revision rows **including** their image questions — 1811 rows, 211 of them
with images, ~64 MB — and writes `data/hle_verified_benchmark_vision.jsonl`. This is a second
config in the same benchmark folder, not a separate benchmark (the same layout `benchmarks/hle`
uses), so it is addressed by config path:

```bash
gym env start --model-type vllm_model --benchmark hle_verified/config_vision

gym eval run --no-serve \
    --agent hle_verified_vision_equivalence_llm_judge_simple_agent \
    --output results/hle_verified_vision_rollouts.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

Unlike the text rows, vision rows are fully materialized: `prepare_vision.py` applies
`prompts/default.yaml` at prepare time and image questions carry an `input_image` block, so
`config_vision.yaml` sets `prompt_config: null` — a pre-populated
`responses_create_params.input` and a `prompt_config` are mutually exclusive. Multimodal
inputs are built by `benchmarks/hle`'s `_build_input`, shared for the same reason the judge
prompt is: the two benchmarks must prompt identically to stay comparable.

Evaluating this requires a **vision-capable policy model**. The judge stays text-only — it
compares the model's final answer against the reference, and never sees the image.

NeMo Skills has no vision counterpart for HLE-Verified, so unlike the text split this one is
Gym-only and not comparable to a Skills run.

## Separate judge model

By default the policy model grades itself. To grade with a different model, point the
resources server's `judge_model_server` at a second `responses_api_models` instance — the
same overlay pattern `benchmarks/hle` uses:

```yaml
hle_verified_equivalence_llm_judge_resources_server:
  resources_servers:
    equivalence_llm_judge:
      judge_model_server:
        type: responses_api_models
        name: judge_model
      judge_responses_create_params:
        input: []
        temperature: 0.0
        top_p: 1.0
        max_output_tokens: 4096
```

Give the judge enough `max_output_tokens` to emit its full reasoning **and** the trailing
`Judgement: yes/no` line. A truncation that severs the verdict is parsed as not-equal, so the
row scores zero regardless of the answer — watch `judgement_parsing_issue_rate` below, which
counts exactly those rows.

## Metrics

`mean/reward` is the headline number: the fraction of questions the judge accepted. The reward
is binary per row, so with `--num-repeats 1` it is plain accuracy; repeats add
`mean_across_repeats/mean/reward` and its confidence interval.

Rows carry `category`, `raw_subject`, and `verified_class` (`gold` / `revision`) so rollouts
can be sliced after the fact — in particular, comparing Gold vs Revision accuracy shows how
much of a model's HLE score was lost to bad upstream answer keys. Vision rows additionally
carry `has_image`.

`judgement_parsing_issue_rate` reports the fraction of rows whose judge verdict did not parse
cleanly. It is emitted even at `0.0` and promoted into `key_metrics` alongside the reward,
because a judge that stops emitting verdicts produces the same column of zeros as a model that
got everything wrong.

Not every flagged row is mis-scored, so read the per-kind rates beside it rather than the
headline alone:

| Kind                       | Effect on the score                                              |
| -------------------------- | ---------------------------------------------------------------- |
| `no_verdict`               | Forces 0 — accuracy is understated by these rows.                 |
| `unparseable_judge_output` | Forces 0 — same.                                                  |
| `truncated_judge_output`   | Only recorded when truncation cost the verdict, so also a 0.      |
| `conflicting_verdicts`     | A verdict *was* picked (the last one); scored, but worth a look.  |
| `repeated_verdict`         | Same — scored, informational.                                     |

A judge cut off *after* its `Judgement:` line is not flagged: the HLE judge prompt puts
`Confidence:` last, so those rows are graded correctly. Affected rows carry
`judgement_parsing_issues` in the rollout JSONL; clean rows omit the field:

```bash
jq '.judge_evaluations[].judgement_parsing_issues[]?' rollouts.jsonl | sort | uniq -c
```
