# HotpotQA Closed-Book

Closed-book variant of the [HotpotQA](https://hotpotqa.github.io/) multi-hop
question-answering benchmark — same questions as the open-book variant, no
context provided. Ported from NeMo Skills' `hotpotqa_closedbook` benchmark.

The benchmark binds to the `hotpotqa_qa` resource server, which uses
deterministic verification (no LLM judge): SQuAD-style EM/F1, alternative-aware
substring matching, and Skills' "unfiltered + filtered" reporting (filtered
metrics drop tasks whose ground-truth answer is too long or looks like a
multi-word proper name).

## Prepare data

`prepare.py` downloads the HuggingFace `hotpotqa/hotpot_qa` distractor
validation split — the exact same source used by Skills'
`nemo_skills/dataset/hotpotqa_closedbook/prepare.py` — and writes
`data/hotpotqa_closedbook_benchmark.jsonl` with one row per problem
(`question`, `expected_answer`, `id`, `type`, `level`).

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/hotpotqa_closedbook/config.yaml]"
```

## Run servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,benchmarks/hotpotqa_closedbook/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=hotpotqa_closedbook_simple_agent \
    +input_jsonl_fpath=benchmarks/hotpotqa_closedbook/data/hotpotqa_closedbook_benchmark.jsonl \
    +output_jsonl_fpath=results/hotpotqa_closedbook_rollouts.jsonl \
    +num_repeats=4 \
    +num_repeats_add_seed=true
```

## Verification

Fully deterministic, matching Skills:

1. **JSON answer extraction** — the prompt instructs the model to emit
   `{"answer": "..."}`. The verify endpoint extracts the predicted answer
   from the last valid JSON object containing an `answer` key (so even if
   the model emits chain-of-thought followed by JSON, the JSON wins).
2. **SQuAD-style EM and F1** — official HotpotQA answer normalization
   (lowercase, strip articles, strip punctuation, collapse whitespace)
   followed by token-overlap F1 / exact-match.
3. **Alternative-aware substring matching** — the ground-truth answer is
   expanded into surface-form variants (article stripping, parens
   normalization, number-word ↔ digit, ampersand ↔ "and", hyphen handling)
   and any variant being a substring of the model output counts as a
   `is_correct` (lenient) match. `is_correct_strict` adds word-boundary
   guards for short alternatives (≤4 chars) and a position guard for long
   model answers (>80 chars).
4. **Unfiltered + filtered metrics** — every metric is reported twice: once
   over all tasks, and once excluding tasks whose ground-truth answer is
   flagged as unreliable for substring evaluation (length >40, or 3-6 word
   strings that look like multi-word proper names).

The reward emitted by `/verify` is `is_correct_strict`. The full set of
scores (`answer_em`, `answer_f1`, `is_correct`, `is_correct_strict`) is
returned in the verify response so downstream metric aggregation can compute
pass@k for each channel.

> **Reasoning-model note**: start the policy vLLM server with
> `--reasoning-parser deepseek_r1` (or the model-specific parser). This
> strips `<think>...</think>` so the JSON-extractor sees clean post-think
> text. Without it the JSON object embedded inside the reasoning trace can
> be picked up instead of the final answer.

## Metrics

For each of the four scoring channels (`answer_em`, `answer_f1`,
`is_correct`, `is_correct_strict`), the resource server emits:

- `pass@1[avg-of-k]/<channel>`, `pass@k/<channel>` — averaged across all
  tasks.
- `majority@k/<channel>` — majority-vote over the `extracted_answer` field.
- `filtered_pass@1[avg-of-k]/<channel>`, `filtered_pass@k/<channel>`,
  `filtered_majority@k/<channel>` — same metrics computed only on tasks
  whose ground truth was not flagged for removal.
- `unfiltered_num_tasks`, `filtered_num_tasks` — the size of each pool.

The headline metric (selected by `get_key_metrics`) is the highest-k
`pass@1[avg-of-k]/is_correct_strict` from both the unfiltered and filtered
pools — this is what Skills' `HotpotQAMetrics` ranks the closed-book run
on.
