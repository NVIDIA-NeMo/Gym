# PerceptionBench

[PerceptionBench](https://github.com/MoonshotAI/PerceptionBench) evaluates ten atomic visual-perception
capabilities in Multimodal Large Language Models across **3,000 verified short-answer VQA items**
(1,800 decomposed from attributed perception failures across 42 source benchmarks and 1,200 newly
authored). Every item is open-ended with a uniquely determined short reference answer and is graded 0/1
by an LLM judge.

## Provenance

- **Upstream code & paper:** [MoonshotAI/PerceptionBench](https://github.com/MoonshotAI/PerceptionBench)
  at commit `ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e` (arXiv: [2607.24957](https://arxiv.org/abs/2607.24957),
  evaluator `eval/eval.py` and `eval/judge_prompt.txt`, license **Apache-2.0**).
- **Upstream dataset:** [`moonshotai/PerceptionBench`](https://huggingface.co/datasets/moonshotai/PerceptionBench)
  at revision `6ba8c3135c7675ad6a5c141536a86b9460c70960`, file `PerceptionBench.jsonl`
  (SHA-256 `f84afb46c3c150b572481ca34b351d6afb1c31b98fef0c0c46d7659333573c47`), split `train`
  (all 3,000 items, no filtering or synthetic generation), license **CC-BY-NC-4.0** (non-commercial evaluation only).

## Measurement contract

- **Task contract & prompt:** Each task sends a single `user` message constructed by interleaving text segments
  and `input_image` blocks (`detail: "high"`, original image bytes with MIME type detected from file magic bytes)
  at the `<|image_N|>` placeholders in `problem`, matching `build_messages` in upstream `eval/eval.py`. No system
  prompt, no `hint`, and no answer-format instructions are added, and reference answers are kept strictly in
  `verifier_metadata`. No tools or sandboxes are used (`simple_agent`, `max_steps: 1`).
- **Scoring & judge:** `resources_servers/perceptionbench` strips `<think>`/`<thinking>` and Gemma `<|channel>thought...<channel|>` blocks from the policy's
  final assistant message and calls `perceptionbench_judge_model` (`/v1/chat/completions`, default
  `openai/gpt-oss-120b`, `temperature: 0.3`, `max_tokens: 4096`) with the byte-identical upstream prompt
  (`resources_servers/perceptionbench/prompt_templates/judge_prompt.txt`). The boolean verdict is parsed after
  `[judge]`, yielding binary reward `1.0` (`True`) or `0.0` (`False`).
- **Metrics:** Primary metric `mean/reward` is micro-average accuracy across all 3,000 items (upstream `Overall`).
  Per-capability accuracies are reported as `by_category/<category>/accuracy` (alongside `macro_category_accuracy`),
  per-source accuracies as `by_source/<source_bmk>/accuracy` (for example `by_source/DA-2K/accuracy`), and missing
  final answers as `mean/no_answer`.
- **Failures & timeouts:** Empty replies, `[error]` prefixes, and unclosed `<think>` blocks score `0.0`
  (`no_answer=1.0`) without calling the judge, matching upstream `is_failed`. Context-window rejections record
  `model_context_limit` and score `0.0`. Transport errors and judge replies without a valid `[judge] True|False`
  verdict after 3 attempts raise `JudgeError`, masking the row (`mask_sample: true`, `failure_kind: judge_failed`)
  rather than scoring `0.0` as upstream does. No episode-level wall-clock deadline is set; episodes are bounded by
  `max_steps: 1`, policy `max_output_tokens` (upstream uses `65536` with maximum reasoning effort), and judge
  `max_tokens: 4096`.

## Prepare

```bash
gym eval prepare --benchmark perceptionbench
```

`prepare.py` downloads `PerceptionBench.jsonl` from the pinned Hugging Face revision (or reads a local copy via
`++prepare_script_args.source_path=/path/to/PerceptionBench.jsonl` or `PERCEPTIONBENCH_DATA_ROOT`), verifies its
SHA-256 and 3,000-row count, and writes `benchmarks/perceptionbench/data/perceptionbench_benchmark.jsonl`
(gitignored).

## Validate and test

```bash
gym env validate perceptionbench
gym env test perceptionbench
gym env test --resources-server perceptionbench +should_validate_data=true
gym env publish perceptionbench
```

## Evaluate

Configure a vision-capable policy endpoint (`policy_base_url` / `policy_api_key` / `policy_model_name`) and an
OpenAI-compatible judge endpoint (`judge_base_url` / `judge_api_key` / `judge_model_name`; default
`openai/gpt-oss-120b`) in `env.yaml`, then run:

```bash
# Smoke run (5 tasks)
gym eval run --benchmark perceptionbench --model-type openai_model \
    --split benchmark --limit 5 --num-repeats 1 \
    --max-output-tokens 65536 \
    --output results/perceptionbench_smoke/rollouts.jsonl

# Full benchmark (3,000 tasks)
gym eval run --benchmark perceptionbench --model-type openai_model \
    --split benchmark --num-repeats 1 \
    --max-output-tokens 65536 \
    --output results/perceptionbench_full/rollouts.jsonl
```

## Published upstream baselines & parity validation

Upstream paper leaderboard highlights (`gpt-oss-120b` judge at `temperature=0.3`, `MAX_TOKENS=65536`, highest
reasoning budget; accuracy in %):

- GPT-5.6-Sol: `59.7%`
- Kimi K3: `58.5%`
- Claude-Fable-5: `57.2%`
- Gemini-3.1-Pro: `56.2%`
- Gemini-3.5-Flash: `52.0%`
- Qwen3.5-397B-A17B: `47.5%`
- Gemma-4-31B: `40.7%`
- GLM-4.6V: `32.5%`

Full 3,000-task parity evaluation in NeMo Gym (`vllm_model`, `num_repeats=1`, `max_output_tokens=65536`,
`gpt-oss-120b` judge at `temperature=0.3`, `3,000/3,000` completed, `0` failed, `0` masked, `0` `no_answer`):

- **Gemma-4-31B** (thinking enabled): **`40.80%`** overall vs **`40.7%`** upstream (`+0.10 pp`; per-capability:
  VRel `44.5%` vs `42.7%`, Count `33.3%` vs `33.9%`, Attr `39.4%` vs `40.3%`, Depth `40.9%` vs `39.1%`,
  Loc `45.5%` vs `44.9%`, Comp `39.8%` vs `43.7%`, FGR `37.2%` vs `39.0%`, Ctx `43.1%` vs `45.9%`,
  OCR `46.7%` vs `46.7%`, Hallu `38.4%` vs `32.1%`; `DA-2K` `31.7%`).
- **Gemini-3.5-Flash** (`reasoning_effort=high`): **`61.20%`** overall (`3,000/3,000` completed, `0` failed,
  `0` masked; live hosted alias evaluated in Nov 2026 vs `52.0%` reported in the July 2026 paper; `DA-2K`
  `46.0%`).

**Limitations:**
- Dataset license is **CC-BY-NC-4.0** (non-commercial research and evaluation only; do not include in training
  corpora).
- Multi-image support is required: ~11.6% of items contain 2–8 images interleaved via `<|image_N|>` placeholders.
- 79 `DA-2K` localization items ask about pixel coordinates while `hint` (carrying image dimensions on 116 items)
  is intentionally omitted to match upstream `eval/eval.py`; serving stacks that resize images beforehand affect
  those items (`by_source/DA-2K/accuracy` isolates them).
