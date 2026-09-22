# Indic GSM8K

This benchmark evaluates grade-school math in the 14 translated Indic-language
configurations published in `anushakamathofficial/indic_GSM8K_v5`. The source is
pinned to commit `f778a10d5d7f4d976574c7296b2e8ce89c30a38d`.

The source provides 14 Indic languages; the referenced ai4bharat run covers 22. The prompt and scoring protocol match, while the v5 translation text is a newer corpus revision.

The integration follows the referenced lm-eval pretraining protocol:

- five examples sampled from the test split with Python `random.Random(42)`;
- `Question: ...\nAnswer:` formatting with a single space before demo answers
  and two real newlines between examples;
- raw vLLM text completions with no chat template;
- greedy generation, a 4,096-token output limit, and stop strings `Question:`,
  `</s>`, and `<|im_end|>`;
- strict `#### <number>` and last-number flexible exact-match metrics.

The reference base-model profiles use a 16,384-token context window.

The source dataset is currently private. Preparation requires a Hugging Face
token with access to the repository. Generated rows are ignored by Git.

```bash
gym eval prepare --benchmark indic/gsm8k

gym eval run \
  --benchmark indic/gsm8k \
  --output results/indic_gsm8k.jsonl \
  --split benchmark \
  --model-url http://localhost:8000/v1 \
  --model-api-key dummy \
  --model <served-model-name>
```

Use `+prepare_script_args.languages='[hi,ta,te]'` to prepare a subset.
