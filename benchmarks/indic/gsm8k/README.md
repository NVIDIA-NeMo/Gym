# Indic GSM8K

For instruct/thinking evaluation from a local Indic GSM8K Parquet file, use
`config_posttrain.yaml` and `prepare_posttrain.py`. They expand
all 1,319 source rows into 18,466 tasks across 14 languages, use zero-shot chat
prompts, retain the original English numeric answer keys, and permit 120,000
output tokens. The prompt requests `#### <answer>`; the existing verifier scores
only the final assistant text with strict and flexible numeric exact match.
Each task records the source file's SHA-256 and has a stable UUID derived from
that checksum, its source ID, and language. The Parquet schema requires `id`,
`answer`, and `question_<Language>_translation` columns; translated answers are
never used as references. The dataset is supplied separately.

```bash
python -m benchmarks.indic.gsm8k.prepare_posttrain \
  --source /path/to/indic-gsm8k/test.parquet \
  --languages bn gu hi kn mr ml ne or pa ta te ur

gym eval run \
  --benchmark indic/gsm8k/config_posttrain \
  --output results/indic-gsm8k-posttrain.jsonl \
  --split benchmark \
  --model-url http://localhost:8000/v1 \
  --model-api-key dummy \
  --model <served-model-name>
```

The default source path is `data/indic-gsm8k/test.parquet`, relative to the working
directory. Omitting `--languages` selects all 14 supported languages. The example
selects the 12-language evaluation set (Kannada `kn`, Marathi `mr`, Malayalam `ml`).
Serve an instruct model with a reasoning parser and sufficient context for the
prompt plus 120,000 output tokens; the evaluated campaign used 131,072 context
tokens. Configure each model's temperature, top-p and top-k explicitly through
Gym's `responses_create_params` and vLLM `sampling_overrides`. The chat config
enables thinking and has no benchmark-specific stop strings. Its prompt is:

> Solve the following math problem. End your response with a line of the form "#### &lt;answer&gt;", using digits for the final numerical answer.

The translated question follows after a blank line. Any model-specific system
instruction must be supplied separately and recorded with the run settings.

### Validate a saved campaign shard

`verify_posttrain.py` checks saved campaign artifacts without regenerating model
responses. It expects `data/<language>.jsonl`, `models.json` (keyed by run-model
name, with `name`, `sampling`, and optional `system_instruction`), the frozen
prompt at `source/benchmarks/indic/gsm8k/prompts/posttrain.yaml`, and
`results/<model>/<language>/{rollouts,transport}.jsonl`. Transport request events
contain `event: transport_request` and the actual HTTP `request_payload`.

```bash
python -m benchmarks.indic.gsm8k.verify_posttrain /path/to/campaign model-key kn
```

Validation checks coverage, UUIDs, prompts, references, requested and actual
sampling settings, thinking enabled, reasoning boundaries, token accounting,
and recomputed strict/flexible scores. It writes `summary.json` and marks only
successful shards with `DONE`. Unicode separators within model text are preserved
when reading JSONL. Cluster launchers and model checkpoints are managed separately.

## Historical base-model protocol

This benchmark evaluates grade-school math in the 14 translated Indic-language
configurations published in `anushakamathofficial/indic_GSM8K_v5`. The source is
pinned to commit `f778a10d5d7f4d976574c7296b2e8ce89c30a38d`.

The source provides 14 Indic languages; the referenced ai4bharat run covers 22. The prompt and scoring protocol match, while the v5 translation text is a newer corpus revision.

The integration follows the referenced lm-eval pretraining protocol:

- five examples sampled from the test split with Python `random.Random(42)`,
  excluding the current evaluation row;
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
gym eval prepare --benchmark indic/gsm8k/config

gym eval run \
  --benchmark indic/gsm8k/config \
  --output results/indic_gsm8k.jsonl \
  --split benchmark \
  --model-url http://localhost:8000/v1 \
  --model-api-key dummy \
  --model <served-model-name>
```

Use `+prepare_script_args.languages='[hi,ta,te]'` to prepare a subset.

This same-split demonstration policy follows the supplied reference; it is not
a strictly held-out test protocol. The primary metrics are `pass@1/strict_match`
and `pass@1/flexible_extract` (percentages). The rollout reward uses strict match.
No majority-vote metric is reported because the two extractors can produce
different answers for the same generation.
