# Indic GSM8K

Zero-shot, thinking-enabled evaluation on
[ai4bharat/indic-gsm8k](https://huggingface.co/datasets/ai4bharat/indic-gsm8k).
The `test` split contains 1,319 questions translated into 14 languages. Preparation
downloads revision `e21f3ff7e9fe703a7f4faeefdba27801be278eb5` and verifies the file checksum.

The default evaluation uses 12 languages: `bn`, `gu`, `hi`, `kn`, `mr`, `ml`,
`ne`, `or`, `pa`, `ta`, `te`, `ur` (15,828 tasks). Use `kn` for Kannada, `mr` for
Marathi, and `ml` for Malayalam. Assamese (`as`) and Sanskrit (`sa`) are also
available through the language option.

```bash
gym eval prepare --benchmark indic/gsm8k/config

gym eval run \
  --benchmark indic/gsm8k/config \
  --output results/indic-gsm8k.jsonl \
  --split benchmark \
  --model-url http://localhost:8000/v1 \
  --model-api-key dummy \
  --model <served-model-name>
```

To select languages, add `+prepare_script_args.languages='[hi,ta,te]'` to the
prepare command. To use a downloaded copy of the same HF file, add
`+prepare_script_args.source=/path/to/test.parquet`.

The config connects Gym's environment server, `simple_agent`, the GSM8K numeric
verifier, and the vLLM model server. Each request permits 120,000 output tokens
with thinking enabled. Serve the model with a reasoning parser and sufficient
context (131,072 tokens for the evaluated setup). Set temperature, top-p and
top-k for the chosen model through Gym's generation settings.

Each prompt asks the model to solve the translated question and finish with
`#### <answer>`. Grading uses the original English numeric answer key and only
the final assistant text. Metrics report strict `####` extraction and flexible
last-number exact match, overall and by language; strict match is the reward.
The dataset is Apache-2.0 licensed. Generated JSONL files are ignored by Git.
