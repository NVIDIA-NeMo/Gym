# Indic GPQA Diamond pretraining evaluation

This recipe evaluates base models with five-shot multiple-choice likelihood on
the translated Indic GPQA Diamond dataset. It was extracted from the completed
48-job base-model campaign and made path-configurable for reuse.

The 12 languages are Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi,
Nepali, Odia, Punjabi, Tamil, Telugu, and Urdu. Assamese, Sanskrit, and English
are intentionally excluded to match the corresponding post-training campaign.
Each language contains 198 questions.

## Metric and prompt

For each question, the reference lm-evaluation-harness sampler selects five
same-language examples from the train split with seed 42, excluding the current
question. Demonstrations end in `Answer: <letter>` and the evaluated prompt ends
in `Answer:`. The evaluator adds no chat template, system prompt, reasoning
prompt, or special tokens.

The score is strict accuracy from:

```text
argmax log P(choice | five-shot prompt), choice in {" A", " B", " C", " D"}
```

The preparation audit verifies that every choice is exactly one distinct token
at the prompt boundary. This is a likelihood evaluation, not sampled generation:
temperature, top-p, top-k, and a generation output-token cap do not apply. vLLM
does request one token to expose the next-token log probabilities.

Option order follows the canonical English GPQA question's MD5-seeded shuffle,
shared across translations. The answer is tracked by option identity rather than
translated string equality. The preparation step checks the translated and
canonical snapshots by `Record ID`, hashes every input, and freezes the rendered
prompts and submitted source in the run directory.

## Setup

The preparation environment needs `jinja2`, `pyarrow`, `pyyaml`, and
`tokenizers`, plus access to the reference lm-evaluation-harness sampler. The GPU
environment needs the campaign-compatible `vllm` build. Copy and edit the sample
configuration; do not commit a configuration containing private filesystem
paths.

```bash
cd benchmarks/gpqa/indic-pretrain
cp campaign.example.json campaign.local.json
python prepare.py --config campaign.local.json --run /path/to/runs/gpqa-5shot
```

`campaign.local.json` supplies the translated parquet, canonical GPQA CSV,
reference task directory, reference sampler, local checkpoints, GPU counts, and
Slurm settings. All configured checkpoints must share a tokenizer because the
prompts and continuation token IDs are frozen once per language.

## Launch

Export the container image and, if needed, the Python executable and mounts.
These variables propagate through `sbatch` to `job.sbatch`.

```bash
export GPQA_CONTAINER_IMAGE=/path/to/evaluation.sqsh
export GPQA_CONTAINER_MOUNTS=/lustre:/lustre
export GPQA_PYTHON=/path/to/python

python launch.py --run /path/to/runs/gpqa-5shot --dry-run
python launch.py --run /path/to/runs/gpqa-5shot --pilots-only
python launch.py --run /path/to/runs/gpqa-5shot
python launch.py --run /path/to/runs/gpqa-5shot --retry-failed
```

The launcher submits one job per model and language. A model's Hindi pilot gates
its other 11 jobs with `afterok`, which avoids allocating GPUs if the pilot fails.
It records every submission in `jobs.json`, refuses duplicate live jobs, and
skips completed outputs. `--retry-failed` asks Slurm accounting for terminal
failed, timed-out, cancelled, preempted, node-failed, boot-failed, and
out-of-memory jobs, then resubmits only those jobs while retaining their ledger
history. The evaluator resumes unfinished work from `scores.jsonl`; a file lock
prevents concurrent writers. `--models MODEL_KEY ...` can limit either launch
mode to selected manifest model keys.

Each job first audits five real questions. Two are independently rescored with
forced continuations. If the optimized next-token scores exceed the BF16 parity
tolerance or have an argmax-changing near tie, the entire fresh job switches to
forced-continuation scoring. The selected backend is saved with every score.
Results are flushed every 16 questions and accepted on resume only when their
data/model identity hash still matches.

Outputs are stored under `RUN/results/MODEL/LANGUAGE/`: `status.json`,
`smoke.json`, `scores.jsonl`, `metrics.json`, and `DONE`. Generated run data,
logs, environments, and model outputs are deliberately excluded from Git.
