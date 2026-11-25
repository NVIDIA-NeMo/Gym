# VoiceBench

[VoiceBench](https://github.com/MatthewCYM/VoiceBench) scores spoken assistants on 11 subsets. The data is the
[`hlt-lab/voicebench`](https://huggingface.co/datasets/hlt-lab/voicebench) dataset at revision
`b02edcef1330480be3a11bd6f7434ac32f05ad08`; it is downloaded by `prepare()` and not copied into the repository.

| Subset | Rows | Score |
|---|---|---|
| `alpacaeval`, `alpacaeval_full`, `alpacaeval_speaker` (11 voices), `commoneval`, `wildvoice` | 199, 636, 6,996, 200, 1,000 | GPT judge rating, 1-5 |
| `sd-qa` (11 regions) | 6,083 | GPT judge accuracy (`gpt`) and PEDANT (`panda`) |
| `openbookqa`, `mmsu` (12 categories) | 455, 3,074 | accuracy of the parsed A-D choice |
| `bbh` | 1,000 | accuracy of the parsed answer |
| `advbench` | 520 | refusal rate |
| `ifeval` | 345 | mean of prompt/instruction strict/loose accuracy |

- **License:** Apache-2.0. Both the VoiceBench source repository and the dataset card state Apache-2.0 (checked at the
  pinned revision); the scorers, their provenance and the judge prompts are listed in
  [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md).
- **Scoring:** all subsets share one JSONL and one [`voicebench`](../../resources_servers/voicebench/README.md) server in
  `subset: all` mode, which reports every subset under `voicebench/<subset>/` (per-voice, per-region and per-category
  splits, and `voicebench/<subset>/aggregation_error` for a subset that could not aggregate, are described in the server
  README). MTBench is not included: the upstream project publishes no scorer for it.

## Run

```bash
gym eval prepare --benchmark voicebench   # streams 11 GB of audio into benchmarks/voicebench/data
gym eval run --benchmark voicebench --model-type vllm_model --split benchmark --num-repeats 1 \
    --output results/voicebench_rollouts.jsonl
```

- Pass `+prepare_script_args.audio_root=/data/voicebench` when the model server sees `data/` at another path.
- Smoke runs: `+prepare_script_args.max_samples_per_profile=N` keeps only the first N rows of every split of every
  subset. A capped run reports `complete: false` for every subset, because it has fewer rows than `expected_rows`.

## Settings

- **Prompts:** each row carries the subset's system prompt from [`prompts/system_prompts.yaml`](prompts/system_prompts.yaml)
  (the prompts of NVIDIA's reference evaluation; VoiceBench defines none) and an empty user message: the question is the
  audio.
- **Sampling:** reasoning on (`reasoning_budget` 16,384), temperature 0.6, top-p 0.95, 20,480 output tokens, seed 0, baked
  into the rows. A response that reaches the token limit is `incomplete`; the `status_safe_*` rewards score it 0.
- **Answers:** `answer_extraction: message` scores the final message; a response that never produced one is scored as
  empty (which the AdvBench rule counts as a refusal).
- **Judge:** the open-ended subsets and SD-QA need a judge model server named `judge_model`; set `judge_base_url`,
  `judge_api_key` and `judge_model_name` in `env.yaml` (VoiceBench scores are defined with `gpt-4o-mini`; the server
  refuses to start while the endpoint is the placeholder). The prompts in [`prompts/judge_open.yaml`](prompts/judge_open.yaml)
  and [`prompts/judge_reference_qa.yaml`](prompts/judge_reference_qa.yaml) are the two prompts of VoiceBench's
  `api_judge.py` at the pinned commit, unchanged; the request follows that script (temperature 0.5, top-p 0.95, `n=3`,
  1,024 tokens). Both judges share one concurrency budget (`judge_max_concurrency`, 40).
- **Scorers:** the MCQ and refusal parsers carry NVIDIA answer-format extensions (`\boxed{}`, Markdown-bold answers, curly
  apostrophes), so OpenBookQA, MMSU and AdvBench can score higher than the public evaluator; see the server README.
- **Limits:** `prepare()` writes absolute audio paths into `responses_create_params.metadata`, so the model server must be
  able to read them (hence `audio_root`). Scores have not been compared with published VoiceBench numbers; the `panda`
  metric has been checked against the reference classifier on 88 SD-QA responses. The reward of a single row is provisional; the `voicebench/` aggregates are the result.
