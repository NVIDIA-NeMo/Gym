# BEAM resources server

[BEAM](https://arxiv.org/abs/2510.27246) ("Beyond a Million Tokens") probes
long-term conversational memory: a model receives a long multi-session chat
(100K / 500K / 1M tokens) and answers probing questions covering ten abilities:
abstention, contradiction resolution, event ordering, information extraction,
instruction following, knowledge update, multi-session reasoning, preference
following, summarization and temporal reasoning.

This server ports the upstream evaluation
([`src/evaluation/compute_metrics.py`](https://github.com/mohammadtavakoli78/BEAM)):

| Ability | Row reward |
|---|---|
| all except event ordering | mean over rubric items of the judge score from `unified_llm_judge_base_prompt` (pass / fail by default; 0 / 0.5 / 1 with `judge_scale: partial`) |
| `event_ordering` | `tau_norm = (tau_b + 1) / 2`: answer lines are aligned to the reference events with YES/NO LLM equivalence calls, then Kendall tau-b is computed over the union |

Config switches (defaults first): `judge_scale: binary | partial` (`partial` is
upstream's 0 / 0.5 / 1 judge) and `question_scoring: mean | all_pass` (`all_pass`
scores a question 1 only when every rubric item passes; for event ordering the
aligned events must also be in reference order). Every row carries `score_mean` and
`score_all_pass`.

Rewards are continuous in [0, 1], as upstream reports them. Deviations from upstream
(think-block stripping, empty answers short-circuiting to 0, NaN tau mapped to 0,
invalid judge scores counted as 0) are listed in the `app.py` docstring.

## Metrics

`compute_metrics` reports `<ability>/score`, `overall` (macro average over the ten
abilities, the paper's headline, under the configured `question_scoring`), the same
per `chat_size/<100K|500K|1M>/`, and `mean/...` / `all_pass/...` for both scorings.
With repeated rollouts it also reports `pass@1[avg-of-k]/overall` and
`pass@1[avg-of-k]/all_pass/overall` for every k (each question scores the mean of
its first k rollouts, then abilities are macro-averaged), plus
`pass@1[avg-of-k]/overall/std_dev_across_runs` / `std_err_across_runs` at the
largest k. The BEAM benchmark configs run 4 rollouts (`pass@1[avg-of-4]`).
A judge call that still fails after the judge model server's retries raises
`JudgeError`: the row is masked, written to `<output>_failures.jsonl`, and left
out of every metric; `gym eval reverify --judge-failed-only` re-judges it.

## Data

Rows are built by [`benchmarks/beam/prepare.py`](../../benchmarks/beam/prepare.py):
one row per probing question. By default the whole chat is rendered as one `User: ... /
Assistant: ...` transcript followed by the question, with no length filter; `--prompt_format
multi_turn` replays it as user/assistant messages, and the benchmark variants drop questions whose
full prompt exceeds a token limit (see [`benchmarks/beam/README.md`](../../benchmarks/beam/README.md)).
`data/example.jsonl` holds 5 smoke-test rows whose
history was cut to the last ~4k tokens to keep the file small; it is not a
benchmark split.

## Judge

The default judge is GPT-5.4 (`azure/openai/gpt-5.4`) on the NVIDIA inference API at
temperature 0; only the API key must be passed:

```bash
gym env start --config resources_servers/beam/configs/beam.yaml --model-type vllm_model \
    +judge_api_key=$INFERENCE_API_KEY
```

Upstream judges with `gpt-4.1-mini`; to reproduce it, add
`+judge_model_name=us/azure/openai/gpt-4.1-mini`. Scores from the two judges are not directly
comparable.

## License

Dataset: CC BY-SA 4.0 ([`Mohammadta/BEAM`](https://huggingface.co/datasets/Mohammadta/BEAM)).
Judge prompts are ported from the upstream code (MIT).
