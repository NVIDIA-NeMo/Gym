# BEAM

[BEAM](https://arxiv.org/abs/2510.27246) (Beyond a Million Tokens): 90 long
multi-session chats (20 x 100K, 35 x 500K, 35 x 1M tokens) with 1,800 probing
questions over ten memory abilities, scored by an LLM judge against per-question
rubrics. Scoring lives in the [`beam`](../../resources_servers/beam) resources server;
this directory supplies data and wiring.

## Default harness

| Part | Default | Upstream BEAM |
|---|---|---|
| Prompt (`prepare.py --prompt_format`) | `single_turn`: the chat as a `User: ... / Assistant: ...` transcript in one user message, then the bare question | `multi_turn`: chat replayed as user/assistant messages |
| Answer-only NOTE (`--answer_only_note`) | off | on (`NOTE: Only provide the answer without any explanations.`) |
| Judge (`judge_scale`) | `binary`: each rubric item pass / fail | `partial`: 0 / 0.5 / 1 |
| Question score (`question_scoring`) | `mean` over rubric items (event ordering: Kendall tau) | `mean` |

Upstream reproduction: `prepare.py --prompt_format multi_turn --answer_only_note`
plus `judge_scale: partial`. `all_pass` (a question counts only if every rubric
item passes) is always reported as `all_pass/overall` next to `mean/overall`.

Muse-Glimmer-30B on the 400 questions of the 100K split (full context, 4
attempts, GPT-5.4 judge): the default harness scores 0.640, close to the 65.1
in the Muse Glimmer report. Same rollouts under other choices: upstream partial
judge 0.743; multi-turn prompt 0.611; `all_pass` 0.454 (measured before `all_pass`
also required event-ordering answers to list the events in reference order).

## Variants

| Variant | Config | Prepare script | Tokenizer | Max prompt tokens | Questions | Output |
|---|---|---|---|---|---|---|
| Default | `config.yaml` | `prepare.py` | `cl100k_base` | none | 1800 | `data/beam_benchmark.jsonl` |
| Muse 128k | `config_muse_128k.yaml` | `prepare_muse_128k.py` | `meta-models/Muse-Glimmer-30B` | 131072 | 260 (100K split only) | `data/beam_muse_128k_benchmark.jsonl` |
| Nemotron-3 1M | `config_n3_1m.yaml` | `prepare_n3_1m.py` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 1048576 | 1540 (440 of 700 from the 1M split) | `data/beam_n3_1m_benchmark.jsonl` |

A question is dropped when its full prompt (whole chat plus question) is longer
than `max_context_tokens`; nothing is truncated. With a HuggingFace tokenizer the
length matches `apply_chat_template` exactly, including the template's default
system prompt. Under the Muse tokenizer the "100K" chats run 93K-179K tokens, so
260 of their 400 questions fit 131072; every 500K / 1M question is dropped.
Serve the policy with a window of at least the limit plus the max output tokens.

Build a variant for another model or limit (omit `--max_context_tokens` to keep
every question, e.g. the full 400-question 100K split with `--splits 100K`):

```bash
python benchmarks/beam/prepare.py \
    --tokenizer_name Qwen/Qwen3-30B-A3B \
    --max_context_tokens 262144 \
    --output_fpath benchmarks/beam/data/beam_qwen3_256k_benchmark.jsonl
```

## Example usage

```bash
gym eval prepare --benchmark beam/config_muse_128k

gym env start --model-type vllm_model --benchmark beam/config_muse_128k \
    +judge_base_url=https://inference-api.nvidia.com/v1 \
    +judge_api_key=$INFERENCE_API_KEY \
    +judge_model_name=azure/openai/gpt-5.4

gym eval run --no-serve \
    --agent beam_muse_128k_benchmark_simple_agent \
    --input benchmarks/beam/data/beam_muse_128k_benchmark.jsonl \
    --output results/beam_muse_128k_rollouts.jsonl \
    --max-output-tokens 16384 \
    --num-repeats 4
```

Serve the policy with a window of at least the variant's prompt limit plus the max
output tokens (e.g. vLLM `--max-model-len 147456` = 131072 + 16384 for Muse 128k).

The Muse 128k runs are graded by GPT-5.4 (`azure/openai/gpt-5.4`, temperature 0); upstream
used `gpt-4.1-mini`, so absolute numbers are not directly comparable to the paper.

Every benchmark config sets `num_repeats: 4`, so each question is answered 4
times when the dataset is collated (`gym eval run --benchmark ...` /
`++split=benchmark`); keep `--num-repeats` at 1 there, or it multiplies on top.
`--input` skips collation, so pass `--num-repeats 4` yourself (as above).

Headline: `pass@1[avg-of-4]/overall` (each question scores the mean of its 4
rollouts, then the ten abilities are macro-averaged; equal to `overall` at 4
rollouts), with `pass@1[avg-of-4]/all_pass/overall` and
`pass@1[avg-of-4]/overall/std_dev_across_runs`, plus `pass@4/overall` (best of
the 4 rollouts). Also reported: `mean/overall`, `all_pass/overall` and
`chat_size/<size>/overall`.

## Training use

The default event-ordering reward, `tau_norm`, matches upstream for evaluation but
is easy to game as an RL reward: naming only the first event or answering "I don't
know." scores well above listing every event in reverse. For RL, set
`question_scoring: all_pass` (every rubric item passes and the events are in
reference order), or reward event ordering with `event_ordering.final_score`
(`tau_norm * f1`).

## License

Dataset: CC BY-SA 4.0 — [`Mohammadta/BEAM`](https://huggingface.co/datasets/Mohammadta/BEAM).
