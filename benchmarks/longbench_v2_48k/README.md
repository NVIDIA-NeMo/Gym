# LongBench v2

[LongBench v2](https://arxiv.org/abs/2412.15204) is a long-context multiple-choice
benchmark: 503 four-way questions over documents, dialogue histories, code
repositories and structured data, with contexts from 8k to 2M words. The model
answers with `The correct answer is (X)`.

Scoring lives in the [`longbench` resources server](../../resources_servers/longbench_v2_48k/README.md);
this benchmark only supplies data and wiring. Each row's reward is `1.0` when the
extracted letter equals the gold letter, so the mean reward is accuracy.

## Variants

| Variant | Config | Agent | Rows | Output |
|---|---|---|---|---|
| Full | `config.yaml` | `longbench_benchmark_simple_agent` | all 503 | `data/longbench_benchmark.jsonl` |
| 48k | `config_48k.yaml` | `longbench_48k_benchmark_simple_agent` | 151 whose prompt is under 48000 tokens | `data/longbench_48k_benchmark.jsonl` |

In the full variant, prompts over 119800 tokens keep their first and last halves
and drop the middle, so every row fits a 131072-token window. No row of the 48k
variant is truncated.

## Prepare data

Requires `transformers` and a Hugging Face token with access to the gated
`google/gemma-4-E4B-it` tokenizer, which selects the 48k subset.

```bash
gym eval prepare --benchmark longbench
gym eval prepare --benchmark longbench/config_48k
```

Either command writes both files in one pass. To truncate with the answering
model's own tokenizer, or for a different window:

```bash
gym eval prepare --benchmark longbench \
    +prepare_script_args.truncate_tokenizer=<hf tokenizer id> \
    +prepare_script_args.max_prompt_tokens=<tokens>
```

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark longbench
```

## Collecting rollouts and scoring

```bash
gym eval run --no-serve \
    --agent longbench_benchmark_simple_agent \
    --input benchmarks/longbench/data/longbench_benchmark.jsonl \
    --output results/longbench_rollouts.jsonl \
    --num-repeats 1
```

For the 48k variant, start with `--benchmark longbench/config_48k` and run
`--agent longbench_48k_benchmark_simple_agent` on
`benchmarks/longbench/data/longbench_48k_benchmark.jsonl`.
