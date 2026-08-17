# Litmus-Bench v0.1

[Litmus-Bench v0.1](https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.1)
evaluates direct-answer molecular-property reasoning on 482 held-out questions.
It uses the reusable [`litmus_agent`](../../resources_servers/litmus_agent/README.md)
verifier; see that documentation for answer extraction and scoring behavior.

The pinned v0.1 test split contains no tool-use questions. Its prompts already
encode either boxed or double-parentheses answers through `use_box_format`, so
the benchmark applies no additional prompt template. The matching Hugging Face
train split is registered by the `litmus_agent` environment configuration.

## Prepare

```bash
gym eval prepare --benchmark litmus-bench
```

This downloads the pinned test release, validates its 482 rows, removes the
obsolete source `agent_ref`, and writes the gitignored artifact to
`benchmarks/litmus-bench/data/litmus-bench_benchmark.jsonl`.

## Evaluate

```bash
gym eval run \
  --benchmark litmus-bench \
  --model-type vllm_model \
  --split benchmark \
  --output results/litmus-bench.jsonl
```

The benchmark config uses five rollouts per question. Override `--num-repeats`
only when intentionally changing the evaluation protocol. It sets
`max_output_tokens` to 131,072 as an upper bound for long reasoning traces; the
effective generation budget is also limited by the model context window minus
the prompt. Use `--max-output-tokens` to select a smaller budget for a run.

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
