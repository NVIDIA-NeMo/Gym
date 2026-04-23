# ProofBench (Judge)

[ProofBench](https://huggingface.co/datasets/wenjiema02/ProofBench) framed as
a **binary judgement task**: the policy model reads a mathematical problem +
a candidate proof and outputs `Judgement: Yes` (proof is correct) or
`Judgement: No` (proof is incorrect). Verification is fully deterministic —
no external LLM judge is called. The gold `Judgement: Yes/No` label is
derived from the upstream expert rating (>= 6 / 7 → Yes).

The policy's judgement is parsed via the same regex NeMo Skills uses
(`is_correct_judgement` format 1), so the two pipelines agree on what counts
as an invalid response.

## Prerequisites

The prepare script downloads from HuggingFace and uses the `Qwen/Qwen3-0.6B`
tokenizer to apply the Skills <=10k-token filter on proof length. Ensure
`HF_TOKEN` / network access is available.

For reasoning models, **serve with `--reasoning-parser`** so `<think>…</think>`
CoT is split out at the server layer and the Judgement regex only sees the
model's committed answer:

```bash
vllm serve <model> --reasoning-parser deepseek_r1
```

Skipping the parser for a reasoning model drags pass@1 down ~10 pp on this
benchmark — the regex picks up `Judgement: X` phrases inside CoT
speculation. See `resources_servers/math_proof_judgement/README.md` for
details.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/proof_bench_judge/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/proof_bench_judge/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (4 rollouts per task)
ng_collect_rollouts \
    +agent_name=proof_bench_judge_math_proof_judgement_simple_agent \
    +input_jsonl_fpath=benchmarks/proof_bench_judge/data/proof_bench_judge_benchmark.jsonl \
    +output_jsonl_fpath=results/proof_bench_judge_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

Reported via the `math_proof_judgement` server's `compute_metrics()`:

* `pass@1[avg-of-k]/{accuracy,precision,recall,f1,no_answer}` — primary parity target
* `pass@k/{accuracy,precision,recall,f1}` — best-of-K selection
* `majority@k/{accuracy,precision,recall,f1}` — majority vote
* `total_positives` — count of gold-Yes rollouts in the batch

## Upstream source

* [wenjiema02/ProofBench](https://huggingface.co/datasets/wenjiema02/ProofBench) (`train` split)
* Skills counterpart: `nemo_skills/dataset/proof-bench-judge`
