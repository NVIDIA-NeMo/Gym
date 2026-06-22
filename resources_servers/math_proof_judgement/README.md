# Math Proof Judgement

Binary judgement of math proofs. The policy model reads a problem + candidate
proof and outputs `Judgement: Yes` (proof correct) or `Judgement: No` (proof
incorrect). Verification is fully deterministic — a single regex parses the
final judgement — so **no external LLM judge is called**. This mirrors NeMo
Skills' `answer-judgement` metric (`is_correct_judgement` format 1).

The primary consumer is the [`proof_bench_judge`](../../benchmarks/proof_bench_judge)
benchmark, which instantiates this server for the `ProofBench` corpus.

## Metrics

The `aggregate_metrics` endpoint emits Gym's standard pass@k / pass@1[avg-of-k]
/ majority@k for:

* `accuracy` — 1.0 if the parsed judgement matches gold, else 0.0
* `true_positive`, `false_positive`, `false_negative`, `true_negative` — one-hot
* `no_answer` — fraction of rollouts with an unparseable judgement

Plus binary-classification metrics, averaged across the K rollouts the same
way NeMo Skills' `AnswerJudgementMetrics` does (per-sample precision/recall/F1
then macro-averaged):

* `pass@1[avg-of-k]/{precision,recall,f1}`
* `pass@k/{precision,recall,f1}`
* `majority@k/{precision,recall,f1}`
* `total_positives` — count of rollouts whose gold judgement is Yes

## JSONL schema

Each input row must set `expected_judgement` alongside the baked-in prompt
messages. The verifier parses `Judgement: Yes/No` out of *both* `expected_judgement`
and the model's response.

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "<judge prompt>"}]},
  "expected_judgement": "Judgement: Yes",
  "problem_id": "...",
  "problem": "...",
  "proof": "..."
}
```

## Reasoning-model requirement

For reasoning models that emit `<think>…</think>` CoT (Nemotron, DeepSeek-R1,
Qwen3, …), **enable vLLM's `--reasoning-parser`** so the CoT is routed to a
separate reasoning output item and doesn't reach the Judgement regex:

```bash
vllm serve <model> --reasoning-parser deepseek_r1   # for <think>-style models
```

Without the reasoning parser, a greedy first-match regex (this server, or
Skills' `is_correct_judgement`) will pick up phrases like *"so Judgement: No
applies if …"* inside the model's reasoning and score them as the committed
verdict — materially lowering accuracy on reasoning models.

The server itself does not strip CoT; that's the vLLM server's job. Pick the
parser name that matches your model's reasoning tokens (see `vllm serve
--help` → `--reasoning-parser`).

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/math_proof_judgement/configs/math_proof_judgement.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=math_proof_judgement_simple_agent \
    +input_jsonl_fpath=resources_servers/math_proof_judgement/data/example.jsonl \
    +output_jsonl_fpath=results/math_proof_judgement_rollouts.jsonl \
    +num_repeats=1
```
