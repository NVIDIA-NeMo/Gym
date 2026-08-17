# Proof-Arena-Judge

Adds the `proof-arena-judge` benchmark to Gym.

## Details

- Evaluation mode: binary `Judgement: Yes/No` over math proofs
- Prompt mirrors Skills' `judge/math-proof-judge`
- Verification reuses Gym's deterministic `math_proof_judgement` server
- Includes the vendored `gemini_imo_2025/*.txt` proof files used by Skills
- Pins MathArena `*_2025_outputs` to the same 2026-03-25 revisions used in
  Skills to avoid the later `grading_details_judge_*` schema change
- Preparation applies the same seed-42 shuffle and Qwen3 <=10k-token proof
  filter used in Skills

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark proof-arena-judge

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark proof-arena-judge

# Collecting rollouts
gym eval run --no-serve \
    --agent proof_arena_judge_math_proof_judgement_simple_agent \
    --input benchmarks/proof-arena-judge/data/proof-arena-judge_benchmark.jsonl \
    --output results/proof-arena-judge/rollouts.jsonl \
    --prompt-config benchmarks/prompts/judge/math-proof-judge.yaml
```
