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
ng_prepare_benchmark "+config_paths=[benchmarks/proof-arena-judge/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/proof-arena-judge/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=proof_arena_judge_math_proof_judgement_simple_agent \
    +input_jsonl_fpath=benchmarks/proof-arena-judge/data/proof-arena-judge_benchmark.jsonl \
    +output_jsonl_fpath=results/proof-arena-judge/rollouts.jsonl \
    +prompt_config=benchmarks/prompts/math_proof_judge.yaml
```
