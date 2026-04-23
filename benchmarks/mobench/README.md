# MOBench

Lean4 formal proof benchmark bound to the
[`math_formal_lean`](../../resources_servers/math_formal_lean/) resources server
and `simple_agent` (single-turn, matching NeMo-Skills' evaluation protocol).

- **Tasks**: 360 theorems (test split, all entries)
- **Source**: [Goedel-LM/Goedel-Prover-V2 MOBench JSONL](https://github.com/Goedel-LM/Goedel-Prover-V2/blob/2e9036e118464aa96a8bebaf9f5b9d091aa3585c/dataset/MOBench.jsonl), pinned to commit `2e9036e118464aa96a8bebaf9f5b9d091aa3585c`.
- **Prompt**: shared `benchmarks/prompts/lean4_formal_proof_deepseek_prover_v2.yaml` (same as miniF2F and ProofNet).
- **Reward**: binary; 1.0 iff the Lean4 compiler accepts the proof with no `sorry`.

## Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mobench/config.yaml]"
```

Downloads the source JSONL, splits the prelude from the theorem block via regex,
normalizes each `formal_statement` to end with ` := by\n`, attaches a fixed
Mathlib/Aesop header, and writes `data/mobench_benchmark.jsonl`. Each row has
`name`, `split`, `header`, `informal_prefix`, `formal_statement`, `goal`.

## Running

Verification shells out to the NeMo-Skills Lean4 sandbox over HTTP. Bring up the
sandbox container separately (see
[`resources_servers/math_formal_lean/README.md`](../../resources_servers/math_formal_lean/README.md))
and set `NEMO_SKILLS_SANDBOX_HOST` / `NEMO_SKILLS_SANDBOX_PORT` before starting
the server.

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/mobench/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=mobench_math_formal_lean_simple_agent \
    +input_jsonl_fpath=benchmarks/mobench/data/mobench_benchmark.jsonl \
    +output_jsonl_fpath=results/mobench_rollouts.jsonl \
    +num_repeats=32 \
    +prompt_config=benchmarks/prompts/lean4_formal_proof_deepseek_prover_v2.yaml \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

Reproduce published MOBench numbers on a DeepSeek-Prover / Goedel-Prover class
model before treating a baseline as real.
