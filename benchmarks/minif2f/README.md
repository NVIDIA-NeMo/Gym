# miniF2F

Lean4 formal proof benchmark bound to the
[`math_formal_lean`](../../resources_servers/math_formal_lean/) resources server
and `simple_agent` (single-turn, no correction loop — matches NeMo-Skills'
evaluation protocol).

- **Tasks**: 244 theorems (test split)
- **Source**: [Goedel-LM/Goedel-Prover-V2 miniF2F JSONL](https://raw.githubusercontent.com/Goedel-LM/Goedel-Prover-V2/refs/heads/main/dataset/minif2f.jsonl)
- **Prompt**: ported from NeMo-Skills `nemo_skills/prompt/config/lean4/formal-proof-deepseek-prover-v2.yaml`
- **Reward**: binary; 1.0 iff the Lean4 compiler accepts the proof with no `sorry`

## Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/minif2f/config.yaml]"
```

Downloads the source JSONL, splits header from theorem body, strips `sorry`
variants, ensures each `formal_statement` ends with ` := by\n`, and writes
`data/minif2f_benchmark.jsonl`. Each row has `name`, `split`, `header`,
`informal_prefix`, `formal_statement`, `goal`.

## Running

Verification shells out to the NeMo-Skills Lean4 sandbox over HTTP. Bring up
the sandbox container separately (see
[`resources_servers/math_formal_lean/README.md`](../../resources_servers/math_formal_lean/README.md))
and set `NEMO_SKILLS_SANDBOX_HOST` / `NEMO_SKILLS_SANDBOX_PORT` before starting
the server.

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/minif2f/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=minif2f_math_formal_lean_simple_agent \
    +input_jsonl_fpath=benchmarks/minif2f/data/minif2f_benchmark.jsonl \
    +output_jsonl_fpath=results/minif2f_rollouts.jsonl \
    +num_repeats=32 \
    +prompt_config=benchmarks/prompts/lean4_formal_proof_deepseek_prover_v2.yaml \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

Reproduce published miniF2F numbers on a DeepSeek-Prover / Goedel-Prover class
model before treating a baseline as real; small policy models will hit ~0%.
