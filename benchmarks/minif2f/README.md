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
gym eval prepare --benchmark minif2f
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
gym env start \
    --model-type vllm_model \
    --benchmark minif2f
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent minif2f_math_formal_lean_simple_agent \
    --input benchmarks/minif2f/data/minif2f_benchmark.jsonl \
    --output results/minif2f_rollouts.jsonl \
    --num-repeats 32 \
    --prompt-config benchmarks/prompts/lean4/formal-proof-deepseek-prover-v2.yaml \
    --temperature 1.0 \
    --max-output-tokens 16384
```

Reproduce published miniF2F numbers on a DeepSeek-Prover / Goedel-Prover class
model before treating a baseline as real; small policy models will hit ~0%.
