# PutnamBench

Lean4 formal proof benchmark bound to the
[`math_formal_lean`](../../resources_servers/math_formal_lean/) resources server
and `simple_agent` (single-turn, matching NeMo-Skills' evaluation protocol).

- **Tasks**: 660 theorems (test split)
- **Source**: [`trishullab/PutnamBench`](https://github.com/trishullab/PutnamBench/tree/64cedd86ef523f3d5f5dc7a21c10e3f69564c7d4) at pinned commit `64cedd86ef523f3d5f5dc7a21c10e3f69564c7d4`. `prepare.py` clones the repo, runs the upstream `lean4/scripts/rewrite_solutions.py` to generate 660 `.lean` files with `sorry`-swapping applied, then regex-parses each.
- **Prompt**: shared `benchmarks/prompts/lean4_formal_proof_deepseek_prover_v2.yaml` (same as miniF2F, MOBench, ProofNet). Intentionally differs from NeMo-Skills' upstream choice of `lean4/formal-proof` for this benchmark.
- **Reward**: binary; 1.0 iff the Lean4 compiler accepts the proof with no `sorry`.

## Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/putnam_bench/config.yaml]"
```

Clones the upstream repo into a temp dir, runs its `rewrite_solutions.py` subprocess, parses the output `.lean` files, and writes `data/putnam_bench_benchmark.jsonl`. Each row has `name`, `split`, `header`, `informal_prefix`, `formal_statement` (no `goal` field — the `math_formal_lean` server doesn't use it, and the upstream prepare omits it).

Network + git required: prepare clones ~100 MB and runs ~10 s of upstream Python.

## Running

Verification shells out to the NeMo-Skills Lean4 sandbox over HTTP. Bring up the
sandbox container separately (see
[`resources_servers/math_formal_lean/README.md`](../../resources_servers/math_formal_lean/README.md))
and set `NEMO_SKILLS_SANDBOX_HOST` / `NEMO_SKILLS_SANDBOX_PORT` before starting
the server.

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/putnam_bench/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=putnam_bench_math_formal_lean_simple_agent \
    +input_jsonl_fpath=benchmarks/putnam_bench/data/putnam_bench_benchmark.jsonl \
    +output_jsonl_fpath=results/putnam_bench_rollouts.jsonl \
    +num_repeats=32 \
    +prompt_config=benchmarks/prompts/lean4_formal_proof_deepseek_prover_v2.yaml \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```
