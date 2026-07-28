# PutnamBench

Lean4 formal proof benchmark bound to the
[`math_formal_lean`](../../resources_servers/math_formal_lean/) resources server
and `simple_agent` (single-turn, matching NeMo-Skills' evaluation protocol).

- **Tasks**: 660 theorems (test split)
- **Source**: [`trishullab/PutnamBench`](https://github.com/trishullab/PutnamBench/tree/64cedd86ef523f3d5f5dc7a21c10e3f69564c7d4) at pinned commit `64cedd86ef523f3d5f5dc7a21c10e3f69564c7d4`. `prepare.py` clones the repo, runs the upstream `lean4/scripts/rewrite_solutions.py` to generate 660 `.lean` files with `sorry`-swapping applied, then regex-parses each.
- **Prompt**: shared `benchmarks/prompts/lean4/formal-proof-deepseek-prover-v2.yaml` (same as miniF2F, MOBench, ProofNet). Intentionally differs from NeMo-Skills' upstream choice of `lean4/formal-proof` for this benchmark.
- **Reward**: binary; 1.0 iff the Lean4 compiler accepts the proof with no `sorry`.

## Preparation

```bash
gym eval prepare --benchmark putnam_bench
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
gym env start \
    --model-type vllm_model \
    --benchmark putnam_bench
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent putnam_bench_math_formal_lean_simple_agent \
    --input benchmarks/putnam_bench/data/putnam_bench_benchmark.jsonl \
    --output results/putnam_bench_rollouts.jsonl \
    --num-repeats 32 \
    --max-output-tokens 16384 \
    --temperature 1.0 \
    --prompt-config benchmarks/prompts/lean4/formal-proof-deepseek-prover-v2.yaml
```
