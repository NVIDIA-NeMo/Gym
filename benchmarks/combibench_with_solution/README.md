# CombiBench — "with solution" setting

The paper's second setting: every published answer is already substituted into
its statement (`sols.card = ((1716) : ℕ )` instead of
`sols.card = hackmath_1_solution`), so the model only has to prove the theorem.
Everything else — prompt, verifier, Lean server, pinned upstream revision — is
shared with [`benchmarks/combibench`](../combibench/), which documents them.

Gym allows one benchmark dataset per workload, which is why this setting is a
separate benchmark rather than a second dataset there.

- **Tasks**: 100 (the `with_solution/*_sol.lean` files at the pinned revision).
- **Reward**: binary; the statement-tamper check applies as in the main
  benchmark, and no answer check is appended because the statements declare no
  `abbrev ..._solution`.

```bash
gym eval prepare --benchmark combibench_with_solution
COMBIBENCH_LEAN_SERVER_URL=http://127.0.0.1:12332 gym env start --model-type vllm_model --benchmark combibench_with_solution
gym eval run --no-serve \
    --agent combibench_with_solution_agent \
    --input benchmarks/combibench_with_solution/data/combibench_test_with_solution.jsonl \
    --output results/combibench_with_solution_rollouts.jsonl \
    --num-repeats 16 \
    --prompt-config benchmarks/combibench/prompt.yaml
```

Against Mathlib v4.24.0 all 100 substituted statements compile with only
`sorry` warnings (`resources_servers/combibench/data/harness_validation_github_test_with_solution.json`);
the Hugging Face copy of this split has the same six non-compiling statements as
its `test` split.
