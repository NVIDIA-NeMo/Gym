# compute_eval

NVIDIA [compute-eval](https://github.com/NVIDIA/compute-eval) benchmark
verifier. Compiles model-generated CUDA / C / C++ / Python kernel solutions
with **nvcc** and runs them against the benchmark's hidden test suite via
`compute_eval.execution.evaluate_solutions(eval_mode="local")`.

## Requirements

- CUDA Toolkit (`nvcc` on PATH) — checked at server startup.
- `compute-eval` Python package — pinned in `requirements.txt` to the
  same commit Skills' `core/requirements.txt` uses.

## Verification

For each rollout the server:

1. Validates the per-task `problem` payload as a `CudaCppProblem` or
   `CudaPythonProblem` (discriminated by the `type` field).
2. Calls `compute_eval.generate_completions._parse_solution` to extract
   fenced code blocks from `response.output_text` into a multi-file
   `FileSolution`.
3. Runs `evaluate_solutions(problem, [solution], eval_mode="local")` in a
   worker thread, bounded by `num_processes` (default 8).
4. Returns `reward=1.0` iff the resulting `GradedSolution.passed` is true.

The Skills evaluator at HEAD (`nemo_skills/evaluation/evaluator/compute_eval.py`)
has a bug introduced by PR #1315: it calls the new plural API
`evaluate_solutions(...)` but treats the return as a single object
(`graded.passed`), which silently returns `passed=False` for every problem.
This server implements the corrected `graded_list[0].passed` path.

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/compute_eval/configs/compute_eval.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=compute_eval_simple_agent \
    +input_jsonl_fpath=resources_servers/compute_eval/data/example.jsonl \
    +output_jsonl_fpath=results/compute_eval_rollouts.jsonl \
    +num_repeats=1
```

For reasoning models that emit `<think>…</think>`, start vLLM with
`--reasoning-parser <name>` (e.g. `deepseek_r1` for Nemotron-3-Nano) so
the parser strips reasoning at the model-output layer on both Skills
and Gym sides.
