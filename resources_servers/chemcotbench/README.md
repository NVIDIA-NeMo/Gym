# ChemCoTBench-V2

For preparation and evaluation through `gym eval`, use the [ChemCoTBench-V2 benchmark integration](../../benchmarks/chemcotbench/README.md).

Gym integration of [ChemCoTBench-V2](https://github.com/fresnellll/ChemCoTBench-V2). The integration contains 5,219 tasks across 29 subtasks: 900 molecular edits, 1,800 reaction predictions, 1,499 molecular-understanding questions, and 1,020 molecular optimizations.

## Prompts and data

Both system and user messages are preserved from the upstream `PromptBuilder` output. The requested reasoning templates are part of the benchmark. Reference states and gold answers are stored only in `verifier_metadata.upstream_record`.

## Automatic benchmark preparation

Use `gym eval prepare --benchmark chemcotbench` from the Gym checkout.
The benchmark folder now downloads and converts its pinned upstream sources.
No external evaluator installation, converted local input, or manual data copying is required.
See the [benchmark README](../../benchmarks/chemcotbench/README.md) for the current workflow.

Molecular optimization source values are recomputed using the original property functions in the same isolated runtime used for scoring. This avoids upstream's false `0.0` defaults. All six single-property and six dual-property tasks are included.

SMILES equivalence uses the upstream per-record system prompt. Input recovery matches bare `SMILES(...)` expressions, excluding `CANONICAL_SMILES(...)` intermediate outputs. Record `mol_und.smiles_equivalent.0097` is excluded because its truncated trace lacks the second input molecule. The 200 reaction-template and 200 mechanism-selection questions remain excluded because the release omits their answer-option texts. This is 5,219 of the full 5,620 records; scores should be reported with this coverage.

## Scoring

The resources server invokes the upstream parser and three layer evaluators in a bounded pool of reusable subprocesses:

- **Reward / Layer 1:** binary final-answer correctness, using verdict precedence (`layer1_top1_acc`, `layer1_exact_match`, then `layer1_top1_acc_strict`). Optimization uses upstream `layer1_outcome`: positive improvement for single-property tasks and the upstream per-property improvement thresholds for dual-property tasks.
- **Yield and temperature predictions:** Layer 1 is 1 only when the absolute error is strictly less than 0.5 in the dataset's numeric units; an error of exactly 0.5 fails. MAE remains available separately.
- **Condition ranking:** Layer 1 checks only the first-ranked option against the reference first option. The rest of the ranking affects the separate NDCG/MRR diagnostics, not the binary reward.
- **Layer 2:** reasoning-template adherence and self-consistency, retained as `layer2_state_score`.
- **Layer 3 Type I:** chemical intermediate-state verification, retained as `layer3_type1`.
- **Layer 3 Type II:** reference-state field matching, retained as `layer3_type2`, with matched/total counts.

Optimization has a different upstream evaluator interface: its Layer 2 score is retained in `layer2_state_score`, its reference-reasoning score in `layer3_step_score`, and its deltas, threshold results, similarity, and detailed layer metrics in `optimization_metrics`. The unrelated Layer 3 Type I/II fields remain null for optimization. Predictions are parsed separately from reference fields so missing outputs cannot inherit gold intermediate states.

Subtask-specific MAE, Tanimoto, FTS, NDCG, and MRR values are retained when upstream supplies them. Missing metrics remain null; unrelated subtask metrics are not substituted for each other. A correct final answer need not pass every Layer 3 check, including for some released reference traces.

Empty, unparseable, and incorrect model answers receive zero and remain unmasked. Think/thinking blocks are stripped before parsing. Scorer timeouts, worker failures, and invalid scorer results return a placeholder zero with `mask_sample: true`, retaining `scoring_error` and `failure_reason`. Gym excludes these unavailable measurements from quality metrics and reports them in masked-rollout coverage. No extra answer-format fallback is added to the upstream parser.

## Setup and run

On first startup the server downloads its pinned source checkout and, when Layer 3 is enabled, the released reference data. Code revision: `dcd35470de4096a1b10ee9ed6f072bcee983a9cc`. Dataset revision: `f0bb2fb00c97cb3257294a639e28f960f2da157e`. Existing task families use Gym's RDKit and scikit-learn. Optimization uses an isolated Python 3.11 environment with PyTDC 0.4.1, RDKit 2022.9.5, NumPy 1.26.4, and scikit-learn 1.2.2 (see `requirements-molopt.txt`), because the legacy pretrained oracles are incompatible with Gym's Python 3.13 dependency set. With `enable_molopt: true` (default), startup uses `uv` to install that environment under `.molopt/`, downloads the three PyTDC pretrained models there, and checks that each model can score a molecule. Successful setup is cached; it does not change Gym's installed packages.

To reuse existing files, set `repo_path` and `data_dir` in the `chemcotbench.resources_servers.chemcotbench` config. The data directory must contain `manifest.json`. Other verifier options are `run_layer3` (default true), `max_concurrency` (4), and `timeout_seconds` (120). Disabling Layer 3 leaves its scores null. Set `molopt_python` and `oracle_dir` to reuse an existing compatible optimization runtime and oracle cache. `enable_molopt: false` skips its setup for runs limited to other families; submitted optimization tasks then return `scoring_error: molopt_disabled` with `mask_sample: true`.

Scoring keeps up to `max_concurrency` worker processes across both Python runtimes.
Workers start lazily and retain imported chemistry modules and Layer 3 reference
evaluators, keyed by task family, subtask, and reference-data path. An idle worker
with the required runtime is reused; if all slots belong to the other runtime, an
idle process is replaced. Requests are scored sequentially within each worker.
Timeouts, cancellations, crashes, and malformed JSON discard that worker; the next
request starts a replacement. Server shutdown reaps all workers. Direct Python
callers must `await server.close()` when finished.

`gym eval run --concurrency 512` controls rollouts, not scoring workers. The default
four scoring workers bound CPU and memory usage independently. Increase the scorer
limit to match available CPU cores and RAM, for example with
`++chemcotbench.resources_servers.chemcotbench.max_concurrency=8` on a benchmark run
(or on `gym env start` when using `--no-serve`). Each worker retains its own caches.
The 120-second timeout includes worker startup and scoring, but excludes time
waiting for a free worker. This worker reuse changes execution only; upstream
scoring rules and the reported metric fields are unchanged.

Use the model YAML shown in the [benchmark README](../../benchmarks/chemcotbench/README.md).
The first command runs servers in the foreground; run the second in another
terminal with the same Gym environment active.

```bash
gym env start --config benchmarks/chemcotbench/config.yaml \
    --model-type vllm_model --config /absolute/path/to/model.yaml

gym eval run --no-serve --agent chemcotbench_simple_agent \
    --input resources_servers/chemcotbench/data/example.jsonl \
    --output results/chemcotbench/example_rollouts.jsonl \
    --num-repeats 1 --concurrency 4 --max-output-tokens 16384 --temperature 0
```

The complete prepared input is `benchmarks/chemcotbench/data/test.jsonl`. Separate family files are not emitted; filter `verifier_metadata.task_family` when needed. For a native Responses endpoint, select `openai_model` with a matching model config; remove the `vllm_model` block from the example YAML.

## Validation and licensing

`gym env test --resources-server chemcotbench` exercises pass/fail reference traces, runtime diagnostics, timeout/cancellation cleanup, and schema checks. `tests/conftest.py` prepares the pinned evaluator, reference data, isolated MolOpt runtime, and oracle files in `pytest_configure`, before test collection. Missing artifacts are downloaded automatically; existing installations and validated caches are reused. Tests using upstream chemistry scoring require RDKit; optional `CHEMCOTBENCH_TEST_REPO` and `CHEMCOTBENCH_TEST_DATA` paths reuse a local checkout and snapshot. `CHEMCOTBENCH_TEST_MOLOPT_PYTHON` and `CHEMCOTBENCH_TEST_ORACLE_DIR` reuse a compatible optimization environment and cache. `verified: false` remains until model baselining and review.

Integration: Apache-2.0. Upstream code, prompts, and dataset: MIT; see [LICENSE-ChemCoTBench](LICENSE-ChemCoTBench). RDKit and scikit-learn: BSD-3-Clause. PyTDC: MIT. Gym: Apache-2.0.
