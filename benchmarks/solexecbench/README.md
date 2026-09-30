# SOL-ExecBench

`solexecbench` prepares all 235 tasks from the public
[`nvidia/SOL-ExecBench`](https://huggingface.co/datasets/nvidia/SOL-ExecBench/tree/63699402f003496acc3af4eb534a5304a8ac1ea9)
dataset at revision `63699402f003496acc3af4eb534a5304a8ac1ea9`. The four source configurations are
L1 (94), L2 (82), Quant (33), and FlashInfer-Bench (26). Every task retains its full ordered workload list:
3,957 workloads in total. No task is filtered for hardware support or difficulty.

The canonical profile asks a model for native Solution JSON containing CUDA C++ source for NVIDIA B200.
It uses Gym's `simple_agent` and the [native SOL verifier](../../resources_servers/sol_execbench/README.md),
which runs pinned SOL-ExecBench revision `a9fa0804c793d438e70850c33fe34426e66d53dd` in an isolated GPU
sandbox. The standalone preparer also supports Triton prompts using the same trusted problems.
The scalar reward is complete-workload correctness; native workload latencies are retained as metrics.
Official SOL score remains unavailable: the pinned dataset has no timing anchors, and the published
leaderboard timing tables omit workload UUIDs, leaving duplicate-axis workloads ambiguous. The adapter
does not infer those missing identities or report partial official scores.

This integration remains experimental; CPU tests and successful preparation do not establish GPU
correctness or reproduce published benchmark results.

## Prepare

From the Gym repository root:

```bash
gym eval prepare --benchmark solexecbench
gym env validate solexecbench
```

Preparation downloads the 304 referenced safetensors files (39,697,148 bytes) from
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/tree/4ee6fc905cdef5ef6b941b73ff4a220c92aec470)
at revision `4ee6fc905cdef5ef6b941b73ff4a220c92aec470`, the immutable commit behind its upstream `1.0` tag.
It writes a trusted problem manifest, its SHA-256 sidecar, the complete benchmark JSONL, and a representative
example consisting of the first L1 task in name order. All downloaded and generated data are ignored by Git.

Dataset `verifier_metadata` contains only `task_id` and `problem_digest`. Definitions, complete workloads,
and asset path/hash allowlists come from the server-owned manifest. The digest preserves input/output
argument order. A changed definition, workload order, or asset content changes the identity. Dataset rows
cannot supply evaluator commands or arbitrary input paths.

The default is one independent response per task. If you change the agent dataset's `num_repeats`,
set the resources server's `samples_per_task` to match and record the override when comparing results. The generated prompts are Gym's integration prompts, not a claim to reproduce an
upstream model's unpublished prompt or generation settings.

The canonical Gym configuration uses CUDA C++ and literal paths so its manifest can be validated
before launching services. The preparer also supports alternative languages and output directories:

```bash
python -m benchmarks.solexecbench.prepare --language triton --output-dir /path/to/data
```

To evaluate that selection, inherit the canonical configuration in your local workload config and update
the agent's dataset paths, the resources server's `problem_manifest_path`, and `prepare_script_args`
(`language` and `output_dir`) together. Read the SHA-256 sidecar from the selected directory.

## Evaluate

Configure an immutable evaluator image, the OpenSandbox GPU allocation, artifact directory, and a Gym
model endpoint as described in the [verifier setup](../../resources_servers/sol_execbench/README.md).
Set `solexecbench_sandbox_image` to your image digest in local Gym configuration. After preparation,
supply the manifest SHA-256 from the generated sidecar and run:

```bash
gym eval run --benchmark solexecbench --model-type openai_model \
  --split benchmark --concurrency 1 --output /path/to/results/solexecbench/rollouts.jsonl \
  +solexecbench_manifest_sha256="$(cat benchmarks/solexecbench/data/problem_manifest.sha256)"
```

Use one evaluator per assigned GPU and preserve the image digest, observed GPU, source revisions,
configuration, model version, sampling settings, and complete verifier artifacts with reported results.
Artifacts default to the ignored data directory's `artifacts/` subdirectory.
The native runtime must be validated on the deployed hardware before benchmark comparisons.

## Licensing

The integration source code is Apache-2.0. The SOL-ExecBench dataset is separately licensed under the
[NVIDIA Evaluation Dataset License](https://huggingface.co/datasets/nvidia/SOL-ExecBench/blob/63699402f003496acc3af4eb534a5304a8ac1ea9/LICENSE)
(`LicenseRef-NVIDIA-Evaluation`). Its terms restrict use to internal evaluation and benchmarking,
prohibit training, and prohibit redistribution of the dataset in whole or part. Review those terms before
downloading. No upstream task content or generated dataset is distributed in this package.
FlashInfer Trace assets are separately Apache-2.0; preparation downloads them directly from their pinned source.
