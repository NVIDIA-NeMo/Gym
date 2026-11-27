# SWE-bench Pro v2

The `swebench/pro_v2/opencode` benchmark runs the 642 tasks from
[SWE-bench Pro v2.0.0](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2).
`swebench/pro_v2_hard/opencode` runs its exact HARD-51 subset. V1 remains separately available.

Preparation checks upstream `SHA256SUMS`, the task image identity, the network-lock declaration,
642/51 unique task IDs and HARD-51 membership. It preserves `instruction.md`, verifier files and
the reference patch byte for byte, including CRLF, and resolves GHCR image tags to SHA-256 digests.
The agent and fresh verifier use that same digest. Preparation also stages a pinned static ripgrep binary.

```bash
python benchmarks/swebench/pro_v2/prepare.py
python benchmarks/swebench/pro_v2_hard/prepare.py
```

## Example input provenance

`data/example.jsonl` contains five genuine Teleport task inputs whose exact source base commits
are Apache-2.0 licensed. Each row records its task license and immutable license URL. The complete
Apache-2.0 text is retained in `data/LICENSE-TELEPORT`; all five source license files are identical.
Five inspected real-model rollouts are still required before this onboarding is ready.
The source inputs are pinned to the [HF dataset revision](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/tree/2d52cb3df914a3fcf80c7f66738b3a88ae37fc50)
and [upstream task tree](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2/tasks).

The upstream MIT license covers evaluation harness and tooling; task content and patches retain
the licenses of their source repositories. `data/LICENSE` preserves the harness's MIT notice.
Do not attribute all task content to MIT or copy incompatible task source into this Apache-2.0 repository.
Record the exact task repository/base commit, file paths, source revisions and applicable licenses for
any committed fixture. Keep input data, gold/empty verifier controls and real model rollouts distinct.
The dataset-source schema cannot express this joined HF/Git subset with both source revisions;
its exact provenance is recorded per row instead of pointing the downloader at the full HF split.

The separate `configs/swebench_pro_v2_example.yaml` selects the example fixture once present and
stages its pinned offline ripgrep tool. It leaves both full-set benchmark presets unchanged.

```bash
gym eval prepare --config resources_servers/swebench_pro_v2/configs/swebench_pro_v2_example.yaml
gym dataset collate --config resources_servers/swebench_pro_v2/configs/swebench_pro_v2_example.yaml \
  +output_dirpath=data/swebench_pro_v2 +mode=example_validation
```

Real `example_rollouts.jsonl` must come from a model run, with five rows and an inspected
agent/verifier outcome for each. Preserve the original run and config separately. Before
publishing a fixture, remove secrets and private endpoints, paths and model identifiers;
retain task IDs, public source/image pins, responses and scoring meaning.

## Locked agent and fresh verification

The default V2 preset requires OpenSandbox network-policy enforcement. `task_network_policy` allows
only the configured policy-model host; other providers fail before creating an agent sandbox. The
head server stays on loopback because the allowed model host must not expose its configuration endpoint.
Keep `absolute_ip_server_types: [responses_api_models]` when enabling `use_absolute_ip`.
Verifier sandboxes keep network access for the task's own dependency setup.

Supply network-free OpenCode installation in the deployment overlay with
`remote_opencode_install_script_path`, `remote_opencode_binary_path` and, for Alpine images,
`remote_opencode_musl_binary_path` on
`swebench_pro_v2_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent`.
The installer and binaries must already be mounted inside the sandbox and match `opencode_version`.
An image with that matching binary can instead set `preinstalled_opencode: true`.
The default online installer cannot run under this policy. The prepared ripgrep binary is uploaded by the harness.

The agent keeps the harness's **10800-second** execution budget, matching the selected V1/OpenCode
configuration. This differs from upstream's 3000-second agent budget; compare scores only after accounting
for that difference. The verifier runs `test.sh` with **3000 seconds**, enforced inside the sandbox.
The per-attempt and total verification budgets also cover sandbox setup and artifact collection.

The verifier applies the model patch in a pristine task image, runs the task's unchanged `tests/test.sh`,
and records its `reward.txt` as the authoritative binary grade. A genuine test-script timeout scores zero.
A missing verdict, provider/wrapper execution failure or artifact transport failure stays inconclusive, is retried on a fresh
sandbox and is masked as `verifier_error` when retries are exhausted. Agent sandbox loss or lost patch
capture remains a typed failure. Redis state leaked by an agent test run is removed before verification;
other V1 image repairs and Go prefetch are disabled.

Task-internal dependency or parser failures can also make upstream `test.sh` write zero. Those zeros remain
scored failures: its binary reward does not distinguish them from candidate compile/test failures. Empty test
reports and network-error text alone cannot safely trigger masking because candidate code can produce either.
Inspect verifier artifacts and independent gold-patch controls when diagnosing such failures; do not interpret
an unmasked zero as proof that infrastructure was healthy. The five example rollouts provide representative
smoke evidence, not a locked full-642/HARD-51 baseline or benchmark certification. Both presets remain unverified.

## Run and verifier controls

See the [published tutorial](https://docs.nvidia.com/nemo/gym/main/evaluation-tutorials/swe-bench-pro-v2)
for model-provider and offline-runtime setup. Both presets use three repeats per task by default.

For a gold-patch control, start the V2 resources server with the OpenSandbox provider and
`is_verifying_golden_patch=true`, then run:

```bash
python resources_servers/swebench_pro/apply_golden_patch.py \
  +benchmark_jsonl=benchmarks/swebench/data/swebench_pro_v2_benchmark.jsonl \
  +server_name=swebench_pro_v2_resources_server \
  +output_fpath=results/swebench_pro_v2_golden.jsonl
```

Keep gold-patch, empty-patch and model-driven rollout evidence separate. Inspect all failures;
a completed control is not a certificate or proof that every upstream task is healthy.
