# nl2repobench_vahid1016_v50: 50 verified held-out nl2repo tasks

A self-contained, verified subset of the vahid_1016 held-out nl2repo test split (the agent implements a whole Python project from `start.md`; reward = fraction of the
reference repo's held-out tests that pass). Everything needed is in this folder: `tasks/` (per task: `start.md`, `test_commands.json`, `test_files.json`,
`test_case_count.txt`, `extra_test_requirements.json`), `benchmark.jsonl` (50 rows), `prepare.py`, `opencode.yaml`, `openhands.yaml`, `verified_tasks_50.json`.

## Why 50 tasks ("verified")
The original 75-task split was graded in an environment where the reference code itself failed its own tests for ~half the tasks (missing test-only dependencies),
and the first grading let the reference's code fill in for anything the agent did not write. A task is in this set only if the **untouched reference code passes 100% of its
held-out tests in the per-task image** under the final grading, so every task has ceiling 1.0 and scores are directly comparable (`verified_tasks_50.json`,
`task_ceilings.json`). Measured: do-nothing agent = 0.003; full reference code through the grading path = 1.0 on all 50.
Caveat: `pythonspeed_profila_pr-1` (5 tests) is timing-sensitive and once scored 0.8 on the untouched reference.

## Grading (resources_servers/nl2repobench)
The verifier runs, in a fresh sandbox from the per-task image (every network mode, `verify_image_override` is unset): (0) reference clone + install, skipped when the image already has
`/tmp/_reference_repo/.git`; (1, 20 tasks) install that task's test-only dependencies; (2) **purge the reference implementation** (git-tracked code files, compiled extensions)
and overlay the agent's `/workspace` (excluding `tests`, `conftest.py`, `pytest.ini`, `tox.ini`, packaging files, `.pth`, ...); (3) run the held-out tests. See
`test_commands.json`.

## Images
Per-task images: `gitlab-master.nvidia.com:5005/fjia/nl2repo_images/vahid1016/<task-lowercase>:1.1` (the `:1.0` images plus test-only dependencies; Dockerfiles and build script in
`sft_runs/lightning_vahid1016/testset/task_images_v11/` of the parent repo). The sandbox provider on your cluster must be able to pull them: set
`GITLAB_REGISTRY_USERNAME` / `GITLAB_REGISTRY_PASSWORD` (read_registry deploy token) before launching; they are read by `task_image_registry_template`'s `image_auth` in the yaml. If you mirror the
images elsewhere, change `task_image_registry_template` (and the `image` field of `benchmark.jsonl`, which must equal the templated name or the server rejects the request).

## Running (two harnesses, three network modes)
Configs reuse the block names of `benchmarks/nl2repobench/opencode.yaml`, so the existing launchers/overrides apply. Parent-repo launchers: `launch_v50_opencode.sh` and
`launch_v50_openhands.sh` (`AGENT_NETWORK_MODE=<mode> CLUSTER=<name> bash launch_v50_<harness>.sh <HF_MODEL_PATH> <EXPERIMENT_NAME>`). Equivalent raw form:
```
--config benchmarks/nl2repobench_vahid1016_v50/<opencode|openhands>.yaml --config responses_api_models/vllm_model/configs/vllm_model.yaml \
++nl2repobench_opencode_resources_server.resources_servers.nl2repobench.agent_network_mode=<mode>
```
Modes (`agent_network_mode`): `no_internet_preinstalled` (recommended: agent sandbox = per-task image, deny-all egress except the model host; the reference repo and its editable-install metadata
are removed before the first turn), `block_target_v2` (generic agent image, open egress with source-hosting domains denied and a package/repo-aware pip/git/curl/python guard; see `guard_v2.py`),
`official` (no restriction; treat as an upper bound). Needs `OPENSANDBOX_DOMAIN` / `OPENSANDBOX_API_KEY` (OpenSandbox provider) in the environment.

- **opencode**: the yaml points at a pre-staged opencode binary (`remote_opencode_install_script_path` / `remote_opencode_binary_path`, `/mnt/s3-data/...` on the original cluster). Override both for your
  cluster, or set them to `null` to download opencode in the sandbox (needs internet, so not in `no_internet_preinstalled`).
- **OpenHands** (`responses_api_agents/openhands_sdk_sandboxed_agent`, OpenHands SDK 1.50): build the offline bundle once with
  `bash responses_api_agents/openhands_sdk_sandboxed_agent/build_bundle.sh <outdir>` and pass
  `++nl2repobench_opencode_sandboxed_agent.responses_api_agents.openhands_sdk_sandboxed_agent.bundle_path=<outdir>/openhands_bundle.tar.gz`. The bundle is required for `no_internet_preinstalled` and
  `block_target_v2`. The agent uses the OpenHands SFT-data user prompt by default (`user_prompt` in the agent config).
- Cluster job limits: Qwen-size rollouts are slow; if a job hits the scheduler's time limit, resume with `ROLLOUTS_FPATH=<existing output jsonl>` and `++resume_from_cache=true`.

## Reading results
Reward per row is already comparable across tasks (all ceilings 1.0). Useful extra fields: `tests_passed`, `test_case_count`, `verifier_error`, `workspace_bytes`; OpenHands rows add
`openhands_finished`, `openhands_execution_status`, `openhands_error_codes`. Guard refusals show up as `nl2repobench guard v2` text in tool outputs (`block_target_v2`).
Background, leak audit and known residual risks: `sft_runs/lightning_vahid1016/testset/LEAK_AUDIT.md` and `OPENHANDS_SETUP.md` in the parent repo.
