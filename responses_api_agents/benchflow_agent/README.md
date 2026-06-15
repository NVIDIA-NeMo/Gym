# BenchFlow Agent for NeMo Gym

This agent integrates [BenchFlow](https://github.com/benchflow-ai/benchflow) into NeMo Gym to run
agentic benchmarks (initially [SkillsBench](https://github.com/benchflow-ai/skillsbench)).

Unlike a typical NeMo Gym agent, BenchFlow does **not** implement its own agent — it installs an
existing harness (e.g. [OpenHands](https://github.com/OpenHands/OpenHands)) into a sandbox and lets
that harness drive the model and execute commands. This server is a thin wrapper around BenchFlow's
Python `Evaluation` API: each NeMo Gym `/run` request runs **exactly one** benchmark task in a
**Singularity/Apptainer** sandbox, extracts the scalar reward, and returns a NeMo Gym response.

This integration is **evaluation-only** (scalar reward + a best-effort trajectory; no training
token-ids/logprobs) and **Singularity-only** (no Docker/Daytona).

## How it works

1. A `/run` request carries an `instance_id` of the form `<dataset_alias>::<task_name>` (e.g.
   `skillsbench::3d-scan-calc`). The task name must exist under the configured `tasks_dir`.
2. The agent resolves the NeMo Gym model server (`model_server`) to an OpenAI-compatible base URL and
   forwards it to the in-container harness via `BENCHFLOW_PROVIDER_BASE_URL` /
   `BENCHFLOW_PROVIDER_API_KEY` (OpenHands reads these). The model is passed as `hosted_vllm/<model>`.
3. The agent builds a single-task `EvaluationConfig` (`environment="singularity"`, `include_tasks={task}`)
   and `await`s `Evaluation(...).run()`, capturing the task's `RolloutResult` via an `on_result` callback.
4. The reward is read from `RolloutResult.rewards["reward"]`; the ACP trajectory is converted into NeMo
   Gym output items; BenchFlow's full artifacts/logs are written under `jobs_dir`.

## Dependencies

`requirements.txt` installs the BenchFlow fork that adds Singularity support and a general
`task_config_overrides` hook (see below). Pin it to the commit SHA that includes those changes.
Apptainer/Singularity must be installed and on `PATH`, and (with prebuilt images) the `.sif` files must
be present on the host.

## Configuration

See [`configs/benchflow_agent.yaml`](configs/benchflow_agent.yaml). Key fields:

- `tasks_dir` — local directory of BenchFlow task definitions (e.g. a cloned SkillsBench `tasks/` dir).
- `images_dir` — directory of prebuilt `.sif` images. When set, each task's `docker_image` is overridden
  to `<images_dir>/<task>.sif` (per-task, since each `/run` handles one task). Leave `null` to use the
  `docker_image` declared in each task's `task.toml`.
- `task_config_overrides` — a mapping deep-merged into each task's parsed `task.toml`, mirroring its
  section structure (e.g. `environment.memory_mb`, `agent.timeout_sec`). Extend freely for any field; no
  `task.toml` files are mutated on disk. **Requires a BenchFlow build with `task_config_overrides`.**
- `agent` (default `openhands`), `sandbox_user` (`none` = root), `agent_idle_timeout`, `skills_dir`
  (`auto`), `skill_nudge`, `usage_tracking`, `provider_api_key`, `max_retries` — mirror the BenchFlow
  CLI knobs.

This agent is normally driven by a benchmark definition under `benchmarks/` (e.g.
`benchmarks/skillsbench/`) that supplies `tasks_dir`/`images_dir`/`task_config_overrides` and a JSONL of
one row per task. See that benchmark's README for the end-to-end `ng_prepare_benchmark` →
`ng_run` → `ng_collect_rollouts` workflow.
