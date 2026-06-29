# PinchBench

[PinchBench](https://github.com/pinchbench/skill) measures how well an LLM performs as the **brain of an
[OpenClaw](https://github.com/openclaw/openclaw) agent** across 147 real-world tasks (calendar,
email triage, CSV/log analysis, coding, research, writing). Each task is graded by deterministic
checks, an LLM judge, or both.

This is an **external-benchmark integration** wired at the agent-server level (the same shape as
`swe_agents` and `harbor_agent`): OpenClaw + PinchBench's own rollout/grading harness *is* the
orchestration, so we wrap it rather than reimplement it.

## Architecture

One **sandbox per task** is the isolation boundary, launched through Gym's provider-neutral
[Sandbox API](../../nemo_gym/sandbox) (`AsyncSandbox`). Each `/run` starts a sandbox that runs the
stock PinchBench `benchmark.py` for a *single* task through OpenClaw, tars its result + transcript,
and exits; the agent (`app.py`) downloads + parses that archive. There is no host bind-mount — the
sandbox provider (apptainer / opensandbox) is config-selected.

```
ng_collect_rollouts ── /run ──> app.py ── AsyncSandbox(provider) ──> [ sandbox ]
                                                                       /opt/run_task.sh
                                                                       ├─ OpenClaw gateway (per-task)
                                                                       ├─ benchmark.py --suite <task>
                                                                       └─ → <work_base>/out/out.tgz
                                            <── reward + grading_breakdown + grading_notes + raw_rollout
```

Per-task sandboxing gives clean isolation (each task gets its own `~/.openclaw` and its own
gateway), which is **cliff-proof**: a task that wedges its own sandbox can't affect any other task.
A gateway shared across all 147 tasks instead throws `WorkspaceVanishedError` once the harness resets
the shared workspace mid-run — the per-task design avoids that entirely.

### The skill is cloned + patched at build time (not vendored)

Following the repo convention (`harbor_agent` / `mini_swe_agent` pin a framework git commit rather
than vendoring task files), `Dockerfile.benchmark` clones PinchBench at a pinned tag and applies a
small NVIDIA integration patch:

- `git clone -b v2.0.0 https://github.com/pinchbench/skill /opt/pinchbench-skill`
- `git apply setup_scripts/nvidia-pinchbench.patch` — the NVIDIA delta (one file, `scripts/lib_agent.py`):
  a custom OpenAI-compatible provider (`custom/<model>`, `contextWindow`/`maxTokens` from env,
  `requiresStringContent`), `JUDGE_BASE_URL`/`JUDGE_API_KEY` judge routing, brave/tavily web-search
  config, and transcript-path resolution. Upstream `v2.0.0` can't reach an OpenAI-compatible endpoint
  unpatched.
- `COPY run_task.sh /opt/run_task.sh` — our per-task entrypoint.

### Config knobs (`configs/pinchbench.yaml`)

| key | meaning |
|---|---|
| `openclaw_mode` | `gateway` — the only supported mode (a per-task OpenClaw gateway daemon; at openclaw 2026.6.5 `openclaw agent` needs a gateway to persist transcripts) |
| `sandbox_provider` | provider + its config, e.g. `{apptainer: {}}` (Slurm/HPC) or `{opensandbox: {}}` (cluster) — see `nemo_gym.sandbox` |
| `sandbox_spec` | the per-task sandbox: `image` (`${sandbox_image}` — a `.sif` path or `docker://` ref built from `Dockerfile.benchmark`), `resources`, `ready_timeout_s`, … |
| `sandbox_work_base` | writable, per-sandbox-isolated working mount (default `/sandbox`); run_task.sh puts the skill copy, `$HOME`, `$TMPDIR` and the benchmark run-root here |
| `task_timeout_s` | per-task exec timeout |
| `model_base_url` / `model_api_key` / `model_name` | policy model OpenClaw runs against |
| `judge_model` / `judge_base_url` / `judge_api_key` | judge for hybrid / `llm_judge` tasks |
| `max_tokens`, `context_window`, `max_concurrent`, `timeout_multiplier` | run tuning |

> **Model wiring:** OpenClaw must point at a **streaming-capable** endpoint directly — *not* a Gym
> model server, which is non-streaming (`stream: Literal[False]`) and would 422 OpenClaw's streamed
> requests. So the policy/judge endpoints are passed straight through to OpenClaw.

## Setup

```bash
# 1. Build the per-task image (Node 22 + openclaw@2026.6.5; clones skill@v2.0.0 + applies the patch)
bash responses_api_agents/pinchbench/setup_scripts/build_image.sh             # docker image
bash responses_api_agents/pinchbench/setup_scripts/build_image.sh --apptainer # also builds pinchbench.sif

# 2. Datasets: data/example.jsonl (5 tasks) is committed. To regenerate the full 147-task set,
#    point at a skill checkout (the skill is not vendored):
git clone -b v2.0.0 https://github.com/pinchbench/skill /tmp/pb-skill
PINCHBENCH_SKILL_DIR=/tmp/pb-skill python responses_api_agents/pinchbench/dataset_preprocess.py
```

Each JSONL line references a task by `task_id`; the full task definition (prompt, grading code,
assets) is reconstructed inside the sandbox from the cloned skill at run time, so the dataset stays
tiny.

## Run

```bash
ng_run "+config_paths=[responses_api_agents/pinchbench/configs/pinchbench.yaml]" \
  +sandbox_image=<pinchbench.sif | docker://pinchbench-openclaw:latest> \
  +model_base_url=<endpoint/v1> +model_api_key=<key> +model_name=<model> \
  +judge_model=<judge> +judge_base_url=<endpoint/v1> +judge_api_key=<key> +brave_api_key=<key>

ng_collect_rollouts +agent_name=pinchbench_agent \
  +input_jsonl_fpath=responses_api_agents/pinchbench/data/example.jsonl \
  +output_jsonl_fpath=results/pinchbench.jsonl +num_samples_in_parallel=4 +num_repeats=1
```

Each rollout returns `reward` (continuous `[0,1]`), `grading_type`, `grading_breakdown`,
`grading_notes`, and `raw_rollout` (the full OpenClaw transcript, also archived to `transcripts_dir`).

## Validation (parity vs vanilla standalone)

The Gym integration was checked against vanilla PinchBench run directly from the skill (no Gym
wrapper, `benchmark.py --suite all`) on `Nemotron-3-Nano-30B-A3B`, n=3, `max_tokens=65536`,
`temperature=1.0`:

| arm | mean ± std |
|---|---|
| Gym (per-task gateway) | **0.583 ± 0.025** |
| Vanilla standalone | **0.564 ± 0.013** |

Δ mean = 0.019, within the run-to-run stochastic spread → parity. (temperature 1 + live web search
make single-run scores noisy; trust aggregates / `num_repeats`.)

## Notes & gotchas

- **OpenClaw pinned to `2026.6.5`** (brave-plugin `2026.6.5`). At `2026.6.x`, `openclaw agent` routes
  through a gateway to persist session transcripts, so `run_task.sh` starts a per-task gateway
  (`--auth token --bind loopback` + `OPENCLAW_GATEWAY_TOKEN`).
- **Reward is continuous** (hybrid = weighted automated + judge). For RL training, threshold it.
- **Reproduce-first:** baseline against the standalone harness before trusting Gym numbers.

## Licensing

- Integration code: Apache 2.0.
- PinchBench skill (cloned at build, not vendored): MIT — see `LICENSE` and the repo `ATTRIBUTIONS.md`.
  `setup_scripts/nvidia-pinchbench.patch` is the NVIDIA integration delta applied over upstream `v2.0.0`.
