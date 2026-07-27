# SWE Agents

A unified Responses-API wrapper that runs LLM-driven agents against real-world software-engineering benchmarks (SWE-bench and friends), executes the proposed patch inside the dataset's evaluation harness, and returns trajectories + a binary "resolved" reward suitable for both evaluation and RL training.

The entrypoint is [`app.py`](app.py), which exposes a `SWEBenchWrapper` (a `SimpleResponsesAPIAgent`) over HTTP. Each `responses` request takes one dataset instance, runs an agent inside an Apptainer container, runs the matching evaluation harness in a second container, and returns the trajectory plus reward.

The wrapper supports two agent harnesses, selected by the `agent_framework` config field:

- **`openhands`** (default) — runs [nv-OpenHands](https://github.com/sdevare-nv/nv-OpenHands), with a custom NeMo-Gym-aware LLM client baked into the fork.
- **`opencode`** — runs [opencode](https://opencode.ai)'s real `processor.ts` agentic loop, with a custom `LanguageModelV3` provider (`@opencode-ai/nemo-gym`) that swaps the LLM transport while keeping opencode's tools, prompts, and parsing intact. See [opencode integration](#opencode-integration) below.

---

## Table of Contents

- [Architecture at a glance](#architecture-at-a-glance)
- [Agent flow per instance](#agent-flow-per-instance)
- [Supported datasets and harnesses](#supported-datasets-and-harnesses)
- [OpenHands integration](#openhands-integration)
- [opencode integration](#opencode-integration)
- [Replay rollouts (OpenHands)](#replay-rollouts-openhands)
- [Prompt and agent-class diversity](#prompt-and-agent-class-diversity)
- [Tool-name diversity](#tool-name-diversity)
- [Configuration reference](#configuration-reference)
- [Quick Start](#quick-start)
- [Batch evaluation / data collection](#batch-evaluation--data-collection)
- [Output format](#output-format)
- [GRPO masking and failure modes](#grpo-masking-and-failure-modes)
- [Memory watchdog (OOM handling)](#memory-watchdog-oom-handling)
- [Debug / profiling](#debug--profiling)

---

## Architecture at a glance

```
                 ┌──────────────────────────────────────────────┐
   client ──▶    │  SWEBenchWrapper  (Responses API server)     │
   (one          │   • _setup_params: build per-instance config │
    instance)    │     ↳ switches OpenHandsHarnessProcessor /   │
                 │       OpenCodeHarnessProcessor on            │
                 │       cfg.agent_framework                    │
                 │   • _build_apptainer_command: bind mounts    │
                 │   • runner_ray_remote: runs on Ray worker    │
                 └────────────────┬─────────────────────────────┘
                                  │ Ray
                                  ▼
                 ┌──────────────────────────────────────────────┐
                 │ RunOpenHandsAgent.process_single_datapoint   │
                 │   (used for both harnesses; the in-SIF       │
                 │   command differs by agent_framework)        │
                 │                                              │
                 │   spawn agent container ──┐                  │
                 │                           │ in parallel      │
                 │   spawn eval container ───┘ (waits for       │
                 │                              prediction.jsonl)│
                 │                                              │
                 │   wait agent → copy patch to /trajectories   │
                 │   wait eval  → produce report.json           │
                 │   postprocess (per-dataset)                  │
                 └──────────────────────────────────────────────┘
```

Two Apptainer containers are launched concurrently per instance:

1. **Agent container** — runs the selected agent harness (OpenHands or opencode) inside the dataset's task SIF (the one that contains the repo at the right base commit). The agent edits files and writes a unified diff.
2. **Eval container** — also a SIF for the same instance. It busy-waits on the predictions file written by the agent, then runs the dataset's local evaluation harness against the patch.

The two containers are launched at the same time so the eval container's spin-up cost (often tens of seconds for SWE-bench's harness) is hidden behind the agent's run time. The eval container blocks on `until [ -f <predictions> ]; do sleep 5; done` until the agent finishes.

Concurrency across instances is bounded by `concurrency` (default 256) via an asyncio semaphore on the server, and Ray's `SPREAD` scheduling distributes per-instance workers across the cluster.

---

## Agent flow per instance

Implemented in `RunOpenHandsAgent.process_single_datapoint` (and `_run_golden_patch_verification` for the verify-only path):

1. **Setup params** (`SWEBenchWrapper._setup_params`)
   - Pick a per-instance `persistent_dir` under `swebench_results_<run_session_id>/<instance_id>_<timestamp>_<uuid>`. This is bind-mounted into both containers as `/trajectories_mount`.
   - Resolve the SIF for this instance (`_find_container`) — supports exact match, `__` → `_1776_` / `_s_` rewrites, and fuzzy `*<id>*.sif` glob, plus dataset-specific rules for `SWE-rebench` and `R2E-Gym`.
   - Write the single-row dataset JSONL (the original `instance_dict`) to a per-instance file and mount it as `/root/dataset/data.jsonl` so OpenHands does not call the HF dataset API at run time.
   - Pick the dataset-specific `BaseDatasetHarnessProcessor` (see below).
   - Resolve any prompt/agent-class override (see [Prompt and agent-class diversity](#prompt-and-agent-class-diversity)).
   - Build both Apptainer command strings.
2. **Spawn agent + eval containers** in parallel via `asyncio.create_subprocess_shell`, each streaming logs to `<persistent_dir>/apptainer_logs/<instance_id>_{agent,eval}.log`.
3. **Wait for the agent** to finish or hit `swebench_agent_timeout` (default 45 min). Copy the produced `output.jsonl` and the most recent `llm_completions/*.json` back from OpenHands' eval-output dir into the per-instance trajectories root.
4. **Extract the patch** from `out_dict["test_result"]["git_patch"]` and rewrite it in SWE-bench-prediction format at `output_for_eval.jsonl`. This file is what the eval container is waiting for.
5. **Wait for the eval container** to finish or hit `swebench_tests_timeout` (default 30 min). It produces a `report.json` whose location is dataset-specific.
6. **Postprocess** the report (per-dataset; e.g. SWE-rebench / NV-internal / SWE-bench-Ext do their parsing host-side because the eval images may not have python3).
7. **Decide `mask_sample`** (GRPO) — see [GRPO masking and failure modes](#grpo-masking-and-failure-modes).
8. **Build the response** — convert the OpenHands chat-completions trajectory to Responses-API items via `VLLMConverter`, attach the tool list, return reward = `1.0 if resolved else 0.0` and metrics.

If `verify_golden_patch=true`, step 2–4 are skipped: the dataset's golden patch (`instance_dict["patch"]`) is written directly as the prediction and the eval container is the only thing that runs. This is a sanity check that a dataset sample's golden patch actually resolves under our local eval. Supported for `swe-bench-ext`, the SWE-bench / SWE-bench_Multilingual families (e.g. `princeton-nlp/SWE-bench_Verified`, `SWE-bench/SWE-bench_Multilingual`), and `SWE-rebench`. See [Golden-patch validation](#golden-patch-validation).

---

## Supported datasets and harnesses

Selection is driven by `problem_info["dataset_name"]` (set in the input JSONL). Each dataset is paired with a `BaseDatasetHarnessProcessor` subclass that knows how to (a) install the harness once, (b) build the in-container eval command, (c) postprocess the resulting report.

| `dataset_name` (substring match)         | Processor class                          | Setup script (one-time)                         | Eval harness                                                                                              |
|------------------------------------------|------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| anything not matched below (default) — e.g. `princeton-nlp/SWE-bench_Verified`, `SWE-bench/SWE-bench_Lite`, `SWE-Gym`, `SWE-bench-Live` | `SweBenchDatasetProcessor` | `setup_scripts/swebench.sh` (HeyyyyyyG fork)   | `swebench.harness.run_local_evaluation` against the mounted JSONL                                          |
| contains `SWE-bench_Multilingual`        | `SweBenchMultilingualDatasetProcessor`   | `setup_scripts/swebench_multilingual.sh` (Kipok fork) | Same harness as SWE-bench but built from the multilingual fork                                       |
| contains `R2E-Gym`                       | `R2EGymDatasetProcessor`                 | `setup_scripts/r2e_gym.sh` (sdevare-nv fork)   | `r2egym.agenthub.run.run_local_evaluation`                                                                 |
| contains `SWE-rebench` (e.g. `SWE-rebench-V2`) | `SWERebenchDatasetProcessor`       | `setup_scripts/swe_rebench.sh` (V2)            | In-container `git apply` + `test_cmd`; **host-side** parsing via SWE-rebench's `log_parsers`               |
| `swe-bench-ext`                          | `SweBenchExtDatasetProcessor`            | none (uses `swe_bench_ext` helper module)      | Run framework-specific test command via `lighthouse`-style flags; host-side parsing via `parse_and_check_tests` |
| `nv-internal-1`                          | `NVInternalDatasetProcessor`             | none                                            | Synthesizes an env+`run_script.sh`+`parsing_script.py` from the instance's docker `ENV` lines and `before_repo_set_cmd`; tests gated by `f2p ⊆ passed ∧ p2p ⊆ passed` |
| `deepswe` ⚠️ **WIP**                     | `DeepSWEDatasetProcessor`                | none (uses Harbor task format inline)           | Harbor `tests/test.sh` verifier baked into the SIF; reads `1`/`0` from `/logs/verifier/reward.txt`. Config: [`swebench_deepswe.yaml`](configs/swebench_deepswe.yaml). |
| `denovoswe` ⚠️ **WIP**                   | `DeNovoSWEDatasetProcessor`              | none (uses bundled `_denovoswe_clean.sh` + in-container `_denovoswe_eval.py`) | Wipes the original source via `clean.sh`, re-injects spec as `README.md`, applies agent patch, lays canonical tests from `test_patch`, runs per-file pytest with `--collect-only` pre-flight; reward = 1 iff every `passed_ptp` test passes. Config: [`swebench_denovoswe.yaml`](configs/swebench_denovoswe.yaml). |

`SWE-Gym` and `SWE-bench-Live` don't get their own processor — they fall into the default `SweBenchDatasetProcessor` row — but both still get dataset-name-specific runtime treatment elsewhere: `SWE-Gym` takes a different conda-activation branch in the opencode harness (see [Dataset-aware environment activation](#opencode-integration)), and `SWE-bench-Live` resolves its OpenHands workspace directory name directly from `instance_id` instead of `{repo}__{version}` (`_resolve_swebench_workspace_dir_name`).

> [!WARNING]
> **`deepswe` and `denovoswe` are work-in-progress and not fully verified.** Golden-patch validation has been run end-to-end but full agent-driven evaluation hasn't been baselined against a reference model yet:
>
> - `deepswe` (Harbor task format) — 110/113 golden-resolved; the three remaining failures are dataset-side broken baselines mitigated via `KNOWN_BASELINE_FIX` in `deepswe_dataprocessor.py`. See [`memory/deepswe-integration.md`](../../docs/memory/deepswe-integration.md) (internal note).
> - `denovoswe` (AweAI-Team/DeNovoSWE doc-to-repo) — 3034/3668 (82.69%) golden-resolved. ~17% of golden failures are dataset noise (brittle `passed_ptp` tests) and ~2% are catastrophic collection blowups (large test suites where one drifted parametrize-label kills the batch). The two known mitigations from AweAgent's reference (`--collect-only` pre-flight + intersect; `failed_ptp` AST removal) are partially implemented — the parser is still permissive about which IDs it captures, so a marginal v1↔v2 regression rate (~0.4%) exists. See `_denovoswe_eval.py` and `DeNovoSWEDatasetProcessor.get_run_command`. The empty-patch false-pass bug (an agent timeout that produced no patch was previously falsely graded against the image's original source) is **fixed** as of `IS_GOLDEN` gating; agent-mode runs that produce empty patches now correctly score 0.
>
> Treat reward signals from `deepswe` / `denovoswe` as directional until a full agent-driven baseline is reproduced and matches published numbers.

All harnesses are installed lazily on first use and locked across nodes with `mkdir`-based cross-node locks (`file_lock`) — atomic on Lustre/NFS where `fcntl.flock` is not. Stale locks older than 1h are auto-broken.

The eval container is built from the same SIF as the agent for some datasets, but for SWE-bench / R2E-Gym / SWE-rebench / SWE-bench-Multilingual the harness venv is mounted at both `/{dataset}_setup` *and* its host absolute path (because `uv venv` bakes absolute paths into its wrappers).

---

## OpenHands integration

The agent always runs inside [OpenHands](https://github.com/All-Hands-AI/OpenHands), via `OpenHandsHarnessProcessor`. The fork pinned by `agent_framework_repo` / `agent_framework_commit` adds the prompt/tool-name diversity hooks documented below.

Key details:

- **Setup is one-time per workspace.** `setup_scripts/openhands.sh` clones the configured fork at the configured commit and bootstraps a miniforge3 + poetry venv at `swe_openhands_setup/OpenHands/`. The setup directory is mounted read-only into every agent container.
- **Commit sync on startup.** On every server start, if the existing checkout's `HEAD` differs from `agent_framework_commit`, the wrapper hard-resets the working tree to that commit (`git fetch --all --tags --prune && git reset --hard HEAD && git clean -fd && git checkout --force <commit>`). The YAML is treated as the golden truth — local edits in `swe_openhands_setup/OpenHands/` are discarded (gitignored paths like `.venv` are preserved). Bump `agent_framework_commit` in the config to roll the fork forward.
- **Inside the container** the agent driver is `OpenHands/evaluation/benchmarks/swe_bench/scripts/run_infer.sh`, parametrized by `agent_cls`, `agent_max_turns`, dataset name + split, and the per-instance dataset JSONL.
- **LLM config** is generated per-run by reading `configs/oh_config.toml`, overriding `llm.model.{model,base_url,temperature,top_p}` from the request, dumping it back as a TOML string, and writing it to `/tmp/config_<run_id>.toml` inside the container.
- **NeMo-Gym wiring** — `NEMO_GYM_METRICS_FPATH`, `NEMO_GYM_CONFIG_DICT`, and `NEMO_GYM_MODEL_SERVER_NAME` are exported into the agent container so the OpenHands fork can call back into the model server registered with this NeMo-Gym instance instead of using a raw HTTP base URL.
- **Workspace safety** — for datasets where the SIF does *not* bake `/workspace`, the agent script aborts if `/workspace` is mounted, because OpenHands' default behaviour deletes everything in `/workspace`. This check is intentionally skipped for `SWE-rebench*`, `nv-internal-1`, and `swe-bench-ext` whose images legitimately use `/workspace` or the agent works in `/{repo_name}` / `/app`.
- **`cryptography<43` shim** — for `nv-internal-1` and `swe-bench-ext`, a `cryptography<43` wheel is installed into a temp dir and prepended to `PYTHONPATH`. This works around openssl/cryptography ABI mismatches in older base images.
- **R2E-Gym test hiding** — when running the *agent* (not eval) under R2E-Gym, the wrapper deletes `/r2e_tests` and `/run_tests.sh` from `/`, `/root`, and `/testbed` so the agent can't peek at the held-out tests.
- **Trajectories** — after the agent finishes, the wrapper copies `output.jsonl` and the latest `llm_completions/*/*.json` out of OpenHands' per-run eval output directory and into `<persistent_dir>/trajectories/<instance_id>/`, then deletes the OpenHands-side dir to keep the shared setup tree clean.

---

## opencode integration

When `agent_framework: opencode`, the agent container runs [opencode](https://opencode.ai)'s real `processor.ts` agentic loop instead of OpenHands. The design rule is **swap the transport, keep the loop**: opencode's tools (`bash`, `edit`, `read`, `glob`, `grep`, `write`, `apply_patch`), prompts, tool-call parsing, and event machinery are unchanged. Only the LLM transport is swapped for a NeMo-Gym-aware provider that threads RL token IDs through.

### What runs where

```
gym _setup_params  ──▶  OpenCodeHarnessProcessor.get_run_command   (host side)
                            │
                            ▼  apptainer exec
   inside SIF:  evaluation/benchmarks/swe_bench/scripts/run_infer.sh
                            │
                            ▼
                    bun packages/opencode/src/bench/cli.ts  …  (subprocess)
                            │  spawns
                            ▼
                    bun packages/opencode/src/index.ts run "<task>"
                            │  loads .opencode/opencode.jsonc → registers
                            │  the `nemo-gym` provider, the `swe-bench`
                            │  agent, sets compaction.auto = false
                            ▼
                    opencode's real processor.ts loop   ← unchanged
                            │  every model call goes through
                            ▼
                    NemoGymLanguageModel  (LanguageModelV3, doStream)
                            │  POST /v1/chat/completions (non-streaming)
                            │  → synthesizes a single-shot stream of
                            │    text-start/delta/end + tool-input-* +
                            │    tool-call + finish parts
                            │  → dumps llm_completions/<id>/<turn>.json
                            ▼
                    NeMo Gym vllm model server
```

### Key files (in the [nv-opencode fork](https://github.com/sdevare-nv/nv-opencode))

- `packages/opencode/src/provider/sdk/nemo-gym/language-model.ts` — `LanguageModelV3` impl. Strips token-ID fields from older assistant messages on input (mirrors `nemo_gym_client.py:85-97`); captures `prompt_token_ids` / `generation_token_ids` / `generation_log_probs` from the response and threads them into `providerMetadata.nemo-gym`.
- `packages/opencode/src/provider/sdk/nemo-gym/index.ts` — `createNemoGym(opts)` factory.
- `packages/opencode/src/provider/provider.ts` — registers `@opencode-ai/nemo-gym` in `BUNDLED_PROVIDERS` so opencode's normal config path picks it up.
- `packages/opencode/src/bench/cli.ts` — per-instance bench driver. Writes a temporary `.opencode/opencode.jsonc` (registers the `nemo-gym` provider with the right `baseURL`, `completionsDir`, `instanceId`; defines a `swe-bench` agent with the SWE-bench tool subset; sets `compaction.auto: false`), then spawns `bun .../src/index.ts run "<task>" --agent swe-bench --model nemo-gym/<model> --format json` against the workspace dir. On exit it captures `git diff` and writes `output.jsonl` in the openhands-compatible shape.
- `evaluation/benchmarks/swe_bench/scripts/run_infer.sh` — in-SIF entry script invoked by the gym harness.

### Token-ID guarantees (RL contract)

The bench loop is non-streaming end-to-end. Per turn:

1. **Input dedup** — assistant messages from earlier turns have `prompt_token_ids` / `generation_token_ids` / `generation_log_probs` stripped before being sent. Only the most recent assistant message keeps those fields, so the server can verify continuity.
2. **Capture** — the response's `choices[0].message.{prompt_token_ids,generation_token_ids,generation_log_probs}` are extracted and put on `providerMetadata.nemo-gym`.
3. **Persist before tool dispatch** — the provider writes `<output_dir>/<instance_id>/bench_run/llm_completions/<instance_id>/<turn>.json` *before* the synthesized stream emits its `finish` part, so a downstream tool crash cannot lose this turn's token IDs.
4. **Contiguity invariant** — for any two consecutive turns, `turn[N+1].response.prompt_token_ids` should prefix-match `turn[N].response.prompt_token_ids ++ turn[N].response.generation_token_ids` (with tool-result tokens appended in between as the server tokenizes them). This is the regression check for "contiguous and unbroken".

### Trajectory format

Per-turn JSON files match the openhands shape exactly:

```jsonc
{
  "messages":  [ /* full message history at this turn */ ],
  "response":  { /* OpenAI ChatCompletion JSON */ },
  "provider_specific_fields": {
    "prompt_token_ids":     [...],
    "generation_token_ids": [...],
    "generation_log_probs": [...]
  },
  "kwargs":   { "tools": [...], "model": "...", ... },
  "timestamp": 1715000000.0
}
```

Final `output.jsonl` matches what `RunOpenHandsAgent.process_single_datapoint` already reads:

```jsonc
{
  "instance_id": "...",
  "test_result": { "git_patch": "diff --git ..." },
  "metadata":    { "llm_config": { "model": "..." } },
  "metrics":     { "bench_run_time": 412.3, "opencode_exit_code": 0 },
  "error":       null
}
```

This means gym's existing `get_openhands_trajectory_from_completions` and patch-extraction logic work without any changes when `agent_framework: opencode`.

### Setup

`OpenCodeHarnessProcessor.setup` runs once per workspace. It:

1. Installs Bun locally at `swe_opencode_setup/bun/` (idempotent; tries the official installer first, falls back to a direct release tarball).
2. `git clone $agent_framework_repo $opencode_dir` and `git checkout $agent_framework_commit`.
3. `bun install --frozen-lockfile` inside the cloned tree.

Both `swe_opencode_setup/opencode/` and `swe_opencode_setup/bun/` are bind-mounted read-only into every agent container at `/opencode_setup/opencode` and `/opencode_setup/bun`.

At container start the wrapper exports `OPENCODE_DISABLE_MODELS_FETCH=1` and seeds an empty `/root/.cache/opencode/models.json` so opencode does not try to fetch the public models catalog over the network — the SIFs typically run with no outbound DNS and the `nemo-gym` provider supplies its own model metadata.

### Prompt overrides

The same `agent_prompt_overrides` / `agent_prompt_override_random` selection machinery described in [Prompt and agent-class diversity](#prompt-and-agent-class-diversity) drives opencode too, but the template format and mount points differ from OpenHands:

- Templates are **flat, single-file, plain-text** (`prompts/opencode_harness/*.txt`), rendered with Python `str.format(workspace_path=..., problem_statement=...)` — not Jinja `.j2` — since `_render_opencode_user_message` does a plain `.format()` call. A malformed template (bad/missing `{...}` placeholder) falls back to the built-in default at `prompts/opencode_harness/user_prompt.txt` and logs a warning rather than failing the rollout.
- Only `user_prompt_template` is required; `system_prompt_template` is optional and typically left unset for opencode (methodology is folded into the user message instead, since opencode's `swe-bench` agent definition already carries its own default system prompt inside the fork's `.opencode/opencode.jsonc`).
- The resolved user template is rendered host-side and mounted read-only at `/opencode_setup/opencode/user_message.txt` (passed to `run_infer.sh` as positional arg `$11`). If `resolved_system_prompt_template` is set, it's mounted at `/opencode_setup/opencode/system_prompt.txt` and passed as the optional `$12` — omitting the arg entirely when unset, so opencode's own default system prompt applies.
- `agent_cls` on the override is informational only for opencode (opencode has no OpenHands-style agent-class switch); `diversify_tool_names` / `camel_case_tool_names` are not implemented in the opencode path.

See `configs/swebench_opencode_multi_tools.yaml` for a working example (11-way user-prompt bundle), and `configs/swebench_opencode_no_instruction.yaml` / `configs/swebench_opencode_empty.yaml` for ablation baselines.

### Dataset-aware environment activation

Before launching opencode, the agent script activates the dataset's Python/venv environment in the parent shell (so `PATH` / `VIRTUAL_ENV` / `CONDA_DEFAULT_ENV` propagate down through `run_infer.sh` → `bun` → opencode's `bash` tool):

| `dataset_name`                                    | Activation                                                              |
|----------------------------------------------------|--------------------------------------------------------------------------|
| contains `SWE-Gym`                                  | Deactivates any venv, then `conda activate testbed` under `/opt/miniconda3`. |
| contains `R2E-Gym`                                  | Deactivates any venv, then sources `/testbed/.venv/bin/activate`.        |
| `nv-internal-1`, `swe-bench-ext`, or contains `SWE-rebench-V2` | No-op — these SIFs already have the right interpreter on `PATH`.  |
| default (SWE-bench Verified/Lite/Multilingual/Live) | `conda activate testbed` under `/opt/miniconda3`.                       |

Each branch is wrapped in `|| true` so a missing/misconfigured env cannot kill the rollout.

For `denovoswe`, the agent script also wipes the original source and re-injects the spec as `README.md` *before* opencode starts (`_denovoswe_clean.sh`, folded into the baseline commit via `git commit --amend`) — mirroring the eval-side clean in `DeNovoSWEDatasetProcessor`. Without this the agent could just read the pre-existing source it's supposed to be regenerating.

### Subagents

When `opencode_subagents_enabled: true`, the wrapper exports `ENABLE_SUBAGENTS=1` so opencode's `task` tool is available and the main agent can spawn subagent sessions. Each session (main and subagent) writes its own per-turn trajectory files, tagged with `session_id` / `parent_session_id`, under `llm_completions/<instance_id>/*.json`.

- `_openhands_dir_copy_from_host` groups completion files by `session_id` and copies only the most recent turn per session back to the host (that file's `messages` carries the full cumulative history for the session).
- `get_openhands_trajectory_from_completions` (used to build the API response's `output`) selects the **main** session — the one with no `parent_session_id` — falling back to the last file on disk for payloads that predate session tagging (the OpenHands path).
- `get_all_session_trajectories_from_completions` returns every session found on disk. `SWEBenchWrapper._inner_responses` filters this to sessions that *do* have a `parent_session_id` (i.e. subagent sessions only) and surfaces them as `subagent_trajectories` in the response metadata / `run()` output — see [Output format](#output-format).

### Termination

There is **no explicit `finish` tool** in the opencode bench path — by design. The trajectory ends via one of four signals:

1. **Natural idle.** The model emits an assistant turn with no tool calls. opencode's `processor.ts` finishes that step, the session transitions to `status.idle`, the `opencode run` subprocess exits 0, our `bench/cli.ts` `finally` block runs `git diff` against the workspace, and writes `output.jsonl`. This is the same idle signal opencode's user-facing `cli/cmd/run.ts` uses (`if event.type === "session.status" && status.type === "idle" → break`).
2. **Max-turns hit.** The agent config sets `steps: agent_max_turns`. When opencode exceeds it the session errors → idle → subprocess exits (often non-zero). `finally` still captures `git diff`; `output.jsonl.error` becomes `"opencode_exit_<N>"`.
3. **Crash / unhandled error inside opencode.** Same as max-turns — `finally` captures the patch, `error` is set, partial `llm_completions/<id>/*.json` files written by the provider before the crash are still on disk.
4. **SIF wall-clock timeout** — the gym `timeout --signal=TERM --kill-after=30 ...` wrapper used by the openhands path applies here too; `_openhands_dir_copy_from_host` glob-copies whatever completions landed.

**Trade-off vs OpenHands.** OpenHands' `CodeActAgent` ships a `finish` tool that the model is trained to call when it's done; the agent stops on that call rather than running to max-turns. opencode has no such tool, so SWE-bench-tuned models that learned the openhands finish-semantic may keep iterating until they hit max-turns even after the issue is fixed. The accepted mitigation for now is to set a generous `agent_max_turns`, lean on a strong system prompt ("stop emitting tool calls when the issue is resolved"), and rely on `git diff` capturing the (already-correct) patch regardless of whether the model gracefully stopped or hit the wall. Adding a finish-tool plugin to the nv-opencode fork is a follow-up if the trained-finish gap meaningfully hurts pass-rate.

### Failure handling

- **Per-turn HTTP / parse errors** — the provider re-raises; prior turns are already persisted.
- **Tool errors** — caught by opencode's processor.ts as usual; appended as tool-result messages.
- **Loop crash / opencode subprocess exit ≠ 0** — covered by the *Termination* section above.
- **SIF wall-clock timeout** — covered by the *Termination* section above.

### Compaction (deferred)

`compaction.auto: false` is hard-coded into the bench config so the loop runs without summarization for now — the user-facing requirement is "main agent loop in opencode without compaction" while keeping the design flexible to enable it later via opencode's existing `compaction.ts`. Token-ID continuity over a compaction event is a future RL-design problem.

---

## Replay rollouts (OpenHands)

The OpenHands harness supports resuming a partial trajectory: the request's `body.input` can carry a prior agent run (system + user + zero or more `function_call` / `function_call_output` Responses-API items), and the agent will continue from that point instead of starting fresh. This is what powers training-time replay and trajectory branching.

**How it's plumbed.** In `SWEBenchWrapper._setup_params`:

1. `_maybe_build_replay_messages(body)` scans `body.input` for any `function_call` / `function_call_output` items. If none are present, the request is treated as a normal seed and replay support is a no-op. If any are present, the input is converted to OpenAI chat-completion message format via the shared `VLLMConverter` and stashed on `problem_info["replay_messages"]` as a JSON string (metadata is typed `Dict[str, str]`).
2. `OpenHandsHarnessProcessor.get_run_command` writes the JSON to `<persistent_dir>/replay_messages.json`, bind-mounts it into the agent container, and forwards the mounted path as positional arg #18 (`REPLAY_MESSAGES_PATH`) to `run_infer.sh`. Positional args 13..17 are emitted as empty-string placeholders so the right shift index lands.
3. The replay's own system message is extracted and written to `<persistent_dir>/replay_system_prompt.j2`, then pinned as `resolved_system_prompt_template` so the standard mount logic bind-mounts it over OpenHands' `system_prompt.j2`. This is **unconditional** — it overrides any `agent_prompt_overrides` selection. Reason: OpenHands' `replay_utils.messages_to_replay_events()` drops system messages, so without this the resumed run would render OpenHands' own `system_prompt.j2` and drift from the recorded conversation.

**Input/output split in the response.** When a replay prefix is present, `_inner_responses` echoes `body.input` back as the response's `input` verbatim and isolates only the genuinely new live-continuation messages as `output`. The boundary is found by matching tool-call ids: every replayed action's `call_id` appears in `body.input`, so the live continuation begins immediately after the last chat message that references one of those ids. Non-replay requests still use the original `split_responses_input_output_items` path.

**`body.model` is optional in replay mode.** Replay JSONLs intentionally omit `model` because the upstream `openai_model` proxy force-overrides to its configured backend. The wrapper coerces `body.model or ""` when writing `oh_config.toml`, and falls back to the agent's `model_server.name` when constructing `NeMoGymResponse.model`.

Replay is not implemented for the `opencode` harness; `_maybe_build_replay_messages` short-circuits unless `agent_framework == "openhands"`.

---

## Prompt and agent-class diversity

OpenHands ships several agent classes; this wrapper supports four:

| `agent_cls`       | Notes                                                                |
|-------------------|----------------------------------------------------------------------|
| `CodeActAgent`    | Default. CodeAct-style react-loop with bash/edit tools.              |
| `OpenCodeAgent`   | OpenCode-style agent.                                                |
| `CodexAgent`      | Codex-style agent.                                                   |
| `Terminus2Agent`  | Terminus 2 agent.                                                    |

On top of agent class, each instance can be run with a different **system / user prompt** chosen from a list of overrides. The `prompts/` directory ships 15 prompt families:

```
prompts/
├── breadth_first/        ├── incremental/         ├── plan_and_execute/
├── codex/                ├── minimalist/          ├── root_cause/
├── divide_and_conquer/   ├── opencode/            ├── surgical/
├── explore_plan_execute/ ├── openhands/           ├── terminus/
├── hypothesis_driven/    ├── test_driven/         ├── verify_first/
```

Each prompt family contains a `system_prompt.j2` and `user_prompt.j2` and is paired with an `agent_cls` in the config. See `configs/swebench_multi_tools.yaml` for the canonical 15-way bundle.

**How selection works** (`SWEBenchWrapper._setup_params`):

```yaml
agent_prompt_overrides:
  - user_prompt_template: prompts/codex/user_prompt.j2
    system_prompt_template: prompts/codex/system_prompt.j2
    agent_cls: CodexAgent
    diversify_tool_names: false
    camel_case_tool_names: false
  - ...
agent_prompt_override_random: false   # default
```

- If `agent_prompt_override_random=false` (default), one override is picked **deterministically per `instance_id`** (`random.Random(instance_id).choice(overrides)`). This means the same instance always gets the same prompt across runs — useful for evaluation reproducibility.
- If `agent_prompt_override_random=true`, one is picked uniformly at random per run — useful for RL training where you want every rollout of the same instance to potentially see a different prompt.
- The selected `user_prompt_template` and `system_prompt_template` are bind-mounted over OpenHands' default Jinja templates at `OpenHands/{user_prompt,system_prompt,system_prompt_long_horizon}.j2` (the same system template is mounted for both the regular and long-horizon variants).
- Paths in the override may be absolute or relative; relative paths are resolved against `nemo_gym.PARENT_DIR`.

If `agent_prompt_overrides` is unset, the OpenHands defaults are used and `agent_cls` defaults to `CodeActAgent`.

---

## Tool-name diversity

Two independent knobs control how OpenHands surfaces tools to the model. They are off by default and enabled by setting them on the override that gets selected.

| Override field            | Env var exported into the agent container | What the OpenHands fork does                                                |
|---------------------------|-------------------------------------------|------------------------------------------------------------------------------|
| `diversify_tool_names`    | `DIVERSIFY_TOOL_NAMES=true`               | Randomly samples from a pool of synonym tool names per run instead of using the canonical name (e.g. `bash` ↔ `execute_command` ↔ `shell`). |
| `camel_case_tool_names`   | `CAMEL_CASE_TOOL_NAMES=true`              | Re-encodes whatever name was chosen above into `camelCase` (e.g. `execute_command` → `executeCommand`). |

The two stack: with both enabled, the model sees a sampled-then-camelCased name. Use this to break the model's reliance on memorising specific tool names during training, and to test robustness during eval. Both are part of `AgentPromptOverride`, so the same per-instance / per-run selection logic that picks the prompt also picks these flags.

---

## Configuration reference

The full schema lives in `SWEBenchWrapperConfig` (and the per-override `AgentPromptOverride`) in `app.py`. Highlights:

| Field                              | Default                                           | Purpose                                                                 |
|------------------------------------|---------------------------------------------------|-------------------------------------------------------------------------|
| `agent_framework`                  | `openhands`                                       | Which agent harness drives the rollout: `openhands` or `opencode`. Switches both the in-SIF runtime mounted (OpenHands vs opencode + Bun) and the `BaseDatasetHarnessProcessor` selected in `_setup_params`. |
| `agent_framework_repo`             | OpenHands official                                | Fork to clone for the agent runtime.                                    |
| `agent_framework_commit`           | `HEAD`                                            | Commit to pin.                                                          |
| `agent_max_turns`                  | `100`                                             | Max agent iterations (OpenHands) / max turns the bench loop will spin (opencode). |
| `agent_config`                     | `null`                                            | Path to the per-harness model config (`configs/oh_config.toml` for OpenHands; opencode builds the config inline in `app.py`). |
| `agent_tools_file`                 | `null`                                            | (SWE-agent only) JSON tool list in OpenAI format.                       |
| `container_formatter`              | `docker://swebench/sweb.eval.x86_64.{instance_id}`| Path template (or list of templates) for SIFs. Supports the `_1776_` / `_s_` rewrites and fuzzy glob fallbacks. |
| `swebench_agent_timeout`           | `2700` (45 min)                                   | Per-instance agent wall-clock budget.                                   |
| `swebench_tests_timeout`           | `1800` (30 min)                                   | Per-instance eval wall-clock budget.                                    |
| `apptainer_memory_limit_mb`        | `65536` (64 GiB)                                  | Cumulative tree-RSS limit enforced by the gym-side memory watchdog (no cgroups in the enroot sandbox — see [Memory watchdog](#memory-watchdog-oom-handling)). `<= 0` disables it. |
| `memory_watchdog_enabled`          | `true`                                            | Enable the RSS memory watchdog for the agent and eval containers.       |
| `memory_watchdog_poll_interval_s`  | `1.0`                                             | Watchdog poll interval (seconds); bounds worst-case RSS overshoot of a fast allocator. |
| `command_exec_timeout`             | `300`                                             | OpenHands per-command timeout inside the agent container.               |
| `concurrency`                      | `256`                                             | Server-side asyncio semaphore for concurrent instances.                 |
| `dataset_path`                     | `null`                                            | Optional default dataset JSONL.                                         |
| `verify_golden_patch`              | `false`                                           | Skip the agent and eval the dataset's own golden patch. Supported for `swe-bench-ext` and the SWE-bench / SWE-bench_Multilingual families. See [Golden-patch validation](#golden-patch-validation). |
| `skip_eval`                        | `false`                                           | Run the agent normally but skip the eval container entirely; reward is forced to `0.0` since the patch is never graded. Useful for collecting agent trajectories without paying the eval cost. |
| `agent_prompt_overrides`           | `null`                                            | List of `AgentPromptOverride` entries. See above.                       |
| `agent_prompt_override_random`     | `false`                                           | `false` = deterministic per `instance_id`; `true` = random per run.     |
| `opencode_subagents_enabled`       | `false`                                           | (opencode only) Enable opencode's `task` tool so the main agent can spawn subagent sessions. See [Subagents](#opencode-integration). |
| `openhands_should_log`             | `false`                                           | If true, sets `LOG_LEVEL=DEBUG`, `LOG_TO_FILE=true`, etc.               |
| `debug`                            | `false`                                           | Enables Profiler around the agent run + dumps callgrind/dot/png.        |

Bundled YAML configs:

- `configs/swebench_openhands.yaml` — single OpenHands `CodeActAgent`, the simplest setup.
- `configs/swebench_openhands_training.yaml` — same shape as above but tuned for training.
- `configs/swe_agent_config.yaml` — alternative SWE-agent path (uses `agent_tools_file`).
- `configs/swebench_multi_tools.yaml` — full 15-way prompt × agent-class × tool-name bundle (OpenHands).
- `configs/swebench_opencode.yaml` — opencode harness (`agent_framework: opencode`), pinned fork + commit, no prompt overrides.
- `configs/swebench_opencode_training.yaml` — opencode harness tuned for training.
- `configs/swebench_opencode_multi_tools.yaml` — opencode harness with the 11-way `prompts/opencode_harness/*.txt` prompt bundle. See [Prompt overrides](#opencode-integration).
- `configs/swebench_opencode_no_instruction.yaml` / `configs/swebench_opencode_empty.yaml` — opencode ablation baselines (minimal / no methodology instructions).
- `configs/swebench_deepswe.yaml` / `configs/swebench_denovoswe.yaml` — the `deepswe` / `denovoswe` datasets (⚠️ WIP, see [Supported datasets and harnesses](#supported-datasets-and-harnesses)).

---

## Quick Start

### Prerequisites — install Apptainer

```bash
apt install -y wget && cd /tmp && \
    wget https://github.com/apptainer/apptainer/releases/download/v1.4.1/apptainer_1.4.1_amd64.deb && \
    apt install -y ./apptainer_1.4.1_amd64.deb
apptainer --version
```

### Step 1 — configure the model

In `env.yaml` at the NeMo-Gym root:

```yaml
# OpenAI
policy_base_url: https://api.openai.com/v1
policy_api_key: <your OpenAI API key>
policy_model_name: gpt-4.1-2025-04-14
```

Or run a local vLLM:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 131072 --enable-expert-parallel \
  --tensor-parallel-size 4 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --port 8000 --enforce-eager
```

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: dummy
policy_model_name: Qwen/Qwen3-Coder-30B-A3B-Instruct
```

### Step 2 — start the SWE-agents server

```bash
# OpenHands single-prompt (swap to
# responses_api_agents/swe_agents/configs/swebench_multi_tools.yaml for full
# prompt × agent-class × tool-name diversity)
gym env start \
    --config responses_api_agents/swe_agents/configs/swebench_openhands.yaml \
    --model-type vllm_model \
    +swe_agents.responses_api_agents.swe_agents.container_formatter=/lustre/xxx/images/swe-bench/swebench_sweb.eval.x86_64.\{instance_id\}.sif \
    +swe_agents.responses_api_agents.swe_agents.model_server.name=vllm_model
```

For converting docker images into the `.sif` files referenced by `container_formatter`, see [`nemo_skills/dataset/swe-bench/dump_images.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/swe-bench/dump_images.py).

You should see something like:

```
INFO:     Started server process [1815588]
INFO:     Uvicorn running on http://127.0.0.1:25347 (Press CTRL+C to quit)
```

### Step 3 — query the agent

```bash
python responses_api_agents/swe_agents/client.py
```

---

## Batch evaluation / data collection

```bash
gym eval run --no-serve \
    --agent swe_agents \
    --input swebench-verified-converted.jsonl \
    --output swebench-verified.openhands.qwen3-30b-coder.jsonl \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --temperature 0.7 \
    --top-p 0.8
```

`gym eval run` defaults to a concurrency of 100; tune to your hardware. View the results with:

```bash
ng_viewer +jsonl_fpath=swebench-verified.openhands.qwen3-30b-coder.jsonl
```

---

## Golden-patch validation

`verify_golden_patch=true` bypasses the agent entirely and runs only the dataset's evaluation harness against the sample's **own** golden patch (`instance_dict["patch"]`). Use it to:

- Verify that a dataset sample is well-formed — i.e. its stored golden patch actually `resolves` under our local eval harness.
- Smoke-test container images, SIF mounts, and harness setup without paying the agent's wall-clock cost.
- Triage failing samples — if the golden patch itself does not resolve, the agent's failure on the same sample is not the agent's fault.

Supported datasets:

| `dataset_name`                           | Processor                            |
|------------------------------------------|--------------------------------------|
| `swe-bench-ext`                          | `SweBenchExtDatasetProcessor`        |
| `princeton-nlp/SWE-bench*` (e.g. `_Verified`, `_Lite`) | `SweBenchDatasetProcessor`           |
| `SWE-bench/SWE-bench_Multilingual` (any name containing `SWE-bench_Multilingual`) | `SweBenchMultilingualDatasetProcessor` |
| any name containing `SWE-rebench`        | `SWERebenchDatasetProcessor`         |
| `deepswe` ⚠️ **WIP**                     | `DeepSWEDatasetProcessor` — golden patch is the Harbor `solution.patch`. |
| `denovoswe` ⚠️ **WIP**                   | `DeNovoSWEDatasetProcessor` — there is no model-style patch; golden = the image's pre-existing source. The harness sets `is_golden=true` (from `verify_golden_patch`) so the eval skips `clean.sh` + patch apply and grades the original source against `passed_ptp`. Empty `instance_dict["patch"]` is accepted only on the golden path. |

For each supported dataset, the wrapper:

1. Extracts the golden patch from `instance_dict["patch"]`.
2. Writes it into the predictions JSONL (`output_for_eval.jsonl`) in standard SWE-bench format (`{"instance_id", "model_patch", "model_name_or_path": "golden_patch_verification"}`) and also as a `.diff` file (used by `swe-bench-ext`).
3. Launches only the eval container; the agent container is skipped.
4. Returns `reward=1.0` iff the harness reports `resolved=true` for that instance.

### Running it

Start the server with `verify_golden_patch=true`:

```bash
# SWE-bench Verified
ng_run "+config_paths=[responses_api_agents/swe_agents/configs/swebench_openhands.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +swe_agents.responses_api_agents.swe_agents.verify_golden_patch=true \
    +swe_agents.responses_api_agents.swe_agents.container_formatter=/lustre/xxx/images/swe-bench/swebench_sweb.eval.x86_64.\{instance_id\}.sif \
    +swe_agents.responses_api_agents.swe_agents.model_server.name=vllm_model
```

Then collect rollouts against the dataset of interest:

```bash
# SWE-bench Verified
ng_collect_rollouts +agent_name=swe_agents \
    +input_jsonl_fpath=/path/to/datasets/swe_public_datasets_val_swebench.jsonl \
    +output_jsonl_fpath=results/swebench_verified.golden.jsonl \
    +num_repeats=1

# SWE-bench Multilingual
ng_collect_rollouts +agent_name=swe_agents \
    +input_jsonl_fpath=/path/to/datasets/swe_swebench_multilingual_test.jsonl \
    +output_jsonl_fpath=results/swebench_multilingual.golden.jsonl \
    +num_repeats=1
```

`num_repeats=1` is sufficient since the golden patch is deterministic — the only variance is harness-side flakiness.

A healthy dataset should report `reward=1.0` (i.e. `resolved=true`) on (close to) 100% of samples. Any sample that fails is either malformed (bad `FAIL_TO_PASS` / `PASS_TO_PASS`, wrong `base_commit`, stale `test_patch`) or its container image diverges from the harness expectation — fix those before training/eval'ing the agent against them.

### Aggregating results

```bash
ng_reward_profile +input_jsonl_fpath=<dataset.jsonl> \
    +rollouts_jsonl_fpath=results/<dataset>.golden.jsonl \
    +output_jsonl_fpath=results/<dataset>.golden.profiled.jsonl \
    +pass_threshold=1.0
python scripts/print_aggregate_results.py +jsonl_fpath=results/<dataset>.golden.profiled.jsonl
jq -C . swebench-verified.openhands.qwen3-30b-coder.jsonl | less -R
```

---

## Output format

Each `responses` call returns a `NeMoGymResponse` whose `output` is a Responses-API conversion of the OpenHands chat-completion trajectory, whose `tools` is the function-tool list the agent saw, and whose `metadata` carries `metrics` (a `SWEBenchMetrics` JSON) and the full `instance_config`. The same metrics are written incrementally to `<persistent_dir>/nemo_gym_metrics.json` for profiling and post-run inspection.

`run` wraps that in `SWEBenchVerifyResponse`:

```jsonc
{
  "responses_create_params": { /* full input incl. system+user prompts the agent saw */ },
  "response":               { /* output messages + tool calls */ },
  "reward": 1.0,            // 1.0 iff resolved, else 0.0
  "resolved": true,
  "patch_exists": true,
  "model_patch": "diff --git ...",
  "agent_error_kind": null, // "max_iteration" | "context_window" | "stuck_in_loop" | "oom" | "other" | null
  "agent_timed_out": false,
  "eval_timed_out": false,
  "oom_killed": false,       // memory watchdog killed the agent container — see Memory watchdog
  "eval_oom_killed": false,  // memory watchdog killed the eval container
  "agent_peak_rss_mb": 1024,
  "eval_peak_rss_mb": 512,
  "ray_queue_time": 0.12,
  "openhands_run_time": 412.3,
  "generation_start_timestamp": "2026-07-01T12:00:00.000000+00:00",
  "evaluation_start_timestamp": "2026-07-01T12:07:00.000000+00:00",
  "per_turn_metrics": {
    "response_latencies": [
      {
        "model": "model-name",
        "latency": 12.34,
        "response_id": "chatcmpl-123",
        "timestamp": "2026-07-01T12:01:00.000000+00:00"
      }
    ],
    "action_execution_latencies": [
      {
        "observation_type": "CmdOutputObservation",
        "observation_id": "42",
        "latency": 0.31,
        "message": "Command finished successfully.",
        "timestamp": "2026-07-01T12:01:01.000000+00:00"
      }
    ],
    "token_usages": [
      {
        "model": "model-name",
        "prompt_tokens": 4096,
        "completion_tokens": 512,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "context_window": 0,
        "per_turn_token": 4608,
        "response_id": "chatcmpl-123"
      }
    ]
  },
  "generation_apptainer_spinup_time": 11.4,
  "create_runtime_time": 2.1,
  "connect_to_runtime_time": 0.8,
  "initialize_runtime_time": 3.2,
  "total_command_exec_time": 48.6,
  "total_model_call_time": 301.7,
  "final_eval_apptainer_spinup_time":  9.7,
  "final_eval_time": 87.2,
  "instance_config": { /* the full per-instance SWEBenchWrapperInstanceConfig */ },
  "subagent_trajectories": null // (opencode only, when opencode_subagents_enabled) list of {session_id, parent_session_id, messages, tools}
}
```

---

## GRPO masking and failure modes

`SWEBenchWrapperInstanceConfig.mask_sample` is set to `True` (so downstream RL drops the gradient for this rollout) when:

1. The patch resolved the tests **but** the agent terminated in a `max_iteration` or `context_window` error — the reward is accidental.
2. The eval container hit `swebench_tests_timeout` — reward is unreliable.
3. The agent hit `swebench_agent_timeout` (wall-clock) regardless of `resolved`.
4. The memory watchdog killed the **agent** container (OOM).
5. The memory watchdog killed the **eval** container (OOM).

Agent error strings are bucketed by `_classify_agent_error`:

| Substring (case-insensitive)        | Bucket           |
|-------------------------------------|------------------|
| `maximum iteration`                 | `max_iteration`  |
| `ContextWindow` / `context window`  | `context_window` |
| `stuck in a loop`                   | `stuck_in_loop`  |
| anything else                       | `other`          |

`oom` is set directly by `_apply_watchdog_stats` when the memory watchdog kills the agent container — it bypasses `_classify_agent_error` since there's no error string to classify (the container is killed, not erroring out on its own). See [Memory watchdog](#memory-watchdog-oom-handling).

---

## Memory watchdog (OOM handling)

The enroot/apptainer sandbox this runs under has no cgroup support (v1 + fakeroot + no systemd — `ulimit -v` and `apptainer --memory` both fail), so per-container memory limits are enforced in userspace by `RunOpenHandsAgent._memory_watchdog`:

- Every `memory_watchdog_poll_interval_s` seconds, it snapshots the container's full process tree (`_scan_container_tree`, via `/proc/<pid>/task/*/children` where available, else a full `/proc` sweep + ppid walk) and sums `rss_bytes` across every process.
- If cumulative tree RSS reaches `apptainer_memory_limit_mb`, it SIGKILLs every process group in the tree (`_kill_container_tree` — `os.killpg` per `pgid`, with a per-pid fallback for stragglers that forked mid-kill) and records `oom_killed=True` plus the peak RSS observed (`agent_peak_rss_mb` / `eval_peak_rss_mb`).
- The watchdog runs as an `asyncio.Task` alongside the container subprocess (started in `_start_container_command`, cancelled in `_finish_container_command`'s `finally`) and is disabled entirely when `memory_watchdog_enabled=false` or `apptainer_memory_limit_mb <= 0`.
- A watchdog-triggered kill surfaces as a `RuntimeError` from `_finish_container_command`, which `process_single_datapoint` catches and turns into `oom_killed` / `eval_oom_killed` on `SWEBenchMetrics` — both feed into [GRPO masking](#grpo-masking-and-failure-modes) regardless of the (now-irrelevant) `resolved` value.
- Peak RSS is tracked even when the limit is never crossed, so `agent_peak_rss_mb` / `eval_peak_rss_mb` are useful for right-sizing `apptainer_memory_limit_mb` from real rollout data.

---

## Debug / profiling

Set `debug=true` to wrap the agent run in a `Profiler` (callgrind output), then auto-render `.dot` and `.png` graphs via `gprof2dot` + `pydot` after the run. Profiling output lands under `<persistent_dir>/profiling/`. Apptainer also exports `NG_PROFILING_DIR` into the agent container so the OpenHands fork can dump matching profiles.

Set `openhands_should_log=true` to flip OpenHands to `LOG_LEVEL=DEBUG`, `LOG_TO_FILE=true`, and write per-event logs. Otherwise the wrapper aggressively quiets OpenHands (`LOG_LEVEL=CRITICAL`, all `DEBUG_*` flags off).

Per-instance Apptainer stdout/stderr is always streamed to `<persistent_dir>/apptainer_logs/<instance_id>_{agent,eval}.log` regardless of these flags.

To inspect a completed run as a timeline, convert the precise per-rollout metrics to Chrome Trace Event Format:

```bash
python responses_api_agents/swe_agents/scripts/swe_trace_converter.py \
    --log-dir /path/to/swebench_results_<run_session_id>
```

`--log-dir` must directly contain the completed `<instance_id>_<timestamp>_<uuid>/` rollout directories and their `nemo_gym_metrics.json` files.

Open the generated `.json` file in [Perfetto](https://ui.perfetto.dev/) to explore rollout concurrency and per-turn timing.

For the opencode harness, the agent script installs an `EXIT` trap (`opencode_log_trap`) that always copies opencode's internal XDG data dir (`/root/.local/share/opencode`) and any `/tmp/bench-*/data/log` directories back to `<persistent_dir>/opencode_logs/` — useful for debugging opencode-internal issues (e.g. session/storage errors) that don't show up in the `llm_completions` trajectory dump.
