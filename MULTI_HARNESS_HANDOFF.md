# NeMo multi-harness training: cross-cluster handoff

Last reconciled: **2026-10-09 10:43 PDT**

This is the operational handoff for resuming the NeMo Gym + NeMo RL
multi-harness work on another cluster. It records what is pushed, what has
actually been validated, the remaining blocker, the source artifacts, and the
exact acceptance gates. The longer design and code map are in
[`fern/versions/latest/pages/training-tutorials/multi-harness-training-handoff.mdx`](fern/versions/latest/pages/training-tutorials/multi-harness-training-handoff.mdx).

## Current live status

- The current implementation commits are Gym `dda31404b` and NeMo RL
  `02936d5f` on branch `ehosseiniasl/multi-harness-training-routing`. They are
  the commits that must be present before the next runtime validation.
- Nano job `2180485`, W&B run
  [`dv8ch7n8`](https://wandb.ai/adlr/multi-harness-RL/runs/dv8ch7n8), completed
  all four optimizer updates available in its one-epoch configuration. The
  final reported reward was `0.625`, gradient norm was approximately `0.6386`,
  maximum TMPE was approximately `1.0351`, and token capture reported zero
  masked samples, zero invalid rows, and zero poisoned rows. Its Slurm state is
  `FAILED` only because the old post-run auditor required nonzero response-level
  usage, while all eight OpenClaw response objects reported zero usage.
- The same run exposed two remaining harness issues. OpenClaw's native
  compaction requests used the `reasoning` assistant field that Gym did not
  accept, so all OpenClaw trajectories likely ended through the harness salvage
  path. Hermes learned the real 16K limit only after a 16,385-token backend
  rejection.
- Gym `dda31404b` fixes those issues. The chat schema now preserves both
  `reasoning` and `reasoning_content`; AnyTerminal keeps policy calls on the
  correlated token-capture URL while routing native compaction to the
  uncorrelated model endpoint; Hermes receives the real 15,872-token context
  and uses the uncorrelated endpoint for compression.
- RL `02936d5f` fixes observability lost during token-capture reassembly.
  Scalar and numeric-histogram rollout metrics now survive the finalizer and
  are mirrored into the resolved harness namespace. W&B should receive keys
  such as `train/anyterminal_openclaw/total_reward/mean`,
  `train/anyterminal_pi/turns_per_sample/mean`, and equivalent OpenCode and
  Hermes token/success/masking/termination metrics. Full-result W&B Tables are
  deliberately not sent through the metadata-only finalizer sidecar.
- The Qwen, Nano, and Super recipes now set both `max_num_epochs` and
  `max_num_steps` to `1000000`; the scheduler allocation is the practical run
  boundary. Do not override either field to four or one. W&B is enabled for
  entity/project `adlr/multi-harness-RL`.
- Focused validation for the new commits passes: 9 Gym schema/streaming/runner
  tests, 3 RL rollout/finalizer metric tests, 5 RL recipe-resolution tests, and
  Ruff formatting/lint checks. A fresh four-harness Nano W&B run is still
  required to prove that the new per-harness series populate with real data.
- Acceptance remains blocked on that clean Nano rerun followed by the requested
  Super 8-node synchronous and Super 16-node asynchronous runs. Both Super runs
  must use
  `/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp`
  and online W&B project `adlr/multi-harness-RL`.

The older evidence below is retained as failure history. When its wording says
"newest," interpret that relative to its section; this live-status section is
authoritative.

## Stop line

- FineEnvs-style fan-out is implemented across NeMo Gym and NeMo RL.
- The P0 harnesses are OpenCode, OpenClaw, Pi, and Hermes.
- Every input row is expanded to all four harnesses when the recipe uses
  `fan_out`. GRPO siblings remain grouped by task and harness.
- Focused unit/config tests pass. The exact fetchable commits are recorded
  below.
- The OpenClaw runtime/config fix is committed at Gym commit `845f13a1e`.
  It upgrades OpenClaw to `2026.6.35`, keeps Node at the required `22.19.0`,
  disables plugins for the terminal profile, and preserves the lean terminal
  prompt/tool policy. The rebuilt AnyTerminal bundle completed a real,
  deterministic two-request `exec pwd` tool loop with a final assistant
  response and a complete capture chain.
- The Pi context/compaction fix is committed at Gym commit `2d0579fea`. It
  advertises a 15,872-token context, uses a remaining-context output policy,
  and compacts with 4,096-token reserve/recent-history limits.
- NeMo RL commit `58d1b148` fixes the finalizer/replay race exposed by job
  `2178307`. Commit `68ff2f02` additionally resolves relative vLLM parser
  plugin paths from the installed `nemo_rl` package root instead of the Ray
  worker's current directory. The focused parser-path regression, Ruff, Ruff
  format, and `git diff --check` pass.
- **End-to-end training validation is not complete.** The newest two-node Nano
  sync job is `2179399`, W&B run
  [`plth7xte`](https://wandb.ai/adlr/multi-harness-RL/runs/plth7xte). It proves
  the parser-plugin fix under the real Ray/vLLM launch, loads all 16 fan-out
  groups, completes the initial 16-rank refit, and dispatches both siblings of
  `configure-git-webserver` through all four harnesses. It then exposes an
  OpenClaw context-compaction/token-lineage blocker described below. This is
  diagnostic evidence, not a pass.
- Do not report either PR as runtime-validated until Nano sync, Super 8-node
  sync, and Super 16-node async meet the gates below.
- Job `2179252` is an invalid predecessor: its Ray worker could not resolve the
  repository-relative `nano_v3` reasoning-parser plugin. It was stopped and
  must never be counted. Job `2179399` is the replacement launched after RL
  commit `68ff2f02` was pushed.

### Newest Nano/TMPE evidence (`2179399`)

The live run was launched interactively on two nodes with no artificial GRPO
step limit (`max_num_steps=1000000`). It uses the real Nano Omni model, the
four Terminal-Bench 2.1 rows, two siblings per harness, online W&B, and exact
imports from both PR worktrees. Runtime evidence includes:

- all 11 Gym services became ready;
- vLLM loaded `nano_v3` from the corrected package-root path and enabled the
  configured reasoning/tool parsers;
- 16/16 GPU worker units became healthy;
- the initial policy-to-generation refit completed in 1.814 seconds;
- all eight first-task sandboxes started: four harnesses times two siblings;
- OpenCode produced 2/2 results, Hermes 2/2, Pi 2/2, and OpenClaw 1/2 by the
  reconciliation time above.

The missing OpenClaw sibling is a real TMPE blocker, not an infrastructure
pass. Its rollout ID is
`e1990b08-e2f0-4dc8-96a8-352f1734220a_g1_a29612efe554b4fc39e77f9b38e5113bd`.
Its first model call was admitted, but two later calls were poisoned as
`unresolved_parent`. OpenClaw then issued its native anchored-context summary
request with `max_tokens=8192`; Gym rejected that request shape with 18
validation errors. The exact system prompt begins `You are an anchored context
summarization assistant for coding sessions.` The next implementation task is
to either disable this compaction path for the terminal profile or preserve
the correct response-parent chain and submit a request within the model's
actual 16,384-token boundary. Do not launch Super validation until a fresh Nano
run completes all 32 rollouts and four optimizer steps without this failure.

Current artifacts:

```text
/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/2179399-logs/ray-driver.log
/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/validation/anyterminal-p0/nano-omni-sync2-debug-grpo/run-nano-omni-sync2-pluginfix-int-20261008-1726
```

The last completed diagnostic is Slurm job `2178116`, W&B run
[`9uirholo`](https://wandb.ai/adlr/multi-harness-RL/runs/9uirholo). It used the
terminal-only OpenClaw policy and the 15,872 context / 4,096 output limits.
Both OpenClaw siblings still returned this locally, with zero input/output
tokens and no token-capture records:

```text
Context overflow: prompt too large for the model. Try /reset (or /new) to
start a fresh session, or use a larger-context model.
```

This is an OpenClaw preflight/prompt-assembly failure, not a vLLM HTTP error.
The other three prompt groups were not enough to satisfy the four-group
training floor, so step 0 failed with `rollout_failed:no_records` for both
OpenClaw siblings.

The first fix after that failure removed OpenClaw bootstrap/context injection,
startup memory, and skills while preserving only the `exec` tool. Job
`2178307` showed that OpenClaw `2026.6.11` could still return fallback/empty
transcripts. Commit `845f13a1e` subsequently upgraded the exact bundled
runtime to `2026.6.35` and disabled plugins in the terminal profile. The
rebuilt bundle passes the standalone tool-loop/capture probe, but has not yet
completed a fresh Nano training run.

An earlier Nano diagnostic is Slurm job `2178307`, W&B run
[`rm12vt5c`](https://wandb.ai/adlr/multi-harness-RL/runs/rm12vt5c). It loaded
exactly 16 fan-out prompt groups and performed the initial policy-to-generation
refit, then completed the first `configure-git-webserver` batch with two
rollouts per harness. OpenCode and Pi emitted non-empty policy text with
non-zero recorded usage. Hermes emitted non-empty text, but response-level
usage remained zero. OpenClaw emitted one local fallback message and one empty
message, both with zero response usage; token capture also reported an
`unresolved_parent`. This run cannot satisfy the training gate.

After rejecting the bad OpenClaw sibling, NeMo RL reported the target step one
group short and closed it early. The root cause of the later shape mismatch was
an atomicity gap: the finalizer marked the low-validity group replay-ready
before the controller applied `min_valid_fraction_per_group`. The train pump
could claim the group in that window; cleanup then deleted its canonical rows,
so packing planned four rows while TQ returned three. Commit `58d1b148` moves
the validity gate into the same mutation cut as publication, ensuring rejected
groups never become selectable. The focused race regression, Ruff, Ruff format,
and targeted Pyrefly checks pass.

The same run proved the old Pi defaults were unsafe for this backend. Pi
advertised a 262,144-token context and fixed 131,072-token output budget while
vLLM only supports 16,384 tokens. One prompt reached 16,385 tokens and was
rejected with HTTP 400. Pi's native overflow compaction then requested an
8,192-token summary through a request shape rejected by Gym validation. Commit
`2d0579fea` fixes the advertised budget and compaction thresholds; it has not
yet been validated by a fresh Nano training run.

## Pull requests, branch, and minimum commits

Use branch `ehosseiniasl/multi-harness-training-routing` in both repositories.

| Repository | Pull request | Minimum implementation commit |
|---|---|---|
| NeMo Gym | [NVIDIA-NeMo/Gym#4082](https://github.com/NVIDIA-NeMo/Gym/pull/4082) | `dda31404b` |
| NeMo RL | [NVIDIA-NeMo/RL#4521](https://github.com/NVIDIA-NeMo/RL/pull/4521) | `02936d5f` |

The Gym branch tip also contains documentation commits newer than the
implementation hash. Fetch the branch tip and use the hashes above only as
minimum ancestry checks:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git nemo-gym-multi-harness
cd nemo-gym-multi-harness
git remote add contributor https://github.com/ehosseiniasl/Gym.git
git fetch contributor ehosseiniasl/multi-harness-training-routing
git switch -c ehosseiniasl/multi-harness-training-routing \
  --track contributor/ehosseiniasl/multi-harness-training-routing
git merge-base --is-ancestor dda31404b HEAD

cd ..
git clone https://github.com/NVIDIA-NeMo/RL.git nemorl-multi-harness
cd nemorl-multi-harness
git remote add contributor https://github.com/ehosseiniasl/NeMo-RL.git
git fetch contributor ehosseiniasl/multi-harness-training-routing
git switch -c ehosseiniasl/multi-harness-training-routing \
  --track contributor/ehosseiniasl/multi-harness-training-routing
git merge-base --is-ancestor 02936d5f HEAD
```

After cloning, both ancestry commands must exit zero. Also run `git status
--short --branch` in each checkout and confirm there are no local changes.

## Routing contract: use `fan_out`

The generic source route is `anyterminal_multi_harness`. It is intentionally
not named after one of the harnesses.

```yaml
env:
  nemo_gym:
    fan_out:
      anyterminal_multi_harness:
        - anyterminal_opencode
        - anyterminal_openclaw
        - anyterminal_pi
        - anyterminal_hermes
```

The three routing modes are different:

| Field | Meaning |
|---|---|
| `agent_map` | Route a matching row to one fixed harness. |
| `agent_pool` | Deterministically select one harness from a list. |
| `fan_out` | Copy every matching row once per harness; this is the required mode. |

Do not replace `fan_out` with `agent_pool` if the requirement is “every prompt
goes to all four harnesses.” Expansion happens before batching and sibling
generation, and NeMo RL stamps each expanded row with its concrete harness so
it is not expanded again inside Gym.

## Files carrying the implementation

NeMo Gym:

- `nemo_gym/openai_utils.py`
- `nemo_gym/rollout_collection.py`
- `nemo_gym/global_config.py`
- `nemo_gym/train_data_utils.py`
- `responses_api_agents/anyterminal_agent/app.py`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness_enroot.yaml`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_openclaw.yaml`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_pi.yaml`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_hermes.yaml`
- `responses_api_agents/hermes_agent/app.py`
- `responses_api_agents/opencode_agent/app.py`
- `responses_api_agents/pi_agent/app.py`
- `tests/unit_tests/test_anyterminal_multi_harness.py`

NeMo RL:

- `nemo_rl/data/datasets/response_datasets/nemogym_dataset.py`
- `nemo_rl/environments/nemo_gym.py`
- `nemo_rl/environments/nemo_gym_shards.py`
- `nemo_rl/experience/rollout_manager.py`
- `nemo_rl/experience/rollout_reassembler_actor.py`
- `nemo_rl/experience/rollouts.py`
- `nemo_rl/algorithms/single_controller.py`
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_qwen3_0_6b_single_controller.yaml`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_nano_omni_sync_2n_debug_single_controller.yaml`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_single_controller.yaml`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_sync_8n_single_controller.yaml`
- `tests/unit/environments/test_anyterminal_multi_harness_recipe.py`
- `tests/unit/models/generation/test_vllm_generation.py`

## Source-cluster artifacts to copy or remap

The source scripts contain hard-coded source-cluster paths. Update `ROOT`,
`USER_ROOT`, `IMAGE`, checkpoint, cache, account, partition, and QoS values
before submitting them elsewhere.

| Artifact | Source-cluster path |
|---|---|
| Validation bundle | `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/validation/anyterminal-p0/` |
| Four-row dataset | `.../validation/anyterminal-p0/collated/train.jsonl` |
| Nano 2-node wrapper | `.../validation/anyterminal-p0/run_nano_omni_sync2_debug_grpo_slurm.sh` |
| Shared in-container driver | `.../validation/anyterminal-p0/run_super_async_grpo_inside.sh` |
| Super 8-node sync wrapper | `.../validation/anyterminal-p0/run_super_sync8_grpo_slurm.sh` |
| Super 16-node async wrapper | `.../validation/anyterminal-p0/run_super_async_grpo_slurm.sh` |
| Run auditor | `.../validation/anyterminal-p0/validate_super_run.py` |
| Gym server venv cache | `.../validation/anyterminal-p0/async-grpo/server_venvs_ray258/` |
| Terminal-Bench Enroot cache | `.../validation/anyterminal-p0/async-grpo/task-images/` |
| Outer RL/Gym container | `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/mingjiel/containers/rl-gym.70626244-ray258.sqsh` |
| Super checkpoint | `/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp` |
| Credentials file | `/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env` |

The credentials file contains `WANDB_API_KEY` and `HF_TOKEN`. Transfer it only
through an approved secret channel. Never commit, print, or paste its contents.
Recreate GitHub authentication on the destination cluster with its normal
`gh auth login` or `ssh-agent` flow.

The four Terminal-Bench 2.1 rows are:

| Stable index | Task | Container |
|---|---|---|
| 0 | `configure-git-webserver` | `alexgshaw/configure-git-webserver:20251031` |
| 1 | `fix-git` | `alexgshaw/fix-git:20260403` |
| 2 | `log-summary-date-ranges` | `alexgshaw/log-summary-date-ranges:20251031` |
| 3 | `modernize-scientific-stack` | `alexgshaw/modernize-scientific-stack:20251031` |

Every JSONL row must retain:

```json
{"task_source": "anyterminal_multi_harness"}
```

Copy the validation bundle if the two clusters share no filesystem. Large
model/container/image caches can instead be recreated or remapped. Verify all
four task images, task assets, and verifier tests from every Ray node before
allocating the full Super jobs.

## Environment setup on the destination cluster

Export these in the driver and propagate them to every Ray worker/container:

```bash
export WANDB_MODE=online
export WANDB_ENTITY=adlr
export WANDB_PROJECT=multi-harness-RL
export WANDB_API_KEY=...  # load from a secret file
export HF_TOKEN=...       # required if the model is not already cached
```

Prepend both PR checkouts to `PYTHONPATH`, or overlay the Gym checkout into
NeMo RL's `3rdparty/Gym-workspace/Gym`. Confirm the actual imports before
requesting GPUs:

```bash
python - <<'PY'
import nemo_gym
import nemo_rl
print(nemo_gym.__file__)
print(nemo_rl.__file__)
PY
```

Both paths must point to the two PR checkouts. A system-installed package does
not validate these changes.

The launchers also set `NEMO_GYM_VENV_DIR`,
`NEMO_GYM_ENROOT_SQSH_CACHE`, vLLM/FlashInfer cache locations, and the
checkpoint/chat-template overrides. Preserve those settings or map them to
equivalent shared paths on the destination cluster.

## Tests before GPU submission

From the Gym checkout:

```bash
uv run pytest -q tests/unit_tests/test_anyterminal_multi_harness.py
uv run pytest -q responses_api_agents/openclaw_agent/tests/test_app.py
uv run pytest -q responses_api_agents/pi_agent/tests/test_app.py \
  -k test_run_stages_private_mcp_config_and_cleans_workspace
uv run pytest -q tests/unit_tests/test_openai_utils.py \
  tests/unit_tests/test_chat_completions_streaming.py \
  responses_api_agents/anyterminal_agent/tests/test_app.py
uv run ruff check responses_api_agents/pi_agent/app.py \
  responses_api_agents/pi_agent/tests/test_app.py \
  tests/unit_tests/test_anyterminal_multi_harness.py
git diff --check
```

At commit `dda31404b`, the nine newly relevant Gym
schema/streaming/AnyTerminal runner tests pass together. The two Hermes config
tests also passed in the extracted Hermes dependency environment. Focused Ruff,
Ruff format, and `git diff --check` pass.

From the RL checkout:

```bash
uv run pytest -q tests/unit/environments/test_anyterminal_multi_harness_recipe.py
uv run pytest -q \
  tests/unit/experience/test_rollout_manager.py::test_capture_metrics_are_mirrored_into_the_resolved_harness_namespace \
  tests/unit/experience/test_rollout_manager.py::TestGenerateForFinalizationFlow::test_request_preserves_per_harness_scalar_metrics \
  tests/unit/single_controller/test_finalizer_lifecycle.py::test_successful_actor_finalization_returns_actor_and_transfers_ownership
uv run pytest -q tests/unit/models/generation/test_vllm_generation.py \
  -k reasoning_parser_plugin
git diff --check
```

At commit `02936d5f`, all five multi-harness recipe tests and all three focused
rollout/finalizer metric tests pass with a working Ray cluster. Focused Ruff,
Ruff format, and `git diff --check` also pass. The source login sandbox cannot
start Ray GCS, so run those tests outside the sandbox or on a compute node.

## Implemented OpenClaw runtime and prompt fix

The Gym profile narrows the OpenClaw tools to `exec` and now also removes the
fixed bootstrap, startup-memory, and skill prompt material:

```yaml
openclaw_config:
  agents:
    defaults:
      workspace: "."
      skipBootstrap: true
      contextInjection: never
      startupContext:
        enabled: false
      skills: []
  skills:
    limits:
      maxSkillsInPrompt: 0
      maxSkillsPromptChars: 0
  tools:
    profile: minimal
    alsoAllow:
      - exec
    deny:
      - session_status
```

Job `2178116` proved tool reduction alone was insufficient. The added controls
target the fixed OpenClaw prompt. `contextInjection:
never` makes the runtime provide no bootstrap/context files; `skills: []`
filters every discovered skill, with zero prompt limits as a defensive second
gate; and startup memory is disabled. `localModelLean` is deliberately not
enabled because this OpenClaw release can use it to add a tool-search surface,
while the explicit terminal policy is already stricter.

OpenClaw is pinned to `2026.6.35`, Node is pinned to `22.19.0`, and the
AnyTerminal profile sets `plugins.enabled=false`. A rebuilt exact runtime
bundle completed a deterministic OpenAI-compatible two-request loop: it sent
an `exec pwd` tool call, consumed the real tool result, emitted non-empty final
assistant text, reported 400 input / 40 output tokens, and produced a complete
session lineage with no gaps. The focused 66-test suite and config validation
also pass. This resolves the standalone runtime/capture reproduction, but only
a fresh Nano run can establish full training compatibility.

If Nano still overflows, measure which prompt sections remain before
increasing the advertised context. Do not simply advertise the full 16,384
vLLM boundary: an earlier run proved that leaves no server-side generation
token.

## Validation order

1. Fetch Gym commit `dda31404b` or newer and RL commit `02936d5f` or newer,
   then rerun their focused tests.
2. Run the two-node synchronous Nano recipe:
   `grpo_anyterminal_multi_harness_nemotron_nano_omni_sync_2n_debug_single_controller.yaml`.
3. Leave `grpo.max_num_epochs` and `grpo.max_num_steps` at `1000000`. Let the
   job run until the scheduler allocation or an explicit operator stop; never
   cap it at four optimizer steps or one dataset epoch.
4. After at least one complete 16-group cycle, check local artifacts and W&B,
   including all four `train/anyterminal_<harness>/...` namespaces, rather than
   relying only on process state or `mask_sample`.
5. After Nano passes, run the requested Super checkpoint with the 8-node sync
   and 16-node async recipes. They may run in parallel after the Nano gate.
6. Add the passing W&B links and score/TMPE/refit summary to both PRs.

The expected shape of each complete dataset cycle is:

```text
4 source tasks x 4 harnesses = 16 prompt groups
16 prompt groups x 2 siblings = 32 rollouts
32 rollouts / train batch 8 = 4 optimizer updates per cycle
cycles repeat until the scheduler or operator ends the run
```

Run the bundled auditor against a completed window. The auditor must accept
`>=4` optimizer updates and must not require the training process to stop at
four:

```bash
python /path/to/validation/anyterminal-p0/validate_super_run.py \
  /path/to/run-directory
```

Do not accept a run unless all of these hold:

1. Exactly two results exist for every task-by-harness pair: 32 total.
2. Every response is non-empty and is real policy output, not local harness
   error text.
3. Every result has `mask_sample=false`, `failure_kind=null`,
   `failure_reason=null`, `agent_timed_out=false`,
   `container_timed_out=false`, and `sandbox_failed=false`.
4. All 32 rollouts have generated token IDs and log probabilities; there are
   no `no_records`, incomplete, or ambiguous capture chains.
5. TMPE is finite, every optimizer step has eight valid samples, and no sample
   is masked for log-probability error.
6. At least four consecutive optimizer updates complete, training remains able
   to continue, and W&B contains per-harness metric series plus the configured
   Gym result tables.
7. Reward and advantages are finite and have non-zero variance. Loss and
   gradient norm are finite, and gradient norm is not identically zero.
8. Policy-to-generation refit completes initially and after every optimizer
   step.

## Diagnostic evidence

| Slurm job | W&B | What it proves | Status |
|---|---|---|---|
| `2180348` | [`ra55t6cy`](https://wandb.ai/adlr/multi-harness-RL/runs/ra55t6cy) | Replacement Nano sync run on Gym `7f8454ac1`; loaded 16 groups, completed service/model initialization and initial refit, then began four-harness rollout collection. | Running at last reconciliation; audit before accepting |
| `2180210` | [`2iseuhsj`](https://wandb.ai/adlr/multi-harness-RL/runs/2iseuhsj) | Confirmed real four-harness rollout and fail-closed rejection of an unresolved OpenClaw response parent; isolated reasoning normalization from a ledger-publication race. | Failed safely at step 0; no optimizer step |
| `2179399` | [`plth7xte`](https://wandb.ai/adlr/multi-harness-RL/runs/plth7xte) | Real Nano sync launch with all 16 fan-out groups, healthy vLLM parser-plugin loading, initial refit, and all four harnesses on the same task. Exposed OpenClaw compaction request rejection plus unresolved token-capture parentage. | Diagnostic; first batch incomplete at reconciliation time |
| `2179252` | none accepted | Reproduced Ray-worker cwd sensitivity in relative parser-plugin resolution. Led to RL commit `68ff2f02`. | Stopped; invalid for acceptance |
| `2179143` | none | Superseded queued Nano attempt. | No acceptance evidence |
| `2178307` | [`rm12vt5c`](https://wandb.ai/adlr/multi-harness-RL/runs/rm12vt5c) | Loaded 16 fan-out groups, completed initial refit, and ran the first 4-harness/2-sibling batch. Exposed Pi context/compaction, OpenClaw transcript/capture, and partial-batch log-probability failures. | Failed at step 0 after 16m47s |
| `2178116` | [`9uirholo`](https://wandb.ai/adlr/multi-harness-RL/runs/9uirholo) | Terminal-only tools and 15,872/4,096 limits were active; OpenClaw still overflowed locally with zero usage. | Failed before step 0 |
| earlier Nano | [`a5y7hxq2`](https://wandb.ai/adlr/multi-harness-RL/runs/a5y7hxq2) | Terminal-only config reached runtime; OpenClaw inherited an 8,192 output reserve and overflowed locally. | Diagnostic only |
| `2177783` | [`z31vasi6`](https://wandb.ai/adlr/multi-harness-RL/runs/z31vasi6) | 15,872 guard removed the prior vLLM HTTP 400, exposing local OpenClaw overflow. | Diagnostic only |
| older Qwen smoke | [`zx1q7m3q`](https://wandb.ai/adlr/multi-harness-RL/runs/zx1q7m3q) | Multi-harness plumbing ran, but reward, advantages, loss, and gradient norm were all zero. | Not learning evidence |

The failed Nano run directory is:

```text
/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/validation/anyterminal-p0/nano-omni-sync2-debug-grpo/run-nano-omni-sync2-lean-20261008-1439
```

Its Ray driver log is:

```text
/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/github_repos/nemorl-multi-harness/2178307-logs/ray-driver.log
```

Job `2178307` ended `FAILED` after 16m47s with exit code 1. Its resolved
configuration had W&B online at `adlr/multi-harness-RL`, full Gym tables
enabled, `max_num_steps=1000000`, and four expected optimizer steps. It
produced all eight first-batch response files, but OpenClaw had one 56-byte
fallback message and one empty message, both at zero usage. Pi had non-empty
output and non-zero usage, but also hit the 16,385-token rejection and invalid
compaction requests. The OpenClaw group was rejected at 1/2 valid siblings;
the resulting partial batch crashed Megatron log-probability dispatch at step
0, before any optimizer step. The run predates Pi commit `2d0579fea` and must
not count as validation of that fix.

For comparison, the two zero-usage responses from failed job `2178116` are
under its run directory at
`anyterminal-results/openclaw/configure-git-webserver_*/response.json`.

At the last reliable scheduler check, Super job `2172269` (8-node sync) and
job `2172270` (16-node async) were pending for `Priority`. They launch mutable
worktrees, so their output must not count unless the resolved runtime config
contains the final OpenClaw fix. On a new cluster, submit fresh jobs instead
of reusing these IDs.

## Resume checklist

1. Fetch both branch tips and verify Gym contains `dda31404b` and RL contains
   `02936d5f` with the minimum-commit ancestry checks.
2. Copy/remap the validation bundle, outer container, caches, four task images,
   dataset, and model checkpoint.
3. Restore secrets without printing them and confirm W&B is online at
   `adlr/multi-harness-RL`.
4. Verify both imported package paths and run the focused tests.
5. Confirm the lean OpenClaw and bounded Pi configurations are present in the
   resolved runtime configuration.
6. Verify the rebuilt OpenClaw `2026.6.35` bundle and plugin-disabled terminal
   profile are the versions used inside the training container.
7. Verify RL commits `58d1b148` and `68ff2f02` are active: rejected groups
   never appear in a selected training batch, and relative parser plugins load
   independently of Ray worker cwd.
8. Run uncapped Nano sync and audit the first complete 32-rollout/four-update
   cycle while the run continues.
9. Run Super 8-node sync and Super 16-node async with the specified Super
   checkpoint.
10. Record per-harness scores, TMPE, reward/advantage/loss/gradient metrics,
   refit timings, and W&B URLs in both PR descriptions.
11. Re-run current PR checks and confirm both local worktrees match their
   remote branches.

Until steps 8 and 9 pass, the correct project status is: **multi-harness fan-out
implemented; OpenCode, OpenClaw, Pi, and Hermes are routed for every task;
native compaction and token-lineage fixes are focused-tested; per-harness W&B
metrics survive token-capture finalization in unit tests; the prior Nano run
completed four optimizer updates but predates the compaction/metrics fixes; a
clean uncapped Nano rerun and both Super training validations remain pending**.
