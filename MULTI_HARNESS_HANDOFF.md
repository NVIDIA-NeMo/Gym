# NeMo multi-harness training: cross-cluster handoff

Last reconciled: **2026-10-08 15:12 PDT**

This is the operational handoff for resuming the NeMo Gym + NeMo RL
multi-harness work on another cluster. It records what is pushed, what has
actually been validated, the remaining blocker, the source artifacts, and the
exact acceptance gates. The longer design and code map are in
[`fern/versions/latest/pages/training-tutorials/multi-harness-training-handoff.mdx`](fern/versions/latest/pages/training-tutorials/multi-harness-training-handoff.mdx).

## Stop line

- FineEnvs-style fan-out is implemented across NeMo Gym and NeMo RL.
- The P0 harnesses are OpenCode, OpenClaw, Pi, and Hermes.
- Every input row is expanded to all four harnesses when the recipe uses
  `fan_out`. GRPO siblings remain grouped by task and harness.
- Focused unit/config tests pass. The exact fetchable commits are recorded
  below.
- The lean OpenClaw prompt fix is pushed at Gym commit `df0c097f7`. Focused
  tests, native config validation, and a real OpenClaw CLI loopback probe pass.
- The Pi context/compaction fix is committed at Gym commit `2d0579fea`. It
  advertises a 15,872-token context, uses a remaining-context output policy,
  and compacts with 4,096-token reserve/recent-history limits.
- **End-to-end training validation is not complete.** Nano sync job `2178307`
  reached real rollouts and W&B, but its first batch exposed Pi and OpenClaw
  runtime blockers described below. It is diagnostic evidence, not a pass.
- Do not report either PR as runtime-validated until Nano sync, Super 8-node
  sync, and Super 16-node async meet the gates below.

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

The fix after that failure removes OpenClaw bootstrap/context injection,
startup memory, and skills while preserving only the `exec` tool. Job
`2178307` is the first real training validation of this complete fix; do not
infer its result from its `RUNNING` state.

The current Nano diagnostic is Slurm job `2178307`, W&B run
[`rm12vt5c`](https://wandb.ai/adlr/multi-harness-RL/runs/rm12vt5c). It loaded
exactly 16 fan-out prompt groups and performed the initial policy-to-generation
refit, then completed the first `configure-git-webserver` batch with two
rollouts per harness. OpenCode and Pi emitted non-empty policy text with
non-zero recorded usage. Hermes emitted non-empty text, but response-level
usage remained zero. OpenClaw emitted one local fallback message and one empty
message, both with zero response usage; token capture also reported an
`unresolved_parent`. This run cannot satisfy the training gate.

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
| NeMo Gym | [NVIDIA-NeMo/Gym#4082](https://github.com/NVIDIA-NeMo/Gym/pull/4082) | `2d0579fea` |
| NeMo RL | [NVIDIA-NeMo/RL#4521](https://github.com/NVIDIA-NeMo/RL/pull/4521) | `abe512a44f6cd2fc1a7c5620c7564759a5a182e7` |

The Gym branch tip will be newer after committing this handoff refresh. Fetch
the branch tip and use the hashes above only as minimum ancestry checks:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git nemo-gym-multi-harness
cd nemo-gym-multi-harness
git remote add contributor https://github.com/ehosseiniasl/Gym.git
git fetch contributor ehosseiniasl/multi-harness-training-routing
git switch -c ehosseiniasl/multi-harness-training-routing \
  --track contributor/ehosseiniasl/multi-harness-training-routing
git merge-base --is-ancestor 2d0579fea HEAD

cd ..
git clone https://github.com/NVIDIA-NeMo/RL.git nemorl-multi-harness
cd nemorl-multi-harness
git remote add contributor https://github.com/ehosseiniasl/NeMo-RL.git
git fetch contributor ehosseiniasl/multi-harness-training-routing
git switch -c ehosseiniasl/multi-harness-training-routing \
  --track contributor/ehosseiniasl/multi-harness-training-routing
git merge-base --is-ancestor abe512a44f6cd2fc1a7c5620c7564759a5a182e7 HEAD
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

- `nemo_gym/rollout_collection.py`
- `nemo_gym/global_config.py`
- `nemo_gym/train_data_utils.py`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness_enroot.yaml`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_openclaw.yaml`
- `responses_api_agents/anyterminal_agent/configs/anyterminal_pi.yaml`
- `responses_api_agents/opencode_agent/app.py`
- `responses_api_agents/pi_agent/app.py`
- `tests/unit_tests/test_anyterminal_multi_harness.py`

NeMo RL:

- `nemo_rl/data/datasets/response_datasets/nemogym_dataset.py`
- `nemo_rl/environments/nemo_gym.py`
- `nemo_rl/environments/nemo_gym_shards.py`
- `nemo_rl/experience/rollouts.py`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_qwen3_0_6b_single_controller.yaml`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_nano_omni_sync_2n_debug_single_controller.yaml`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_single_controller.yaml`
- `examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_super_omni_sync_8n_single_controller.yaml`
- `tests/unit/environments/test_anyterminal_multi_harness_recipe.py`

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
uv run ruff check responses_api_agents/pi_agent/app.py \
  responses_api_agents/pi_agent/tests/test_app.py \
  tests/unit_tests/test_anyterminal_multi_harness.py
git diff --check
```

Before the Pi fix, the OpenClaw plus multi-harness focused set passed **57/57**.
For commit `2d0579fea`, all eight Pi settings-staging variants and all three
multi-harness profile/fan-out tests pass; Ruff and `git diff --check` also pass.
The complete 55-test Pi-plus-profile invocation reached 45 completed
tests/skips and then hung in the pre-existing
`test_prefixed_responses_preserves_observations_through_fastapi` streaming test,
so do not describe that full invocation as passing. Re-run it with a test
timeout or on the destination compute environment.

From the RL checkout:

```bash
uv run pytest -q tests/unit/environments/test_anyterminal_multi_harness_recipe.py
git diff --check
```

On the source login node, the RL pytest suite could not initialize its autouse
Ray fixture because GCS was unavailable. That is an infrastructure limitation,
not a passing test. The direct config-resolution assertions did pass for the
Qwen, Nano, and Super recipes, including OpenClaw context `15872` and output
limit `4096`. Re-run the real pytest on a compute node with working Ray.

## Implemented OpenClaw prompt reduction under runtime validation

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
target the remaining OpenClaw 2026.6.11 fixed prompt. `contextInjection:
never` makes the runtime provide no bootstrap/context files; `skills: []`
filters every discovered skill, with zero prompt limits as a defensive second
gate; and startup memory is disabled. `localModelLean` is deliberately not
enabled because this OpenClaw release can use it to add a tool-search surface,
while the explicit terminal policy is already stricter.

The config-resolution and merge tests pass in the focused 57-test Gym suite,
and `openclaw config validate --json` reports `valid: true` with no warnings.
A real OpenClaw 2026.6.11 CLI probe against a deterministic local
OpenAI-compatible endpoint also completed with two requests and no overflow.
OpenClaw reported a 15,872-token budget, an estimated 2,073-token prompt,
6,707 system-prompt characters, no injected workspace files, zero skill
prompt characters, and only the `exec` tool. This is positive preflight
evidence, but job `2178307` proved it is not enough: the real OpenClaw rollout
made model calls but returned `Agent couldn't generate a response` for one
sibling and an empty message for the other. One capture chain ended with
`unresolved_parent`. Diagnose CLI transcript parsing and parent-chain handling
before starting another acceptance run.

If Nano still overflows, measure which prompt sections remain before
increasing the advertised context. Do not simply advertise the full 16,384
vLLM boundary: an earlier run proved that leaves no server-side generation
token.

## Validation order

1. Fetch the branch containing both the lean OpenClaw prompt configuration and
   Pi commit `2d0579fea`, then rerun their focused tests.
2. Fix and test the OpenClaw empty/fallback transcript plus unresolved capture
   parent. Do not treat the lean-prompt probe as sufficient.
3. Run the two-node synchronous Nano recipe:
   `grpo_anyterminal_multi_harness_nemotron_nano_omni_sync_2n_debug_single_controller.yaml`.
4. Let the four-row epoch finish naturally. Do not lower or otherwise use
   `grpo.max_num_steps` to truncate it.
5. Check the full local artifacts and W&B tables, not just the process exit
   code or `mask_sample`.
6. After Nano passes, run the requested Super checkpoint with the 8-node sync
   and 16-node async recipes. They may run in parallel after the Nano gate.
7. Add the passing W&B links and score/TMPE/refit summary to both PRs.

The expected full-run shape is:

```text
4 source tasks x 4 harnesses = 16 prompt groups
16 prompt groups x 2 siblings = 32 rollouts
32 rollouts / train batch 8 = 4 optimizer steps
```

Run the bundled auditor after a completed run:

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
6. Four optimizer steps complete and W&B contains the complete Gym result
   tables.
7. Reward and advantages are finite and have non-zero variance. Loss and
   gradient norm are finite, and gradient norm is not identically zero.
8. Policy-to-generation refit completes initially and after every optimizer
   step.

## Diagnostic evidence

| Slurm job | W&B | What it proves | Status |
|---|---|---|---|
| `2178307` | [`rm12vt5c`](https://wandb.ai/adlr/multi-harness-RL/runs/rm12vt5c) | Loaded 16 fan-out groups, completed initial refit, and ran the first 4-harness/2-sibling batch. Exposed Pi context/compaction and OpenClaw transcript/capture failures. | Diagnostic only; cannot pass |
| `2178116` | [`9uirholo`](https://wandb.ai/adlr/multi-harness-RL/runs/9uirholo) | Terminal-only tools and 15,872/4,096 limits were active; OpenClaw still overflowed locally with zero usage. | Failed before step 0 |
| earlier Nano | [`a5y7hxq2`](https://wandb.ai/adlr/multi-harness-RL/runs/a5y7hxq2) | Terminal-only config reached runtime; OpenClaw inherited an 8,192 output reserve and overflowed locally. | Diagnostic only |
| `2177783` | [`z31vasi6`](https://wandb.ai/adlr/multi-harness-RL/runs/z31vasi6) | 15,872 guard removed the prior vLLM HTTP 400, exposing local OpenClaw overflow. | Diagnostic only |
| older Qwen smoke | [`zx1q7m3q`](https://wandb.ai/adlr/multi-harness-RL/runs/zx1q7m3q) | Multi-harness plumbing ran, but reward, advantages, loss, and gradient norm were all zero. | Not learning evidence |

The active Nano run directory is:

```text
/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/validation/anyterminal-p0/nano-omni-sync2-debug-grpo/run-nano-omni-sync2-lean-20261008-1439
```

Its Ray driver log is:

```text
/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/github_repos/nemorl-multi-harness/2178307-logs/ray-driver.log
```

At the last check (15:12 PDT), job `2178307` was still running on two nodes. Its
resolved configuration has W&B online at `adlr/multi-harness-RL`, full Gym
tables enabled, `max_num_steps=1000000`, and four expected optimizer steps. It
had produced all eight first-batch response files, but OpenClaw had one 56-byte
fallback message and one empty message, both at zero usage. Pi had non-empty
output and non-zero usage, but also hit the 16,385-token rejection and invalid
compaction requests. The run predates Pi commit `2d0579fea` and must not count
as validation of that fix.

For comparison, the two zero-usage responses from failed job `2178116` are
under its run directory at
`anyterminal-results/openclaw/configure-git-webserver_*/response.json`.

At the last reliable scheduler check, Super job `2172269` (8-node sync) and
job `2172270` (16-node async) were pending for `Priority`. They launch mutable
worktrees, so their output must not count unless the resolved runtime config
contains the final OpenClaw fix. On a new cluster, submit fresh jobs instead
of reusing these IDs.

## Resume checklist

1. Fetch both branch tips and pass the minimum-commit ancestry checks.
2. Copy/remap the validation bundle, outer container, caches, four task images,
   dataset, and model checkpoint.
3. Restore secrets without printing them and confirm W&B is online at
   `adlr/multi-harness-RL`.
4. Verify both imported package paths and run the focused tests.
5. Confirm the lean OpenClaw and bounded Pi configurations are present in the
   resolved runtime configuration.
6. Reproduce/fix OpenClaw's fallback/empty transcript and unresolved parent.
7. Run Nano sync and audit all 32 rollouts plus four optimizer steps.
8. Run Super 8-node sync and Super 16-node async with the specified Super
   checkpoint.
9. Record per-harness scores, TMPE, reward/advantage/loss/gradient metrics,
   refit timings, and W&B URLs in both PR descriptions.
10. Re-run current PR checks and confirm both local worktrees match their
   remote branches.

Until steps 7 and 8 pass, the correct project status is: **multi-harness fan-out
implemented; Pi context fix committed and focused-tested; OpenClaw real-rollout
transcript/capture still blocked; Nano and Super training validation pending**.
