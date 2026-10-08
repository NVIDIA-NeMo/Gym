# Multi-harness training handoff

Last updated: 2026-10-08 13:58 PDT

This file is the short operational handoff for resuming the NeMo Gym + NeMo
RL multi-harness work on another cluster. The longer design, code map, test
sequence, and evidence log are in
[`fern/versions/latest/pages/training-tutorials/multi-harness-training-handoff.mdx`](fern/versions/latest/pages/training-tutorials/multi-harness-training-handoff.mdx).

## Repositories

Use the latest tip of `ehosseiniasl/multi-harness-training-routing` in both
repositories:

- Gym PR: <https://github.com/NVIDIA-NeMo/Gym/pull/4082>
- NeMo RL PR: <https://github.com/NVIDIA-NeMo/RL/pull/4521>
- Verified Gym predecessor: `e86059d9a4ff148458934dffc429ba43a6fe6d36`
- Current NeMo RL head: `85c7997529c6db3601420c6278ccc47877bd4720`

Fetch the branch tip rather than detaching at those hashes, because this
handoff commit is newer than the recorded Gym predecessor.

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git nemo-gym-multi-harness
cd nemo-gym-multi-harness
git remote add contributor https://github.com/ehosseiniasl/Gym.git
git fetch contributor ehosseiniasl/multi-harness-training-routing
git switch -c ehosseiniasl/multi-harness-training-routing \
  --track contributor/ehosseiniasl/multi-harness-training-routing

cd ..
git clone https://github.com/NVIDIA-NeMo/RL.git nemorl-multi-harness
cd nemorl-multi-harness
git remote add contributor https://github.com/ehosseiniasl/NeMo-RL.git
git fetch contributor ehosseiniasl/multi-harness-training-routing
git switch -c ehosseiniasl/multi-harness-training-routing \
  --track contributor/ehosseiniasl/multi-harness-training-routing
```

## Current blocker

The fan-out implementation is pushed and the four P0 harnesses are OpenCode,
OpenClaw, Pi, and Hermes. Each source task is expanded to all four harnesses
before GRPO sibling generation.

The latest Nano sync job (`2177783`, W&B
[`z31vasi6`](https://wandb.ai/adlr/multi-harness-RL/runs/z31vasi6)) dispatched
all four harness groups. The 15,872-token OpenClaw guard removed the prior
vLLM HTTP 400, but OpenClaw rejected both siblings locally because its broad
`coding` tool profile made the initial prompt too large. Both responses had
zero token usage and token capture rejected them as
`rollout_failed:no_records`. No optimizer step completed.

Do not treat `mask_sample=false` as success in this run. The Gym metadata had
no timeout or sandbox error, but the output was only OpenClaw's local context
overflow message and contained no policy tokens.

## First change on the new cluster

In
`responses_api_agents/anyterminal_agent/configs/anyterminal_openclaw.yaml`,
replace the `coding` profile with a terminal-only policy:

```yaml
openclaw_config:
  agents:
    defaults:
      workspace: "."
  tools:
    profile: minimal
    alsoAllow:
      - exec
    deny:
      - session_status
```

This should leave only OpenClaw's `exec` tool after Gym adds its required
headless `message` deny. It is the proposed fix, not yet runtime validated.
Add assertions for the resolved AnyTerminal config and OpenClaw's final merged
config, then run:

```bash
uv run pytest -q tests/unit_tests/test_anyterminal_multi_harness.py
uv run pytest -q responses_api_agents/openclaw_agent/tests/test_app.py
git diff --check
```

Commit and push that fix to Gym PR #4082 before rerunning. The RL recipe
already advertises a 15,872-token OpenClaw context at head `85c79975`.

## Environment and inputs

Set these in the driver and every Ray worker:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
export WANDB_MODE=online
export WANDB_ENTITY=adlr
export WANDB_PROJECT=multi-harness-RL
```

The source cluster loaded secrets from
`/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env`.
Move credentials through a secure cluster-approved channel; never commit or
print them.

Verify that Python imports the two PR checkouts, not installed packages:

```bash
python -c 'import nemo_gym, nemo_rl; print(nemo_gym.__file__); print(nemo_rl.__file__)'
```

Source-cluster inputs:

- Data: `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/validation/anyterminal-p0/collated/train.jsonl`
- Super checkpoint: `/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp`
- W&B project: <https://wandb.ai/adlr/multi-harness-RL>

The four Terminal-Bench 2.1 tasks are `configure-git-webserver`, `fix-git`,
`log-summary-date-ranges`, and `modernize-scientific-stack`. Preserve
`task_source: anyterminal_multi_harness` on every row. Confirm task assets,
verifiers, and Enroot images are visible from every node.

## Validation order

1. Run the Nano 3 Omni two-node synchronous debug recipe:
   `examples/nemo_gym/grpo_anyterminal_multi_harness_nemotron_nano_omni_sync_2n_debug_single_controller.yaml`.
2. Require both OpenClaw siblings to reach vLLM and produce complete generated
   token IDs and log probabilities. Reject local error text even if the Gym
   verifier fields look clean.
3. Let the four-task epoch finish naturally; do not cap
   `grpo.max_num_steps`.
4. After Nano passes, run the requested Super checkpoint with the 8-node sync
   recipe and the 16-node async recipe. Keep W&B and full Gym result tables
   enabled.

Expected full-run shape:

```text
4 source tasks x 4 harnesses = 16 prompt groups
16 prompt groups x 2 siblings = 32 rollouts
32 rollouts / train batch 8 = 4 optimizer steps
```

Require all 32 outputs to be non-empty and unmasked with no agent, container,
or sandbox failure; complete token capture; finite TMPE; four optimizer steps;
initial and post-step generation refits; and finite reward, advantage, loss,
and gradient metrics with non-zero reward/advantage variance and a non-zero
gradient norm.

## Source-cluster state and artifacts

- `2177783`: Nano sync, failed after 10m15s on the OpenClaw issue above.
- `2172269`: 8-node Super sync, still pending for priority at handoff.
- `2172270`: 16-node Super async, still pending for priority at handoff.

The pending jobs launch from mutable worktrees and do not contain the proposed
OpenClaw fix. Do not let them consume GPUs without deliberately updating or
replacing them.

Useful source paths:

- Launchers: `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/validation/anyterminal-p0/`
- Latest run: `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/validation/anyterminal-p0/nano-omni-sync2-debug-grpo/run-nano-omni-sync2-debug-20261008-134546`
- Ray log: `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/github_repos/nemorl-multi-harness/2177783-logs/ray-driver.log`
- Failed responses: `anyterminal-results/openclaw/configure-git-webserver_*/response.json`

After successful Nano, Super sync, and Super async runs, add the W&B links and
per-harness reward/TMPE/refit evidence to both PR descriptions and rerun their
checks.
