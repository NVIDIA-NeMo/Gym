# RDKit ES-Step-140 LoRA GRPO

This bundle continues the Inferno RDKit policy after ES Step 140 with native
Megatron LoRA GRPO in NeMo-RL v0.6.0. It is pinned to Gym commit
`1a4912e231bb2795b062f7de97496caaf382c7f6` and the IAD worktree:

```text
/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo
```

All model/container work runs through Slurm. The login node only stages code
and submits jobs. Every allocation copies the bounded Gym checkout to
`/raid/scratch/$USER` before Python starts; Python never imports or executes
from Lustre. Enroot, Hugging Face, UV, W&B, and server virtual-environment
caches are node-local. Checkpoints, logs, model directories, and prepared JSONL
files are persisted on Lustre as artifacts. This no-tool RDKit environment has
no SQLite database. Environments that do use SQLite must copy the database and
its `-wal` and `-shm` companions to `/raid/scratch` before opening it.

## Initialization and training

`merge_es_adapter.py` verifies the exact Inferno adapter SHA-256 and provenance,
then streams the frozen public HF safetensor shards and applies every LoRA
`B @ A * (alpha / rank)` delta directly. It requires all 11,916 adapter tensors
to map exactly once, rejects shape mismatches and non-finite values, preserves
the base tensor dtypes, and publishes the sharded HF directory atomically. This
avoids importing the Nemotron-H implementation merely to add weights. The
preferred GPU gate converts that merged model once and exercises one rollout.

NeMo-RL converts that merged HF model to Megatron and creates a new zero-output
adapter with:

- rank 8, alpha 8, dropout 0;
- Xavier A and zero B initialization;
- `linear_qkv`, `linear_proj`, `linear_fc1`, and `linear_fc2` targets;
- constant `3e-6` LR after 10 warmup steps from `3e-7`.

Production uses 8 pool0 nodes (64 H100 GPUs), 64 prompts × 16 generations per
step, 200 optimizer steps, validation every 5 steps, and 12-hour continuation
jobs. Checkpoints and W&B identity are shared across continuations.

## Validation semantics

NeMo-RL v0.6 does not apply `num_val_generations_per_prompt` in the NeMo-Gym
validation loop. `prepare_direct_data.py` therefore materializes
`test_eval4.jsonl`, which repeats each of the 1,000 held-out prompts four times.
The validation-only baseline refuses to pass unless exactly 4,000 rollouts are
recorded and accuracy lies in `[0.518, 0.578]`. It exits before an optimizer
update and writes `results/es140_lora/baseline_step0.json`.

## Gated launch order

From the worktree, with `WANDB_API_KEY` exported:

```bash
sbatch cluster/rdkit_no_tool_grpo/import_nemo_rl_sqsh_direct.sbatch
sbatch cluster/rdkit_no_tool_grpo/dependency_preflight.sbatch
sbatch cluster/rdkit_no_tool_grpo/build_integration_sqsh.sbatch
sbatch cluster/rdkit_no_tool_grpo/merge_es_adapter.sbatch
sbatch cluster/rdkit_no_tool_grpo/preflight_resource_server.sbatch
cluster/rdkit_no_tool_grpo/submit_conversion_rollout_gate.sh
```

The dependency preflight installs the exact overlay in a disposable node-local
container and imports every Gym, verifier, vLLM-server, PEFT, Torch, and direct
merge module before the longer jobs are submitted. The derived integration
squashfs adds Gym's declared runtime dependencies, `openai==2.6.1`, and
`peft==0.18.1`. No virtual environment or cache is created on Lustre. Each
later job should use an `afterok` dependency on the preceding gate.

The preferred GPU compatibility gate is now the 4-node conversion-and-rollout
job. It runs the normal NeMo-RL setup path, persists the topology-neutral
HF-to-Megatron `iter_0000000` checkpoint, loads the native LoRA policy, starts
vLLM and NeMo-Gym, and scores exactly one held-out rollout. It does not perform
an optimizer update, does not log to W&B, and has a one-hour time limit:

```bash
cluster/rdkit_no_tool_grpo/submit_conversion_rollout_gate.sh
```

The older 64-GPU one-update smoke is retained for debugging but should not be
used as a routine gate because an actual checkpointed training allocation tests
the same training path without discarding useful optimizer work. After the
conversion-and-rollout gate completes:

```bash
START_DEPENDENCY=afterok:<conversion_rollout_job_id> \
  cluster/rdkit_no_tool_grpo/submit_baseline.sh

START_DEPENDENCY=afterok:<baseline_job_id> CHAIN_JOBS=3 \
  cluster/rdkit_no_tool_grpo/submit_chain.sh
```

The production submitter also runs a CPU preflight that verifies the current
Gym commit, data hashes and counts, container stamp, ES adapter, merged-model
manifest, fixed-prompt parity, and the successful 4,000-rollout baseline.

The production W&B ID and name are:

```text
rdkit-nemotron3-nano-grpo-lora-r8-a8-es140-64p16g-i200-lr3e-6-32k-iad-p0-64g
```
