# Nemotron-3 Nano RDKit ES-Step-140 LoRA GRPO Plan

Last updated: 2026-08-20

This experiment adapts the pinned full-rank RDKit GRPO configuration to a
LoRA-only continuation initialized from the best Inferno ES checkpoint at Step
140. It deliberately preserves task, sampling, batch, sequence, and validation
semantics while changing only initialization, trainable parameters, and
checkpoint retention.

## Provenance

| Item | Value |
| --- | --- |
| Gym base commit | `1a4912e231bb2795b062f7de97496caaf382c7f6` |
| NeMo-RL image | `nvidia/nemo-rl:v0.6.0` |
| HF base | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Inferno selection | post-Step 140 (`update_count=140`) |
| ES adapter SHA-256 | `e281afb74b78d5ef337c233c1b1faeb42ac72208ca76ed4b5563e1ca4adb3de4` |
| ES adapter | rank 8, alpha 8, BF16 |
| Train data | 1,024 prompts; SHA-256 `7acc2e5b...00cf` |
| Test data | 1,000 prompts; SHA-256 `b02ee40a...3303` |

The source PEFT adapter cannot be mapped naively onto Megatron's fused QKV and
MoE module layout. Instead, it is merged into the frozen HF checkpoint using
PEFT's safe merge. NeMo-RL then converts the resulting model and initializes a
fresh native Megatron adapter whose output is exactly zero at startup. This
preserves the ES-selected policy function without inventing an adapter mapping.

## Controlled configuration

| Dimension | Setting |
| --- | --- |
| Allocation | 8 pool0 nodes × 8 H100 GPUs |
| Training parallelism | TP2, PP2, CP8, EP8 |
| Generation | colocated vLLM, TP4, 0.5 memory utilization |
| Train sample batch | 64 unique prompts × 16 rollouts = 1,024 trajectories |
| Optimizer steps | 200 |
| Evaluation | every 5 steps, test only |
| Validation sample batch | 1,000 prompts × 4 stochastic rollouts = 4,000 |
| Completion / total limits | 32,768 / 65,536 tokens |
| Sampling | temperature 1, top-p 1, thinking enabled, `nano_v3` parser |
| Task | one-turn direct RDKit answer, no tools |
| Optimizer | distributed Adam, betas 0.9/0.999, epsilon 1e-8 |
| Schedule | 10-step warmup from 3e-7, then constant 3e-6 |
| Regularization | weight decay 0, grad clip 1 |
| New LoRA | rank 8, alpha 8, dropout 0, BF16-following dtype |
| Targets | `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2` |
| Initialization | Xavier A, zero B |
| Checkpointing | every 5 steps, optimizer included, retain top 10 |

The new adapter excludes Mamba input/output projections and the router. The
four targets match NeMo-RL's documented native Megatron LoRA names and avoid the
known no-gradient Mamba output-projection path.

## Gates

1. Import the v0.6.0 container on a CPU Slurm node.
2. Create a small container-compatible Gym/OpenAI/PEFT integration overlay on a
   CPU Slurm node, then merge the validated ES adapter on CPU and write a file-hashed
   manifest atomically.
3. Install and exercise the pinned RDKit resource server inside that container.
4. Require top-1 identity on 16 fixed prompts and minimum full-logit cosine
   similarity of 0.9999 between base+PEFT and merged HF inference.
5. Run one complete 64-GPU optimizer-step smoke using 64×16 trajectories.
6. Run a validation-only 64-GPU baseline with no optimizer update. Require
   exactly 4,000 scored rollouts and accuracy in `[0.518, 0.578]`.
7. Submit three 12-hour production jobs with `afterok` dependencies, shared
   checkpoint directory, and a single resumed W&B identity.

The explicit repeated validation file is necessary because NeMo-RL v0.6's
NeMo-Gym validation path does not use the configured
`num_val_generations_per_prompt`. Without this correction, the nominal 4×
setting silently produces only one rollout per test row.

## Success and rollback

Successful launch means all gates pass, W&B contains the zero-update baseline,
the first production job restores the merged ES function and creates only new
LoRA optimizer state, and a checkpoint is durable by Step 5 or the timeout
deadline. Every production continuation depends on successful completion of
the prior one.

Rollback is non-destructive: cancel pending continuation jobs and retain the
separate worktree, merged HF artifact, W&B run, and LoRA checkpoint directory.
The original Gym source checkout and Inferno adapter are never modified in
place; the merged model is published as a new ignored directory under Gym's
`artifacts/` area.
