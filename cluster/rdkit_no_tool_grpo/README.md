# RDKit ES-Step-140 LoRA GRPO

This bundle continues the Inferno RDKit policy after ES Step 140 with native
Megatron LoRA GRPO in NeMo-RL v0.6.0. It is pinned to Gym commit
`1a4912e231bb2795b062f7de97496caaf382c7f6` and the IAD worktree:

```text
/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo
```

All model/container work runs through Slurm. The login node only stages code
and submits jobs.

## Initialization and training

`merge_es_adapter.py` verifies the exact Inferno adapter SHA-256 and provenance,
merges it into the frozen public HF base in BF16, and publishes a sharded HF
directory atomically. `validate_merged_model.py` then compares logits on 16
fixed prompts between base+PEFT and the merged model.

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
sbatch cluster/rdkit_no_tool_grpo/setup_integration_venv.sbatch
sbatch cluster/rdkit_no_tool_grpo/merge_es_adapter.sbatch
sbatch cluster/rdkit_no_tool_grpo/preflight_resource_server.sbatch
sbatch cluster/rdkit_no_tool_grpo/validate_merged_model.sbatch
cluster/rdkit_no_tool_grpo/submit_smoke.sh
```

The integration venv is a small `--system-site-packages` overlay because the
NeMo-RL v0.6 image lacks both Hugging Face PEFT and the dependencies needed to
import the pinned Gym checkout. It installs Gym editable without resolving its
broad dependency set, pins only `openai==2.6.1` and `peft==0.17.1`, and reuses
the image's Ray, Torch, Transformers, Accelerate, and Safetensors packages.
Each later job should use an `afterok` dependency on the preceding gate. After
the one-update 64-GPU smoke completes:

```bash
START_DEPENDENCY=afterok:<smoke_job_id> \
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
