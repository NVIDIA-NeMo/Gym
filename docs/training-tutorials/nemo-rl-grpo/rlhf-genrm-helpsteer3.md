(training-nemo-rl-grpo-rlhf-genrm-helpsteer3)=

# RLHF-style GRPO with GenRM Compare and HelpSteer3

This recipe connects **NeMo RL** (GRPO), **NeMo Gym**, and the GenRM compare resources server ({doc}`/resources-server/index`) so you can improve a chat policy using a **generative reward model** instead of hand-written verifiers. Prompts come from the public [**HelpSteer3**](https://huggingface.co/datasets/nvidia/HelpSteer3) preference subset; the policy you tune can be [**Nemotron 3 Super**](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) or [**Nemotron 3 Nano**](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) on Hugging Face.

:::{card}

**Goal**: Prepare HelpSteer3 as NeMo Gym JSONL, wire `genrm_compare.yaml` into a GRPO training config, and evaluate on **Arena-Hard v2**–style benchmarks.

**Time**: ~1–2 hours to read and prepare data; training time depends on your cluster (large MoE runs match your existing NeMo RL recipes).

^^^

**In this guide, you will**:

1. Convert HelpSteer3 **preference** conversations into NeMo Gym JSONL (same shape as `resources_servers/genrm_compare/data/example.jsonl`).
2. Point NeMo RL’s data paths at those files and load `resources_servers/genrm_compare/configs/genrm_compare.yaml` in `env.nemo_gym.config_paths`.
3. Align `num_rollouts_per_prompt` / `num_generations_per_prompt` with GRPO cohort sizes.
4. Run **Arena-Hard-Auto** to compare base vs post-training checkpoints.

:::

> **Scope:** NeMo RL launch commands and full cluster YAMLs live in the [NeMo RL](https://github.com/NVIDIA-NeMo/RL) repository and your job scripts. This page documents the **NeMo Gym** side (data + environment wiring) and how it fits GRPO RLHF workflows.

---

## Prerequisites

- Completed or skimmed the {ref}`NeMo RL GRPO tutorial <training-nemo-rl-grpo-index>` so you know how `data` and `env.nemo_gym` work in your training YAML.
- **NeMo Gym** installed ({doc}`/get-started/detailed-setup`) with Hugging Face dependencies (`datasets` is included in the Gym dev install).
- **Hardware**: Same class of cluster you would use for Nemotron-3 GRPO (policy + vLLM + GenRM judge). Large models require multi-GPU / multi-node settings from your NeMo RL recipe.
- Optional: read {doc}`/data/index` for the JSONL schema and {doc}`/data/prepare-validate` for `ng_prepare_data`.

---

## 1. How the pieces fit together

| Component | Role |
|-----------|------|
| **Policy model** | Nemotron 3 Super or Nano (HF checkpoint path in NeMo RL `policy.model_name` or your launcher’s equivalent). |
| **Training prompts** | Rows from HelpSteer3 **preference**: multi-turn `context` becomes `responses_create_params.input`. |
| **Reward** | GenRM compares **multiple rollouts per prompt** (GRPO group) via `genrm_compare` → pairwise JSON scores → aggregated rewards. |
| **Human preference columns** | `response1` / `response2` / `overall_preference` are **not** used during GRPO; the environment generates new answers and judges them with GenRM. |

NeMo RL’s dataloader reads **static JSONL** for `NemoGymDataset`. There is no built-in “stream HelpSteer3 inside the trainer” hook in NeMo Gym today, so conversion is done **once** (or in CI) with the script below—not inside the training loop.

---

## 2. Convert HelpSteer3 to NeMo Gym JSONL

Use the converter maintained next to the GenRM compare server:

```bash
cd /path/to/nemo-gym   # repository root

uv run python resources_servers/genrm_compare/scripts/helpsteer3_to_nemo_gym_jsonl.py \
  --output-dir data/helpsteer3_gym
```

This writes `train.jsonl` and `validation.jsonl`. Each line looks like the shipped example (OpenAI-style `input`, empty `tools`, `agent_ref` → `genrm_simple_agent`):

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "..."}],
    "tools": [],
    "parallel_tool_calls": false
  },
  "agent_ref": {"type": "responses_api_agents", "name": "genrm_simple_agent"},
  "dataset": "helpsteer3_preference"
}
```

**Details the script applies:**

- **Subset**: `config_name=preference` (multi-turn chat before the pairwise human labels).
- **GenRM requirement**: The compare server expects the conversation history to end with a **user** turn. The script **truncates** after the last user message.
- **HTML entities** in text (e.g. `&lt;`) are unescaped to literal `<` for cleaner prompts.

**Options:**

| Flag | Purpose |
|------|---------|
| `--max-samples N` | Write at most `N` rows **per split** (debug). |
| `--hf-token` | Token if access is restricted. |
| `--agent-ref-name` | Must match the `responses_api_agents` block name merged from `genrm_compare.yaml` (default `genrm_simple_agent`). |

Validate with {doc}`/data/prepare-validate` once your NeMo Gym `env.yaml` points at a policy server.

---

## 3. Wire `genrm_compare.yaml` in NeMo RL

Load the GenRM stack by appending these paths to `env.nemo_gym.config_paths` (order can matter for overrides; follow your NeMo RL template):

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
      - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
      - resources_servers/genrm_compare/configs/genrm_compare.yaml
      - responses_api_models/genrm_model/configs/genrm_model.yaml
```

From `genrm_compare.yaml`, the important blocks are:

- **`genrm_compare_resources_server`** → `/compare` pairwise judging.
- **`genrm_model`** → vLLM (or compatible) server for the GenRM; set `genrm_model_name` in the environment to a checkpoint whose template supports `response_1` / `response_2` (and optionally `principle` if you enable `use_principle`). See `resources_servers/genrm_compare/README.md` for compatible models.
- **`genrm_simple_agent`** → agent that uses the policy for rollouts and GenRM for rewards.

Override resource server settings to match GRPO **group size**, e.g.:

```yaml
nemo_gym:
  genrm_compare_resources_server:
    resources_servers:
      genrm_compare:
        num_rollouts_per_prompt: ${grpo.num_generations_per_prompt}
```

Keep **`num_rollouts_per_prompt`** and **`grpo.num_generations_per_prompt`** equal so cohorts flush correctly (see `resources_servers/genrm_compare/README.md`).

### Example NeMo RL config (Nemotron 3 Super + GenRM + HelpSteer3)

A **pipeclean-scale** NeMo RL YAML that matches the layout of large-cluster GRPO runs (async GRPO, Megatron MoE policy, non-colocated vLLM rollouts, separate GenRM judge with `tensor_parallel_size: 4`) lives in the Gym repo:

`examples/nemo_rl/grpo_nemotron3_super_genrm_helpsteer3_pipeclean.yaml`

It sets **`policy.model_name`** to the Hugging Face checkpoint [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16). Copy or compose-config it from your NeMo RL job (paths are relative to the Gym tree bundled with your run).

**You still must set at launch** (or override in Hydra):

| Item | Notes |
|------|--------|
| **`data.train.data_path` / `data.validation.data_path`** | Paths to JSONL from the HelpSteer3 conversion step (e.g. `data/helpsteer3_gym/train.jsonl`). |
| **`env.nemo_gym.genrm_model.responses_api_models.genrm_model.model`** | GenRM checkpoint (HF id or local path); `null` in the example. |
| **`genrm_model_name`** | If your launcher uses this env var instead of inline YAML, keep it consistent with the GenRM `model` field. |
| **`cluster` / GPU counts** | The example keeps a 64-node × 4 GPU skeleton; your submit script should override to match allocation. |
| **`NRL_MAX_STEPS`** | Optional; shortens the default `grpo.max_num_steps: 10` for smoke tests. |

**Batch consistency:** `policy.train_global_batch_size` should match `grpo.num_prompts_per_step × grpo.num_generations_per_prompt` (here `16 × 4 = 64`), same as in the reference pipeclean recipe.

**Principle-guided judging:** The example has `use_principle: true` and a long `default_principle`. Use this only if your GenRM supports the `principle` role; otherwise set `use_principle: false` (see the compatibility table in `resources_servers/genrm_compare/README.md`).

**Nemotron 3 Nano:** Swap in [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) for `policy.model_name` and **resize** Megatron/vLLM parallel sizes to your Nano recipe—do not reuse the Super MoE EP/TP numbers without checking the NeMo RL Nano launch config.

---

## 4. Evaluation: Arena-Hard v2

To show improvement after GRPO, run an **open-ended** benchmark correlated with Chatbot Arena:

- **Tooling**: [lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto) (Arena-Hard-Auto).
- **Data**: [lmarena-ai/arena-hard-auto](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto) on Hugging Face (includes Arena-Hard v2 prompts).

**Suggested workflow:**

1. Export **baseline** and **GRPO** checkpoints behind the same inference stack (vLLM or your production server) with identical decoding settings.
2. Run Arena-Hard-Auto twice, only swapping the model endpoint or checkpoint path.
3. Compare aggregate win rates / judged scores the pipeline prints (and publish alongside training logs).

This is the same class of evaluation HelpSteer3’s authors report for instruction-following quality (see the [HelpSteer3 dataset card](https://huggingface.co/datasets/nvidia/HelpSteer3)); your GRPO + GenRM run targets **policy improvement** rather than training a standalone reward model.

---

## 5. Troubleshooting

| Symptom | Check |
|---------|--------|
| No reward variance | GenRM JSON parse failures → see `default_score` / logs in `genrm_compare`; confirm `genrm_model_name` loads and outputs `score_1`, `score_2`, `ranking`. |
| Hangs or partial cohorts | `num_rollouts_per_prompt` ≠ `num_generations_per_prompt`. |
| Invalid template errors | Policy or GenRM chat template mismatch; Nemotron-3 models need their matching tokenizer / template flags in NeMo RL. |
| Empty JSONL after conversion | Rows whose `context` does not contain a **user** turn after trimming are skipped; try another subset or inspect raw rows. |

---

## See also

- `examples/nemo_rl/grpo_nemotron3_super_genrm_helpsteer3_pipeclean.yaml` — NeMo RL example recipe (Nemotron 3 Super + GenRM + HelpSteer3).
- `resources_servers/genrm_compare/README.md` — GenRM compare API and configuration.
- {doc}`Gym configuration <gym-configuration>` — `env.nemo_gym` structure in NeMo RL.
- {doc}`/model-recipes/nemotron-3-nano` and {doc}`/model-recipes/nemotron-3-super` — cluster launch patterns for Nemotron-3.

```{button-ref} training-nemo-rl-grpo-index
:color: secondary
:outline:
:ref-type: ref

← Back to NeMo RL GRPO overview
```
