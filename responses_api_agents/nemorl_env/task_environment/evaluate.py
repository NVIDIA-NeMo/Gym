# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


repo = Path(os.environ.get("NEMORL_ROOT", "/workspace/NeMo-RL"))
work = Path("/testbed")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
model_config = AutoConfig.from_pretrained(model_name, revision=model_revision, trust_remote_code=False)
with torch.device("meta"):
    reference = AutoModelForCausalLM.from_config(model_config, trust_remote_code=False, attn_implementation="eager")

state = load_file(work / "model.safetensors", device="cpu")
if any(not tensor.is_floating_point() or not tensor.isfinite().all().item() for tensor in state.values()):
    raise ValueError("checkpoint must contain finite floating-point weights")
if model_config.tie_word_embeddings and "lm_head.weight" not in state:
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
reference.load_state_dict(state, strict=True, assign=True)
model_dir = work / "eval_model"
model_dir.mkdir()
save_file(state, str(model_dir / "model.safetensors"), metadata={"format": "pt"})
model_config.save_pretrained(model_dir)
AutoTokenizer.from_pretrained(model_name, revision=model_revision, trust_remote_code=False).save_pretrained(model_dir)
del state, reference


cfg = OmegaConf.load(repo / "examples/configs/grpo_math_1B.yaml")
cfg.grpo.num_prompts_per_step = 8
cfg.grpo.num_generations_per_prompt = 8
cfg.grpo.max_num_steps = 1
cfg.grpo.val_period = 0
cfg.grpo.val_at_start = True
cfg.grpo.val_at_end = False
cfg.grpo.stop_at_validation_metric = "accuracy"
cfg.grpo.stop_at_validation_threshold = 0.0
cfg.grpo.max_val_samples = 62
cfg.grpo.val_batch_size = 2
cfg.grpo.val_num_generations_per_prompt = 1
cfg.policy.model_name = str(model_dir)
cfg.policy.tokenizer.name = str(model_dir)
cfg.policy.train_global_batch_size = 64
cfg.policy.train_micro_batch_size = 1
cfg.policy.logprob_batch_size = 1
cfg.policy.dtensor_cfg._v2 = False
cfg.policy.dtensor_cfg.activation_checkpointing = True
cfg.policy.max_total_sequence_length = 32768
cfg.policy.generation.max_new_tokens = 32768
cfg.policy.generation.vllm_cfg.max_model_len = 32768
cfg.policy.generation.vllm_cfg.gpu_memory_utilization = 0.2
cfg.policy.generation.vllm_cfg.enforce_eager = True
cfg.policy.generation.temperature = cfg.policy.generation.val_temperature = 0.0
cfg.policy.generation.top_p = cfg.policy.generation.val_top_p = 1.0
cfg.policy.generation.top_k = cfg.policy.generation.val_top_k = -1
cfg.loss_fn.reference_policy_kl_penalty = 0.0
cfg.loss_fn.force_on_policy_ratio = True
cfg.data.validation = [
    {
        "dataset_name": "ResponseDataset",
        "data_path": "/root/math_eval.jsonl",
        "input_key": "input",
        "output_key": "output",
        "split_validation_size": 0,
        "prompt_file": "examples/prompts/cot.txt",
    },
    {
        "dataset_name": "ResponseDataset",
        "data_path": "/root/aime25.jsonl",
        "input_key": "question",
        "output_key": "expected_answer",
        "split_validation_size": 0,
        "prompt_file": "/testbed/cot_prompt.txt",
    },
]

cfg.data.train = cfg.data.validation[0]
cfg.checkpointing.enabled = False
cfg.checkpointing.checkpoint_dir = "/testbed/eval_checkpoints"
cfg.checkpointing.checkpoint_must_save_by = None
cfg.checkpointing.model_save_format = None
cfg.checkpointing.save_optimizer = False
cfg.cluster.gpus_per_node = 1
cfg.cluster.num_nodes = 1
cfg.logger.log_dir = "/testbed/results/logs"
cfg.logger.wandb_enabled = "WANDB_API_KEY" in os.environ
cfg.logger.wandb.project = os.environ.get("WANDB_PROJECT", "cmunley-rlenv")
cfg.logger.wandb.name = os.environ.get("WANDB_NAME", "nemorl-env-inner-eval")
for path, count in (("/root/math_eval.jsonl", 32), ("/root/aime25.jsonl", 30)):
    if len(Path(path).read_text().splitlines()) != count:
        raise ValueError(f"evaluation dataset must contain exactly {count} rows: {path}")
OmegaConf.save(cfg, work / "resolved_aime25.yaml")
eval_log = work / "aime25.log"
with eval_log.open("w") as stream:
    subprocess.run(
        [sys.executable, str(repo / "examples/run_grpo.py"), "--config", str(work / "resolved_aime25.yaml")],
        cwd=repo,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
    )
sample_counts = re.findall(r"• Samples processed:\s*([0-9]+)", eval_log.read_text())
if not sample_counts or int(sample_counts[-1]) != 62:
    raise RuntimeError("evaluation did not report all 32 math and 30 AIME25 samples")
rollout_file = max(work.glob("results/logs/exp_*/val_data_step*.jsonl"), key=lambda path: path.stat().st_mtime)
rows = [json.loads(line) for line in rollout_file.read_text().splitlines()]
rewards = [float(row["rewards"][0]) for row in rows]
if len(rewards) != 62 or any(reward not in (0.0, 1.0) for reward in rewards):
    raise RuntimeError("evaluation must contain exactly 62 binary math rewards")
math_accuracy = sum(rewards[:32]) / 32
aime_accuracy = sum(rewards[32:]) / 30
wandb_url = ""
if cfg.logger.wandb_enabled:
    import wandb

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=cfg.logger.wandb.project,
        name=f"{cfg.logger.wandb.name}-rollouts",
        job_type="evaluation",
    )
    run.log(
        {
            "aime25/rollouts": wandb.Table(
                columns=["idx", "reward", "conversation"],
                data=[[row["idx"], row["rewards"], json.dumps(row["content"])] for row in rows[32:]],
            ),
            "math_eval/accuracy": math_accuracy,
            "aime25/accuracy": aime_accuracy,
        }
    )
    wandb_url = run.url
    run.finish()
print(
    "NEMORL_ENV_RESULT="
    + json.dumps(
        {
            "reward": (math_accuracy + aime_accuracy) / 2,
            "math_eval_exact": math_accuracy,
            "aime25_exact": aime_accuracy,
            "completed": 1,
            "wandb_url": wandb_url,
        }
    )
)
