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
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


repo = Path(os.environ.get("NEMORL_ROOT", "/testbed/NeMo-RL"))
work = Path("/testbed")
seconds = int(os.environ.get("INNER_TRAIN_SECONDS", "3600"))
base = OmegaConf.load(repo / "examples/configs/grpo_math_1B.yaml")
authored = OmegaConf.load(work / "recipe.yaml")
cfg = OmegaConf.merge(base, authored)


cfg.data = base.data
cfg.env = base.env
cfg.grpo.num_prompts_per_step = 8
cfg.grpo.num_generations_per_prompt = 8
cfg.grpo.max_num_steps = 1_000_000
cfg.grpo.max_num_epochs = 1_000_000
cfg.grpo.val_at_start = False
cfg.grpo.val_at_end = False
cfg.grpo.val_period = 0
cfg.policy.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cfg.policy.tokenizer.name = cfg.policy.model_name
cfg.policy.train_global_batch_size = 64
cfg.data.train = OmegaConf.create(
    {
        "dataset_name": "ResponseDataset",
        "data_path": "/testbed/train_math.jsonl",
        "input_key": "input",
        "output_key": "output",
        "split_validation_size": 0,
        "seed": 42,
    }
)
cfg.data.validation = None


cfg.policy.train_micro_batch_size = 1
cfg.policy.logprob_batch_size = 1
cfg.policy.dtensor_cfg._v2 = False

cfg.policy.generation.val_temperature = cfg.policy.generation.temperature
cfg.policy.generation.val_top_p = cfg.policy.generation.top_p
cfg.policy.generation.val_top_k = cfg.policy.generation.top_k
cfg.policy.generation.vllm_cfg.gpu_memory_utilization = 0.2
cfg.loss_fn.force_on_policy_ratio = True
cfg.checkpointing.enabled = True
cfg.checkpointing.checkpoint_dir = "/testbed/results/grpo"
cfg.checkpointing.save_period = 1_000_000

hours, remainder = divmod(seconds, 3600)
minutes, remainder = divmod(remainder, 60)
cfg.checkpointing.checkpoint_must_save_by = f"00:{hours:02d}:{minutes:02d}:{remainder:02d}"
cfg.checkpointing.metric_name = None
cfg.checkpointing.keep_top_k = 1
cfg.checkpointing.model_save_format = None
cfg.checkpointing.save_consolidated = False
cfg.checkpointing.save_optimizer = False
cfg.cluster.gpus_per_node = 1
cfg.cluster.num_nodes = 1
cfg.logger.wandb_enabled = False
cfg.logger.log_dir = "/testbed/results/logs"
OmegaConf.save(cfg, work / "resolved_grpo.yaml")

train_log = work / "train.log"
train_command = [sys.executable, str(repo / "examples/run_grpo.py"), "--config", str(work / "resolved_grpo.yaml")]
if "INNER_TRAIN_STARTED_AT" in os.environ:
    train_command[1:1] = [
        "-c",
        """import os, runpy, sys
from nemo_rl.utils.timer import TimeoutChecker
original_init = TimeoutChecker.__init__
def anchored_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.start_time = float(os.environ["INNER_TRAIN_STARTED_AT"])
TimeoutChecker.__init__ = anchored_init
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
""",
    ]
with train_log.open("w") as stream:
    subprocess.run(
        train_command,
        cwd=repo,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
    )
train_rewards = re.findall(r"• Avg Reward:\s*([0-9]+(?:\.[0-9]+)?)", train_log.read_text())
if not train_rewards:
    raise RuntimeError("training did not report an average reward")
train_reward = float(train_rewards[-1])

checkpoints = sorted((work / "results/grpo").glob("step_*"), key=lambda p: int(p.name.split("_")[-1]))
if not checkpoints:
    raise RuntimeError("NeMo-RL exited without a checkpoint")
checkpoint = checkpoints[-1]
steps = int(checkpoint.name.split("_")[-1])


weights_path = work / "export.pt"
dcp_to_torch_save(str(checkpoint / "policy/weights"), str(weights_path))
state = torch.load(weights_path, map_location="cpu", weights_only=True)
if set(state) == {"model"}:
    state = state["model"]
save_file(
    {
        name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
        for name, tensor in state.items()
    },
    str(work / "model.safetensors"),
    metadata={"format": "pt"},
)
weights_path.unlink()
print(
    "NEMORL_ENV_TRAIN_RESULT="
    + json.dumps(
        {
            "train_reward": train_reward,
            "inner_steps": steps,
        }
    )
)
