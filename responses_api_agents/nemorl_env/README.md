# NeMo RL environment

A native Gym agent for evaluating agent-authored post-training code.

1. Claude Code edits a recipe and pinned NeMo RL/Gym sources in a CPU sandbox
   (60 minutes, 200 turns, 256K context). Core loss and data-processing changes are accepted.
2. A one-GPU sandbox applies the patch and trains Qwen2.5-1.5B-Instruct with
   GRPO: 8 prompts × 8 responses. Training, authored builds, and checkpoint export
   share a hard 60-minute budget after pristine setup.
3. A fresh, unpatched GPU sandbox serves tensor weights with vLLM and runs
   `gym eval run` on 32 held-out math examples and 30 AIME25 problems with boxed
   answers. Generation gets the model's 32K context minus the prompt tokens.

Reward is `(math accuracy + AIME25 accuracy) / 2`. Held-out data and evaluator
credentials never enter author/training sandboxes. Infrastructure failures are
errors, not zero rewards; an empty patch receives zero. Outer training/token
capture is not enabled. Full end-to-end evaluation validation is still pending.

## Setup

Configure `policy_model` with an OpenAI-compatible author endpoint. Set:

- `OPENSANDBOX_CELL3_DOMAIN` and `OPENSANDBOX_CELL3_API_KEY` for CPU authoring.
- `OPENSANDBOX_CELL4_DOMAIN` and `OPENSANDBOX_CELL4_API_KEY` for GPU execution.
- `NEMORL_ENV_AUTHOR_IMAGE`, `NEMORL_ENV_REGISTRY_USER`, and `NEMORL_ENV_REGISTRY_PASSWORD`
  for the author image, which must provide Python at `/agent_deps_mount/bin/python`, uv, and Git.

Prepare AIME25 with `gym eval prepare --benchmark aime25`.
`app.py` orchestrates the sandboxes; `author_worker.py` runs Claude Code.
`task_environment/` contains the recipe, training data, runner, and evaluator.
`data/train.jsonl` contains eight author prompts; `data/math_eval.jsonl` is held out.
Checkpoint transfers are serialized and buffered in memory; provision driver
memory for one transfer and disk for queued checkpoints.

Math data: the first 512 training rows of NVIDIA's
[OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
(CC-BY-4.0), split 480/32 with `problem`/`expected_answer` renamed to `input`/`output`.
