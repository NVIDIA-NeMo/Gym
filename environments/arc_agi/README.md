# ARC-AGI

The Abstraction and Reasoning Corpus for Artificial General Intelligence ([ARC-AGI-1](https://github.com/fchollet/ARC-AGI) and [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)) is a benchmark for general reasoning. Each task provides several input-output grid pairs; the system must infer the underlying transformation and apply it to a new input.

The commands below use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with tensor parallelism set to 1. The model contains about 58.8 GiB of BF16 weights, in addition to runtime and KV-cache memory. Use an 80 GB-class CUDA GPU for this exact configuration; a 32 GB GPU is insufficient.

## Install Gym once

Run these commands from the parent directory of the Gym checkout:

```bash
cd Gym/
uv venv
source .venv/bin/activate
uv sync
```

Activate this environment in every new terminal used below. You do not need to rerun `uv sync` for every evaluation.

## Install and launch the local vLLM server (terminal 1)

For a checkout-based run, install vLLM once in the active environment:

```bash
uv pip install -U "vllm>=0.12.0"
```

The published NeMo-Gym container already includes vLLM, but intentionally omits codec-bearing packages. vLLM 0.24 imports `torchvision` during kernel warmup for this configuration, so container users must restore the image's pinned optional dependencies before launching the server:

```bash
bash docker/install_codec_deps.sh
```

The script is idempotent. On later checkout runs, skip the vLLM installation when `vllm --version` reports version 0.12.0 or newer. The model weights are also reused from the Hugging Face cache after the first download.

Then launch the server from the Gym repository root:

```bash

if [ ! -f nano_v3_reasoning_parser.py ]; then
  hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    nano_v3_reasoning_parser.py --local-dir .
fi

vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 10240 \
  --trust-remote-code \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

Keep `nano_v3_reasoning_parser.py` in the Gym repository root and reuse it on later runs. The conditional command downloads it only when it is absent. `hf download` works anonymously for this public file and automatically uses an existing `HF_TOKEN` or `hf auth login` session when available. Authentication is recommended on systems that share an outbound IP, because anonymous Hugging Face resolver requests can be rate-limited.

Keep this terminal running.

## Configure Gym (terminal 2)

From the Gym repository root, create or update `env.yaml`:

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## Create the datasets (terminal 2)

Clone both source repositories into the Gym repository root. Run each clone command only when that checkout is not already present.

```bash
git clone https://github.com/fchollet/ARC-AGI
git clone https://github.com/arcprize/ARC-AGI-2

cd environments/arc_agi
python prepare.py
python prepare.py --version 2
cd ../..
```

The preparation commands create:

- `environments/arc_agi/data/arc_agi_1_training.jsonl`
- `environments/arc_agi/data/arc_agi_1_evaluation.jsonl`
- `environments/arc_agi/data/example_1.jsonl`
- `environments/arc_agi/data/arc_agi_2_training.jsonl`
- `environments/arc_agi/data/arc_agi_2_evaluation.jsonl`
- `environments/arc_agi/data/example_2.jsonl`

## Start the ARC-AGI environment (terminal 2)

The same environment serves ARC-AGI-1 and ARC-AGI-2:

```bash
gym env start --environment arc_agi --model-type vllm_model
```

Keep this terminal running.

## Collect rollouts (terminal 3)

Open another terminal, activate the existing Gym environment, and run from the Gym repository root:

```bash
source .venv/bin/activate

gym eval run --no-serve \
  --agent arc_agi_simple_agent \
  --input environments/arc_agi/data/example_1.jsonl \
  --output environments/arc_agi/data/example_1_rollouts.jsonl

gym eval run --no-serve \
  --agent arc_agi_simple_agent \
  --input environments/arc_agi/data/example_2.jsonl \
  --output environments/arc_agi/data/example_2_rollouts.jsonl
```

When both evaluations finish, stop `gym env start` and `vllm serve` with `Ctrl-C` in their respective terminals.

## Working-tree hygiene

The `ARC-AGI/` and `ARC-AGI-2/` checkouts, `nano_v3_reasoning_parser.py`, generated `example_1.jsonl` and `example_2.jsonl` files, rollout outputs, and their derived sidecars are local run artifacts. The repository does not currently ignore all of these paths, so they may appear in `git status --short`. Do not commit them; remove them after the run if you need to restore a clean checkout.

For training, see the [NeMo RL GRPO tutorial](https://docs.nvidia.com/nemo/gym/tutorials/training-tutorials/nemo-rl-grpo).
