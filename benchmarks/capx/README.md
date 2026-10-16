# CaP-Bench (CaP-X)

[CaP-X](https://github.com/capgym/cap-x) ([paper](https://arxiv.org/abs/2603.22435)) benchmarks coding agents on robot
manipulation. The model writes a Python program against robot perception and control APIs; the program runs in a
Robosuite simulator, and the episode is scored by the simulator's task-completion check (reward 1 or 0). See
[`resources_servers/capx`](../../resources_servers/capx/README.md) for the episode protocol, process model, and failure
masking.

## Tasks and tiers

7 tasks (Cube Lift, Cube Stack, Spill Wipe, Peg/Nut Assembly, Cube Re-Stack, Two-Arm Lift, Two-Arm Handover) at the
text-only CaP-Bench tiers, 100 seeds per cell. Re-Stack is scored in M1 only, as in the paper, because it requires
checking the initial condition before acting. The full set has 3,100 rows.

| Tier | Paper | CaP-X config suffix | Episode |
|---|---|---|---|
| `privileged` | S1 | `_privileged` | single-turn, ground-truth object state APIs |
| `full` | S2 | none | single-turn, perception APIs (SAM3, Contact-GraspNet) |
| `reduced_api` | S3 | `_reduced_api` (`_reduced` for two-arm tasks) | single-turn, low-level APIs with examples |
| `reduced_api_exampleless` | S4 | `_reduced_api_exampleless` | single-turn, low-level APIs, no examples |
| `multiturn` | M1 | `_multiturn` | multi-turn: execute, read stdout/stderr, then REGENERATE or FINISH |

Not covered: M2-M4 (image feedback or a visual-differencing VLM), LIBERO-PRO, and BEHAVIOR.

## Metrics

`pass@1[avg-of-1]/accuracy` is the success rate over all episodes. The per-tier rates are
`pass@1[avg-of-1]/accuracy_<tier>`. Episodes masked as environment failures are excluded.

## Run

```bash
gym eval prepare --benchmark capx
gym env start --benchmark capx --model-type vllm_model
gym eval run --no-serve --agent capx_gymnasium_agent \
    --input benchmarks/capx/data/capbench_benchmark.jsonl --output results/capx.jsonl \
    ++observability_enabled=true ++model_call_capture_dir=$PWD/results/model_calls
```

On first start, the resources server installs the pinned CaP-X commit into `resources_servers/capx/.capx` (git, `uvx`,
and network required; takes several minutes). It then starts CaP-X's API servers. The node needs:

- one GPU for the API servers (SAM2, SAM3, Contact-GraspNet, OWL-ViT; about 20 GB) and EGL rendering;
- EGL/X11 libraries: `libegl1 libgl1 libglu1-mesa libx11-6 libxext6 libxrender1`;
- `HF_TOKEN` with access to the gated `facebook/sam3` weights;
- about one CPU per concurrent episode (`max_workers`).

Set `capx_root` (or `CAPX_ROOT`) to reuse an existing, set-up checkout. Set `launch_api_servers: false` if the API servers
already run on ports 8113-8117. Sampling follows CaP-X's harness: temperature 1.0 and 20,480 output tokens.

A subset for smoke runs: `python benchmarks/capx/prepare.py --trials 3 --tiers privileged --output subset.jsonl`.
