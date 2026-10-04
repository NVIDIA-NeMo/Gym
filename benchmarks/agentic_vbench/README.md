# Agentic-VBench

Evaluate video-editing agents on the 100 pinned Agentic-VBench tasks: Repair (18),
Assembly (18), Sequencing (28), and Repurpose (36). This benchmark uses the
`agentic_vbench_agent` adapter and Gym's `legacy_agent` environment server.
The agent delegates execution and verification to the upstream Harbor/OpenCode runner; it does not replace the benchmark with a question-answering task.

Task revision: `410a75fe3dae37de4344fc9e5317da089505330a`.
Harbor: `0.6.6`. OpenCode: `1.14.39`.
The advertised model context is 262,144 tokens and output capability is 100,000 tokens.
The task's own timeouts govern execution. The Factory request proxy applies sampling overrides. Run independent request seeds 201, 202, and 203 for
the standard three-rollout report.

## Runtime requirements

This development adapter calls `harbor_runner.py` with a separate Python environment
containing upstream `harbor==0.6.6`. Its custom Podman environment uses the original
task Dockerfiles and native verifier. It requires rootless Podman, slirp4netns,
`/usr/bin/tini`, and a reachable OpenAI-compatible model endpoint.

Synthetic container build, execution, file transfer, host-local HTTP access, and
cleanup have passed on the CPU cluster. The original voicebank repair image also passed these checks (CPU job 2116776).
A build-only apt configuration mount runs apt as mapped container root to support
the cluster’s single-UID namespace. Original task Dockerfiles stay unchanged.
Other task images and real model rollouts still require validation.
This migration has not yet produced a validated benchmark result.

Set these variables to real paths before preparing or running:

```bash
export AGENTIC_VBENCH_ROOT=/path/to/pinned/agentic-vbench
export AGENTIC_VBENCH_HARBOR_PYTHON=/path/to/harbor-venv/bin/python
export AGENTIC_VBENCH_OUTPUT_ROOT=/path/outside/checkouts/episodes
export AGENTIC_VBENCH_RUNTIME_ROOT=/node/local/writable/avb-runtime
export AGENTIC_VBENCH_MODEL_BASE_URL=http://model-host:8000/v1
export AGENTIC_VBENCH_MODEL_ID=agentic-vbench-model
export AGENTIC_VBENCH_DATASET=/path/outside/checkouts/inputs.jsonl
export AGENTIC_VBENCH_CREDENTIALS_FILE=/secure/path/credentials.env
export RAY_TMPDIR=/tmp/avb-ray-unique-run
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS

# Resolve variables before Gym converts dataset paths into Path objects.
python - <<'PYCONFIG'
import os
import socket
from pathlib import Path
from omegaconf import OmegaConf
config = OmegaConf.load("benchmarks/agentic_vbench/config.yaml")
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    config.head_server = {"host": "127.0.0.1", "port": listener.getsockname()[1]}
output = Path(os.environ["AGENTIC_VBENCH_DATASET"]).with_suffix(".yaml")
output.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config, output, resolve=True)
PYCONFIG

gym eval prepare --config "${AGENTIC_VBENCH_DATASET%.jsonl}.yaml"
gym eval run --config "${AGENTIC_VBENCH_DATASET%.jsonl}.yaml" --split benchmark \
  --output /path/outside/checkouts/rollouts.jsonl --concurrency 4 --num-repeats 1
```

`AGENTIC_VBENCH_TASKS` selects `all` (the default), or a space-separated list
of families and exact task IDs. Family directory names such as
`agentic_vbench_repair` are accepted. Overlapping selections are rejected.
Preparation validates the entire pinned inventory even for a subset.
The credential file is needed for Repurpose's native Anthropic/Gemini judges.
Keep its contents out of configs and source control. Source prompts and task assets
are not changed. The model receives no automatic initial frames or captions; it
must inspect extracted images with OpenCode's `read` tool.

## Results and failure handling

Each completed Gym row retains reward, family, task ID, verifier status, original
Harbor trajectory, and the episode artifact path. Valid zero-reward outcomes are
retained. Unscored setup/verifier failures are errors, not synthesized zeros.
Identical HTTP retries reuse the same episode and persisted result. An incomplete
episode requires inspection before an explicit infrastructure retry; the adapter
never automatically reruns a model trajectory. Use a new output root for each
model and independent evaluation rollout.

Gym's generic `mean/reward` is task-weighted. The benchmark's leaderboard-style
score is the equal mean of four family scores, averaged across three complete
rollouts. Do not report a subset smoke result as the benchmark score. The Eval
Factory companion integration validates selected IDs, emits the native TSV layout,
and aggregates the selected episodes. End-to-end result and request audits remain
required before reporting a benchmark comparison.

This adapter does not produce model token IDs/logprobs and is intended for evaluation,
not RL training. Request capture must be configured in the Factory pipeline. Do not treat unit tests or an unaudited partial run as a validated baseline.
