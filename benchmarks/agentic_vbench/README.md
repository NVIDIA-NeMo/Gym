# Agentic-VBench

Evaluate video-editing agents on the 100 pinned Agentic-VBench tasks: Repair (18),
Assembly (18), Sequencing (28), and Repurpose (36). This benchmark uses the
`agentic_vbench_agent` adapter and Gym's `legacy_agent` environment server.
The agent delegates execution and verification to the existing AgenticVBench
Harbor/OpenCode runner; it does not replace the benchmark with a question-answering task.

Task revision: `610c4ecc69ac56fc62e8cfbd3b28dddd88f22863`.
Harbor: `0.6.6`. OpenCode: `1.14.39`.
The advertised model context is 131,072 tokens and output capability is 32,000 tokens.
The task's own timeouts govern execution. Gym sampling arguments do not override
the pinned OpenCode protocol. Run independent server seeds 201, 202, and 203 for
the standard three-rollout report.

## Runtime requirements

This integration currently uses the Linux/HPC runner in the AgenticVBench repository:
`evals/VLMEvalKit/shell/run_agentic_vbench_harbor_trial.sh`.
It requires rootless Podman, curl, flock, internet access for task setup, and a
reachable OpenAI-compatible model endpoint with native image input and tool calling.
The runner installs pinned Harbor and podman-compose into a locked node-local cache.
Run Gym on a CPU worker that supports rootless Podman; a generic evaluation container
without that runtime is insufficient.

Set these variables to real paths before preparing or running:

```bash
export AGENTIC_VBENCH_ROOT=/path/to/pinned/agentic-vbench
export AGENTIC_VBENCH_EVALKIT_ROOT=/path/to/AgenticVBench/evals/VLMEvalKit
export AGENTIC_VBENCH_OUTPUT_ROOT=/path/outside/checkouts/episodes
export AGENTIC_VBENCH_RUNTIME_ROOT=/node/local/writable/avb-runtime
export AGENTIC_VBENCH_MODEL_BASE_URL=http://model-host:8000/v1
export AGENTIC_VBENCH_MODEL_ID=agentic-vbench-model
export AGENTIC_VBENCH_DATASET=/path/outside/checkouts/inputs.jsonl
export AGENTIC_VBENCH_CREDENTIALS_FILE=/secure/path/credentials.env
export RAY_TMPDIR=/tmp

gym eval prepare --benchmark agentic_vbench
gym eval run --benchmark agentic_vbench --split benchmark \
  --output /path/outside/checkouts/rollouts.jsonl --concurrency 4 --num-repeats 1
```

`AGENTIC_VBENCH_TASKS` optionally selects one family or one exact task ID; the default
is `all`. Preparation validates the entire pinned inventory even for a subset.
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
and runs the existing parser/secret/aggregation audits.

This adapter does not produce model token IDs/logprobs and is intended for evaluation,
not RL training. Raw wire captures come from the companion serving launcher's passive
proxy. Do not treat unit tests or an unaudited partial run as a validated baseline.
