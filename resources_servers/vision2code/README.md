# Vision2Code

Image recreation: **reference image → Python submission → Enroot render → vision judge → reward**.
The `vision2code_agent` extends `SimpleAgent`, defaults to one policy call, and supports
optional feedback-driven revisions. The verifier
compares the rendered PNG with the reference using one of 16 dataset rubrics or the generic rubric.
Policy and judge use separate configurable OpenAI-compatible vision endpoints.

## Create the Python environment

From the Gym repository root, create and activate a virtual environment with Python 3.13.14
or newer, then install Gym and the renderer's host dependencies:

```bash
uv venv --python 3.13.14
source .venv/bin/activate
uv sync --frozen --extra dev
cd resources_servers/vision2code
uv pip install -r requirements.txt
cd ../..
```

The evaluation worker also needs Enroot and the renderer image built below. Policy and
judge inference run on separately configured endpoints; a local GPU is not required for
the Gym servers. The container has its own Python dependencies.

## Prepare tasks

Run from the Gym repository root with Gym and [requirements.txt](requirements.txt) installed.
Create a JSONL manifest with an explicit rubric for each image:

```json
{"id":"chem-001","image_path":"images/chem-001.png","rubric":"ChemVQA-2K"}
{"id":"chart-001","image_path":"images/chart-001.png","rubric":"ChartQA"}
{"id":"other-001","image_path":"images/other-001.png","rubric":"generic"}
```

| Category | Rubric names |
| --- | --- |
| Charts | `ChartQA`, `dvqa`, `figureqa`, `matplotlib` |
| Chemistry | `ChemVQA-2K`, `ChemVQA-2K-spacing-v1` |
| Documents / electronics | `DocVQA`, `EEE-Bench` |
| Geometry | `geometry3k`, `GEOQA_8K_R1V`, `Geoperception` |
| Graphs | `Graph-Algorithms`, `GraphVQA-Swift` |
| Problem diagrams / physics / spatial scenes | `OlympiadBench`, `Physics`, `spatialvlm_qa` |
| General recreation | `generic` |

```bash
python -m resources_servers.vision2code.prepare /data/manifest.jsonl /data/tasks.jsonl
```

Relative image paths resolve against the manifest directory. Preparation validates static
PNG/JPEG/WebP/GIF images, embeds them in the policy input, and records their absolute paths,
dimensions, and hashes in `verifier_metadata`. The resource server must still be able to read
these files; changed references are rejected. A portable synthetic task is provided in
[data/example.jsonl](data/example.jsonl).

## Renderer and evaluation

Build the renderer once on a Docker host, then place the `.sqsh` on the evaluation worker:

```bash
docker build -t vision2code-render:1 resources_servers/vision2code/container
enroot import -o /images/vision2code-render.sqsh dockerd://vision2code-render:1
```

The image contains matplotlib, NumPy, Pillow, pandas, NetworkX, and DejaVu/Liberation fonts.
Each submission runs in a fresh sandbox with network/PID isolation, a read-only rootfs,
and a private writable `/workspace`. References and host credentials are not mounted.
The launcher supplies `OUTPUT_PATH`, `TARGET_WIDTH`, and `TARGET_HEIGHT`; code must save
`/workspace/output.png`. Defaults: 60-second execution timeout and eight concurrent renders.
The worker or scheduler supplies resource limits; Enroot does not enforce per-attempt CPU/memory limits.

Optionally check isolation when setting up a new worker or renderer image; Slurm is optional.
This diagnostic is separate from normal evaluation, which checks isolation before each render:

```bash
python -m resources_servers.vision2code.smoke \
  --image /images/vision2code-render.sqsh --artifacts /results/vision2code/isolation \
  --forbid-path /data/images/chem-001.png
```

`--forbid-path` asserts that the supplied host path is **not visible inside the sandbox**;
it does not configure mounts or deny access itself. Replace the example with a reference
image that exists on your worker to check that generated code cannot read and copy it.
Repeat the flag for additional reference or credential paths, or omit it to use the
default home, checkout, and credential-directory checks. There is no
unsandboxed fallback. Enroot is an HPC execution boundary, not a hardened multi-tenant runtime.

Configure existing inference services; Gym's adapters do not launch them:

```bash
export VISION2CODE_SQSH=/images/vision2code-render.sqsh
export VISION2CODE_ARTIFACTS=/results/vision2code/artifacts
export VISION2CODE_POLICY_URL=http://policy-host:8000/v1
export VISION2CODE_POLICY_MODEL=your-vision-model
export VISION2CODE_JUDGE_URL=http://judge-host:8000/v1
export VISION2CODE_JUDGE_MODEL=your-vision-judge
# Set VISION2CODE_POLICY_KEY and VISION2CODE_JUDGE_KEY if required; both default to EMPTY.

gym env start --config resources_servers/vision2code/configs/vision2code.yaml

gym eval run --no-serve --agent vision2code_simple_agent \
  --input /data/tasks.jsonl --output /results/vision2code/rollouts.jsonl \
  --num-repeats 1 --temperature 0.7 --max-output-tokens 8192
```

Start with `--limit 1` and inspect the code, render, judge scores, and cleanup diagnostics.
For optional feedback-driven revisions, see [Multi-pass revisions](#multi-pass-revisions).
For reasoning endpoints, configure each model adapter's `uses_reasoning_parser` to match
its response format. The optional [Qwen config](configs/vision2code_qwen.yaml) uses
`NVIDIA_API_KEY` and the same hosted model for both roles; use an independent judge for baselining.

For NeMo RL, use `uv sync --frozen --extra dev` and keep per-server venvs on the base's
Ray version, **2.56.1**. Before server setup, set `UV_CONSTRAINT` to a file containing
`ray==2.56.1`; update reused venvs separately. RL trainer configuration is external to this environment.

## Multi-pass revisions

The agent reuses `SimpleAgent` for model requests, verification, cookies, and rollout
capture. Enable up to three generate-and-verify passes, including the initial attempt:

```bash
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code_simple_agent.responses_api_agents.vision2code_agent.num_passes=3
```

Continue using `--agent vision2code_simple_agent` in `gym eval run`; the instance name
is retained for compatibility.
Agent settings live under `vision2code_simple_agent.responses_api_agents.vision2code_agent`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `num_passes` | `1` | Maximum generate-and-verify cycles, including the initial attempt. Positive integer; no fixed upper bound. |
| `max_steps` | `1` | Fixed at one policy response per pass. Use `num_passes` for revisions. |
| `skip_verification` | `false` | Inherited single-pass option; incompatible with multiple passes. |

Each pass submits a complete Python program. After a successful render, feedback lists
the readable label and judge rationale for each category scored below 5, under
"Improve on these." Scores, weights, perfect-score categories, overall issues, and the
overall summary are omitted. If no category needs correction but the configured reward
still permits another pass (for example, with Blend A), feedback asks the model to recheck
the image against the reference. A failed render supplies its status, return code, error
type, and at most the last 4,000 characters of stderr. The conversation retains the
original reference, instructions, previous responses, and feedback. The generated PNG
is not added to the policy conversation.

The loop stops at exact reward `1.0`, the pass limit, or a masked infrastructure/judge
failure. Early stopping uses the configured verifier reward, including reward shaping
or Blend A. Code errors, timeouts, and invalid images can be revised.

The **last attempt** supplies the returned response, reward, rubric, and artifact paths,
even if an earlier attempt scored higher. Multi-pass results add an `attempts` list
with each pass's number, reward, feedback, and artifacts. Policy token usage and rollout
trajectory include all executed passes. With `num_passes=1`, the agent delegates directly
to `SimpleAgent` and preserves its response format.

Every pass uses a fresh Enroot sandbox from the same image. Files, processes, and Python
state do not persist; only conversation history carries forward. This prevents a stale
`output.png` or modified workspace from affecting the next score.

Each additional pass costs a policy call and render, plus a judge call for a valid image.
The default execution timeout is 60 seconds **per render**. Model output-token limits
apply per call; the agent has no aggregate token or wall-clock budget. Start with a small
pass budget and inspect timings before increasing it.

## Performance

Inspect `render.timings` (startup, execution, cleanup) and `timings.judge_s` in each
attempt's `result.json` before tuning. A fresh sandbox preserves independent attempts;
reusing a live workspace can let stale outputs or modified files affect scores.

The renderer uses one combined network/PID isolation and mount probe before running
code. Workspace setup is shared by evaluation and the isolation smoke. Use node-local
temporary storage for Enroot's `data_path` and `runtime_path` when cluster defaults point
to a shared filesystem. These existing provider settings live under
`vision2code.resources_servers.vision2code.sandbox_provider.enroot.create`.

Increase `render_concurrency` only when render queueing limits throughput and the worker
has enough CPU and memory. Increase evaluation `--concurrency` to overlap model/judge
latency across tasks within endpoint capacity. Passes within a task remain sequential.
Changing judge models, reasoning settings, or token limits may change scores; validate
those separately rather than treating them as equivalent performance optimizations.

## Rewards

All three modes return rewards in `[0, 1]`. Let `R` be the final rubric rating in `[0, 5]`
and `L` the raw weighted judge score before caps and final rounding.

| Mode | Configuration | Reward |
| --- | --- | --- |
| Normal (linear) | `use_blended_reward=false`, `reward_power=1` | `R / 5` |
| Squared (default) | `use_blended_reward=false`, `reward_power=2` | `(R / 5)²` |
| Blend A | `use_blended_reward=true` | `0.1 × RGB + 0.2 × foreground F1 + 0.2 × edge F1 + 0.5 × (L / 5)` |

A **rubric** is the checklist given to the vision judge. It scores each category from 0 to 5
and explains its assessment. Categories and weights depend on the image type: ChartQA uses
numeric information (30%), chart structure/layout (30%), text/label accuracy and legibility
(25%), and completeness (15%). Chemistry rubrics instead emphasize chemical structure and
labels. The generic rubric weights information, layout, text, and completeness equally.

The weighted category mean is `L`. Dataset rubrics then apply **caps**: maximum allowed
ratings when critical categories are weak or the image is blank/degenerate. For example,
a chart with correct numbers but the wrong orientation/type can be capped at 3.5/5.
After caps and rounding, the result is `R`; generic scoring has no automatic caps.
`R=4` gives **0.8 normal** or **0.64 squared**. Other finite powers of at least 1 remain supported.

**We removed the `mostly_empty_canvas` filter from reward penalties.** Sparse diagrams
are not capped merely for having a mostly white or black canvas. The flag remains in
inspection diagnostics; other blank/degenerate checks and dataset-specific caps still apply.

Blend A mixes three computed image metrics with the judge's assessment:

| Component | What it measures | Reward weight |
| --- | --- | --- |
| Foreground RGB similarity | How closely pixel colors match where either image has foreground content. Empty background does not inflate the score. | 10% |
| Foreground F1 | Whether drawn regions occur in the right places, balancing extra content against missing content. | 20% |
| Edge F1 | Whether boundaries, lines, and outlines occur in the right places. | 20% |
| Raw judge score `L / 5` | Rubric-based correctness, layout, and readability that pixel matching alone cannot assess. | 50% |

The three image metrics range from 0 to 1, with 1 meaning a perfect match under that metric.
Foreground means pixels sufficiently different from the reference's estimated background color.
F1 combines precision (how much generated content matches the reference) and recall (how much
reference content was reproduced); a match may be within **2 pixels** in the analysis image.

The candidate is placed at the top-left of a canvas matching the reference dimensions;
overflow is cropped and uncovered space uses the estimated background color. Both images are
then downscaled together if the longest side exceeds **1024 pixels**. There is no automatic
alignment (image registration), so shifted drawings can lose credit.

Blend A uses **raw `L`**, bypassing rubric caps and `reward_power`. For example, RGB=0.9,
foreground F1=0.8, edge F1=0.7, and `L=4` give reward **0.79**. The blended 0–5 rating is rounded
to four decimals before division by 5. If a metric is undefined, such as when neither image
has any edges, the sample is masked (excluded from scored aggregates); the remaining components
do not receive its weight.

Choose a mode by appending an override to `gym env start`:

```bash
# Normal; omit the override for default squared rewards.
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code.resources_servers.vision2code.reward_power=1

# Blend A; ignores reward_power.
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code.resources_servers.vision2code.use_blended_reward=true
```

## Failures and artifacts

Code errors, timeouts, and missing/invalid PNGs receive **zero** without a judge call.
Infrastructure failures, missing/changed references, judge failures, and invalid judge scores
are **masked** and excluded from quality aggregates. Masking fields and failure-sidecar routing
are retained in the rollout.

Artifacts include `code.py`, reference `image.png`, valid `output.png`, `execution.json`,
`judge.json`, and `result.json` in a unique attempt directory. Results include category scores, caps,
reward settings, timings, and cleanup diagnostics. Outputs must be regular, readable,
nonanimated PNGs of at most 32 MiB and 25 million pixels. Dimension mismatches are recorded;
the renderer does not resize outputs. Cancellation waits for bounded execution and cleanup.

## Validation and provenance

```bash
python -m pytest resources_servers/vision2code/tests responses_api_agents/vision2code_agent/tests

# Real renderer tests, on an Enroot worker:
VISION2CODE_TEST_SQSH=/images/vision2code-render.sqsh \
  python -m pytest resources_servers/vision2code/tests/test_enroot_integration.py
```

Golden fixtures cover all 17 rubrics, exact prompts, rounding/caps, failure results, and
Blend A metrics. Server tests cover reward modes, masking, concurrency, and cancellation;
the offline verifier fixture checks aggregation only. Live judge quality requires model rollouts.

Adapted from [Vision2Code](https://github.com/image2code/vision2code), under Apache-2.0;
see [NOTICE](NOTICE), [LICENSE.upstream](LICENSE.upstream), and [provenance](tests/fixtures/provenance.json).
Dataset licenses are separate. This is a configurable environment, not a reproduction of all
upstream benchmark results. The [catalog entry](../../environments/vision2code/manifest.yaml)
remains experimental and unverified pending broader baselining.
