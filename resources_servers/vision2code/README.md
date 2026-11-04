# [Vision2Code](https://github.com/image2code/vision2code)

The goal is to improve a model's visual understanding through image recreation: the model
studies a reference image, writes Python to recreate it, and learns from a vision judge's
feedback. This Gym environment provides rendering, rubric rewards, and optional revisions
for evaluation and training.

**Reference image → Python code → rendered image → judge feedback and reward → revision.**

## Rubrics

Choose a rubric for each reference image. The judge compares the reference and recreation
using the following criteria:

| Rubric | What it judges |
| --- | --- |
| `ChartQA` | Chart values, structure, labels, and completeness. |
| `dvqa` | Chart values, layout, label accuracy, and legibility. |
| `figureqa` | Chart information, structure, text, and visual completeness. |
| `matplotlib` | Plot layout, plotted elements, ticks, annotations, and styling. |
| `ChemVQA-2K` | Molecular structure, chemical symbols, bonds, and readability. |
| `ChemVQA-2K-spacing-v1` | Chemical fidelity plus clearance between labels, bonds, charges, and subscripts. |
| `DocVQA` | Text content, document layout, legibility, and key visual elements. |
| `EEE-Bench` | Circuit connectivity, components, labels, directions, and background color. |
| `geometry3k` | Geometric construction, labels, markers, and their placement. |
| `GEOQA_8K_R1V` | Geometric precision and correct association of numbers, labels, and markers. |
| `Geoperception` | Geometric structure, annotations, spatial placement, and readability. |
| `Graph-Algorithms` | Graph connectivity, ordering, edge direction, and labels. |
| `GraphVQA-Swift` | Graph structure, labels, and mapping of visual attributes to nodes and edges. |
| `OlympiadBench` | Problem-diagram semantics, symbols, labels, and spatial precision. |
| `Physics` | Physical setup, vectors, labels, and consistent object geometry. |
| `spatialvlm_qa` | Object identity, spatial relations, depth, and scene appearance. |
| `generic` | Information, layout, annotations, and completeness, with equal weights. |

Exact weights and score caps are defined in [rubrics.py](rubrics.py) and
[generic_rubric.py](generic_rubric.py).

## Setup

Run commands from the Gym repository root:

```bash
uv venv --python 3.13.14
source .venv/bin/activate
uv sync --frozen --extra dev
uv pip install -r resources_servers/vision2code/requirements.txt
```

The worker needs Enroot. Build the renderer image on a Docker host and place the resulting
`.sqsh` on the worker:

```bash
docker build -t vision2code-render:1 resources_servers/vision2code/container
enroot import -o /images/vision2code-render.sqsh dockerd://vision2code-render:1
```

[`container/`](container/) defines Python, plotting libraries, and fonts inside the image;
[`enroot/`](enroot/) defines its runtime mounts. Each attempt gets a fresh sandbox with a
read-only rootfs, a private `/workspace`, and network/PID isolation. Reference files and
host credentials are not mounted. Enroot does not enforce per-attempt CPU/memory limits;
the worker or scheduler must supply them.

The generated program receives `OUTPUT_PATH`, `TARGET_WIDTH`, and `TARGET_HEIGHT` and must
save `/workspace/output.png`. It should recreate the background too: transparency is not a
substitute for an opaque reference background. The judge and blank-image checks preserve
alpha; Blend A's pixel metrics composite it onto white.

## Prepare tasks

Create a JSONL manifest with one image and rubric per row:

```json
{"id":"chem-001","image_path":"images/chem-001.png","rubric":"ChemVQA-2K"}
{"id":"chart-001","image_path":"images/chart-001.png","rubric":"ChartQA"}
{"id":"other-001","image_path":"images/other-001.png","rubric":"generic"}
```

```bash
python -m resources_servers.vision2code.prepare /data/manifest.jsonl /data/tasks.jsonl
```

Image paths resolve relative to the manifest. Preparation embeds the image in the model
input and records its path, dimensions, and hash for verification. Keep the source images
accessible to the resource server. Five synthetic [example tasks](data/example.jsonl) and
[real rollouts](data/example_rollouts.jsonl) are included.

## Run evaluation

Configure separate policy and judge endpoints. Gym connects to existing inference services:

```bash
export VISION2CODE_SQSH=/images/vision2code-render.sqsh
export VISION2CODE_ARTIFACTS=/results/vision2code/artifacts
export VISION2CODE_POLICY_URL=http://policy-host:8000/v1
export VISION2CODE_POLICY_MODEL=your-vision-model
export VISION2CODE_JUDGE_URL=http://judge-host:8000/v1
export VISION2CODE_JUDGE_MODEL=your-vision-judge
# Set VISION2CODE_POLICY_KEY and VISION2CODE_JUDGE_KEY if authentication is required.

gym env start --config resources_servers/vision2code/configs/vision2code.yaml
```

In another shell with the same environment activated:

```bash
gym eval run --no-serve --agent vision2code_simple_agent \
  --input /data/tasks.jsonl --output /results/vision2code/rollouts.jsonl \
  --num-repeats 1 --temperature 0.7 --max-output-tokens 8192
```

Start with `--limit 1` to inspect a rollout. For reasoning endpoints, set the model adapter's
`uses_reasoning_parser` to match its response format. The optional
[Qwen config](configs/vision2code_qwen.yaml) uses Qwen3.8-27B for both roles and authenticates
with `NVIDIA_API_KEY`; configure an independent judge for benchmark comparisons.

## Multi-pass revisions

Set `num_passes` to the maximum number of attempts, including the initial generation.
For example, `n=3` allows an initial attempt and two revisions:

```bash
n=3
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code_simple_agent.responses_api_agents.vision2code_agent.num_passes="$n"

# In another shell:
gym eval run --no-serve --agent vision2code_simple_agent \
  --input /data/tasks.jsonl --output /results/vision2code/pass3.jsonl \
  --num-repeats 1 --temperature 0.7 --max-output-tokens 8192
```

Restart the servers when changing the pass budget or reward mode. `--num-repeats` means
independent rollouts per task; it does not control revision turns.

Each revision receives the judge's rationales for imperfect categories, or error details
when rendering fails. The conversation retains the reference, previous code, and feedback;
generated images and sandbox files are not carried into the next pass. Each pass submits a
complete replacement program.

The agent stops at reward `1.0`, the pass limit, or a masked/structured failure. It returns
the **last attempt**, even if an earlier attempt scored higher. The rollout's `attempts`
list records each pass's reward, feedback, and artifact paths. Each extra pass costs a
policy call and render, plus a judge call for a valid image; token limits apply per call.

### Example: three saved passes

This Qwen3.8-27B rollout uses the `matplotlib` rubric, GPT-6.1-Sol judge, and linear reward.
The reference and all three renders are reused from the [saved evaluation](#test-mini-results).

| Reference | Pass 1 · 0.78 | Pass 2 · 0.84 | Pass 3 · 0.92 |
| --- | --- | --- | --- |
| ![Reference error-bar plot](results/revision-example/reference.png) | ![Pass 1 recreation](results/revision-example/pass1.png) | ![Pass 2 recreation](results/revision-example/pass2.png) | ![Pass 3 recreation](results/revision-example/pass3.png) |

Feedback led to a corrected background, a connected point-to-point line, and removal of
extra error-bar caps. This is one illustrative improving example; revisions can also
plateau or regress. [Saved feedback and provenance](results/revision-example/provenance.json).
Reference: Matplotlib's [error-bar example](https://matplotlib.org/stable/gallery/statistics/errorbar.html),
under the [Matplotlib license](results/revision-example/LICENSE.matplotlib).

## Rewards

Let `R` be the final rubric rating on a 0–5 scale and `L` the weighted judge score before
caps and rounding. All reward modes return values in `[0, 1]`.

| Mode | Configuration | Reward |
| --- | --- | --- |
| Linear | `reward_power=1` | `R / 5` |
| Squared (default) | `reward_power=2` | `(R / 5)²` |
| Blend A | `use_blended_reward=true` | `0.1 × RGB + 0.2 × foreground F1 + 0.2 × edge F1 + 0.5 × (L / 5)` |

For example, a rubric rating of 4 gives 0.8 linear reward or 0.64 squared reward. Dataset
rubrics cap scores for critical errors or blank/degenerate images; generic scoring has no
automatic caps. Sparse content alone (`mostly_empty_canvas`) does not trigger a penalty.

Blend A combines foreground color similarity, foreground overlap, edge overlap, and the
raw judge score. It bypasses rubric caps and `reward_power`. Pixel metrics use a maximum
1024-pixel analysis dimension and a 2-pixel F1 tolerance, without automatic alignment;
undefined metrics mask the sample. See [image_metrics.py](image_metrics.py) for the protocol.

Append the appropriate override to the server-start command, including multi-pass runs:

```bash
# Linear; squared is the default.
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code.resources_servers.vision2code.reward_power=1

# Blend A.
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code.resources_servers.vision2code.use_blended_reward=true
```

## Test-mini results

Saved Gym evaluations of **Qwen3.8-27B**, judged by **GPT-6.1-Sol**, cover the same
539 tasks across 15 rubrics in both runs. Settings: temperature 0, 32,768 output tokens
per policy call, one rollout per task, category-level feedback, and linear reward for
collection and early stopping.

| Maximum passes | Tasks scored | Successful final renders | Linear | Squared (offline) | Blend A (offline) |
| --- | --- | --- | --- | --- | --- |
| 1 | 539 | 503 | 0.7947 | 0.7001 | 0.6622 |
| 3 | 539 | 528 | 0.8815 | 0.8204 | 0.7267 |

Means weight each task equally, including zero rewards for code/render failures. Neither
run has masked samples. Squared and Blend A values rescore the same final attempts offline;
they are not separate runs with those rewards. These saved runs were audited from their
records, not rerun against the final PR checkout. See the
[per-task scores](results/test-mini-qwen.csv) and [provenance](results/test-mini-provenance.json).

Kimi K2.5 and Qwen3.5 scores in the upstream report are not Gym reproductions. A comparable
Kimi baseline still needs the same tasks and judge, starting with a small smoke check.

## Failures and artifacts

Code errors, execution timeouts, and missing/invalid PNGs receive zero reward. Infrastructure,
reference-image, and judge failures are masked and excluded from scored aggregates.
Each attempt saves the Python code, reference, generated image, execution diagnostics,
judge output, and result in its own artifact directory.

Outputs must be regular, readable, nonanimated PNGs of at most 32 MiB and 25 million pixels.
Dimension mismatches are recorded without resizing. Rendering defaults to a 60-second
timeout and concurrency of eight. Inspect render/judge timings before raising concurrency.

## Validation

```bash
python -m pytest resources_servers/vision2code/tests responses_api_agents/vision2code_agent/tests

# Real renderer tests on an Enroot worker:
VISION2CODE_TEST_SQSH=/images/vision2code-render.sqsh \
  python -m pytest resources_servers/vision2code/tests/test_enroot_integration.py

# Optional isolation check for a new worker/image:
python -m resources_servers.vision2code.smoke \
  --image /images/vision2code-render.sqsh --artifacts /results/vision2code/isolation \
  --forbid-path /data/images/chem-001.png
```

`--forbid-path` checks that an existing host file is invisible inside the sandbox; it does
not configure mounts. Live judge quality requires model rollouts. The environment remains
experimental and `verified: false` pending broader baselining.

Adapted code is Apache-2.0; see [NOTICE](NOTICE), [LICENSE.upstream](LICENSE.upstream), and
[source provenance](tests/fixtures/provenance.json). Dataset licenses are separate.
