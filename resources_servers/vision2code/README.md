# Vision2Code

This environment is based on the [Vision2Code benchmark](https://github.com/image2code/vision2code).
It evaluates models' image reconstruction capabilities using 16 dataset-specific rubrics
spanning chemistry, physics, documents, charts, geometry, graphs, and spatial reasoning.

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

`ChemVQA-2K-spacing-v1` is an optional variant of the chemistry rubric. It also checks
clearance around atom labels, bonds, charges, and subscripts. Added crowding limits the
rating to 4.5/5; overlap that obscures notation limits it to 4.0/5. It uses the same chemistry
images and is not a separate dataset.

Exact weights and score caps are defined in [rubrics.py](rubrics.py) and
[generic_rubric.py](generic_rubric.py).

## Setup

Run commands from the Gym repository root:

```bash
uv venv --python 3.13.14
source .venv/bin/activate
uv sync --frozen --extra dev
uv sync --locked --project resources_servers/vision2code
source resources_servers/vision2code/.venv/bin/activate
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

Create a JSONL manifest with one image and rubric per row. The included
[manifest.jsonl](data/manifest.jsonl) lists the five synthetic example images:

```json
{"id":"chem-001","image_path":"images/chem-001.png","rubric":"ChemVQA-2K"}
{"id":"chart-001","image_path":"images/chart-001.png","rubric":"ChartQA"}
{"id":"other-001","image_path":"images/other-001.png","rubric":"generic"}
```

```bash
python -m resources_servers.vision2code.prepare \
  resources_servers/vision2code/data/manifest.jsonl /data/tasks.jsonl
```

Paths in the manifest are relative to the manifest file. Preparation sends the reference
as an embedded image in the policy input; its actual filesystem path is hidden from the
policy model and unavailable inside the rendering sandbox. The resource server keeps the
path, dimensions, and checksum in verifier metadata so it can check the reference and score
the output. Keep those source files accessible to the resource server. Prepared
[example tasks](data/example.jsonl) and [real rollouts](data/example_rollouts.jsonl) are included.

## Run evaluation

Provide two running OpenAI-compatible vision-model endpoints: a **policy** that writes Python
and a **judge** that compares the rendered image with the reference. These can be hosted APIs
or models you serve yourself. The commands below start Gym and connect it to those endpoints;
they do not launch the model inference servers.

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
`uses_reasoning_parser` and `uses_interleaved_reasoning` to match its response format.
Configure these on the policy and judge adapters independently.

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

When a rubric category scores **below 5/5**, its judge rationale is added as feedback for
the next revision. Render failures instead provide execution-error details. The conversation
retains the reference, previous code, and feedback; generated images and sandbox files are
not carried into the next pass. Each pass submits a complete replacement program.

The agent stops at reward `1.0`, the pass limit, or a masked/structured failure. It returns
the **last attempt**, even if an earlier attempt scored higher. The rollout's `attempts`
list records each pass's reward, feedback, and artifact paths. Each extra pass costs a
policy call and render, plus a judge call for a valid image; token limits apply per call.

### Example: passes 1–3

This Qwen3.8-27B rollout uses the `matplotlib` rubric, GPT-6.1-Sol judge, and linear reward.
The reference and all three renders are reused from the [saved evaluation](#benchmark-results).

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
caps and rounding. Let `C` be that raw judge score reduced by any applicable rubric caps,
without rounding. All reward modes return values in `[0, 1]`.

| Mode | Configuration | Reward |
| --- | --- | --- |
| Linear | `reward_power=1` | `R / 5` |
| Squared (default) | `reward_power=2` | `(R / 5)²` |
| Blend A | `use_blended_reward=true` | `0.1 × RGB + 0.2 × foreground F1 + 0.2 × edge F1 + 0.5 × (L / 5)` |
| Blend A capped | Both `use_blended_reward=true` and `blended_reward_apply_judge_caps=true` | Same pixel terms as Blend A, plus `0.5 × (C / 5)` |

Linear reward directly scales the rubric rating: 4/5 becomes 0.8. Squared reward makes
imperfect reconstructions less rewarding: the same 4/5 becomes 0.64. Dataset rubrics limit
ratings when important content is wrong or the output is blank or degenerate. A sparse but
valid image is not penalized just for having a mostly empty canvas. The generic rubric
uses an equal-weight judge average without these automatic caps.

Blend A adds image measurements to the judge's semantic assessment. Color similarity
checks foreground colors, foreground F1 checks where content appears, and edge F1 checks
outlines. Together they contribute half the reward; the judge contributes the other half.
This supplies a finer visual signal when the judge gives similar ratings to visibly
different outputs. The pixel measurements use at most a 1024-pixel analysis dimension and
allow a 2-pixel tolerance for F1 matches. Images are not automatically aligned. If a metric
cannot be computed, the sample is masked rather than silently changing the weights.

**Blend A capped** keeps those pixel terms and weights, but applies the rubric's existing
caps to the raw judge score before blending. This retains penalties for semantic errors
such as wrong chemical structure or circuit wiring. Only the judge term is capped; the
whole blend is not clipped to that limit. For example, perfect pixel scores and a raw
judge score of 3.8/5 produce 0.88 with Blend A. If a rubric cap limits the judge to 2.5/5,
Blend A capped produces 0.75. Both blend modes bypass `reward_power`. See
[image_metrics.py](image_metrics.py) for the calculation.

The judge model is part of the reward definition: changing its model, prompt, or decoding
settings can change category ratings, cap activation, and the final reward. Keep the judge
fixed when comparing policies or reward variants, and record its identity with the results.

Append the appropriate override to the server-start command, including multi-pass runs:

```bash
# Linear; squared is the default.
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code.resources_servers.vision2code.reward_power=1

# Blend A.
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code.resources_servers.vision2code.use_blended_reward=true

# Blend A capped: the same blend with caps on the judge term.
gym env start --config resources_servers/vision2code/configs/vision2code.yaml \
  +vision2code.resources_servers.vision2code.use_blended_reward=true \
  +vision2code.resources_servers.vision2code.blended_reward_apply_judge_caps=true
```

### Reward choice for RL

We compared RL checkpoints after 120 training steps on the same 1,994 held-out images
across 14 rubrics. All checkpoints were evaluated with GPT-6.1-Sol, thinking enabled,
temperature 0, and a 32,768-token budget. The table uses the **same uncapped Blend A
metric** for every checkpoint, normalized to `[0, 1]`, including zero scores for failed renders.
Each row is a separately trained model, not a different way of scoring the same outputs.

| Reward used during training | Training judge | Evaluation: uncapped Blend A | Failed renders |
| --- | --- | --- | --- |
| Blend A | GPT-6.1-Sol | 0.6568 | 41 |
| Blend A capped | GPT-6.1-Sol | **0.6771** | 45 |
| Blend A with squared capped judge on spatial tasks | GPT-6.1-Sol | 0.6606 | 89 |
| Blend A | Qwen3.8-27B | 0.6487 | 44 |

The model trained with Blend A capped achieved the highest evaluation score in this
comparison, motivating its use for RL. It outperformed the model trained with uncapped
Blend A by 0.0203 on the common metric. Applying caps to the same output can only lower
or preserve its reward; the improvement here comes from comparing the trained models.
This was not a uniform improvement: document render failures rose from
4 to 23, so inspect domain-level quality and render reliability alongside mean reward.
These results describe the tested training recipes, not a universal ranking or a
multi-seed result. [Comparison settings and summary](results/rl-reward-comparison.json).

## Benchmark results

Gym evaluations of **Qwen3.8-27B** on **test-mini**, judged by **GPT-6.1-Sol**, cover
the same 539 tasks across 15 rubrics. Both runs use temperature 0, up to 32,768 output
tokens per policy call, and one rollout per task. Revisions use category-level feedback;
generation and early stopping use linear reward.

| Maximum passes | Tasks scored | Successful final renders | Linear | Squared (offline) | Blend A (offline) |
| --- | --- | --- | --- | --- | --- |
| 1 | 539 | 503 | 0.7947 | 0.7001 | 0.6622 |
| 3 | 539 | 528 | 0.8815 | 0.8204 | 0.7267 |

Mean rewards weight every task equally, including zero rewards for code/render failures.
Neither run has masked samples. Squared and Blend A scores are offline rescoring of the
same final attempts, rather than separate collection runs with those rewards.

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

```

The Enroot tests check rendering, output validation, cleanup, and worker isolation.
Live judge quality requires model rollouts. The environment remains
experimental and `verified: false` pending broader baselining.

Adapted code is Apache-2.0; see [NOTICE](NOTICE), [LICENSE.upstream](LICENSE.upstream), and
[source provenance](tests/fixtures/provenance.json). Dataset licenses are separate.
