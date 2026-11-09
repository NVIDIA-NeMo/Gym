# Vision2Code

Vision2Code is a NeMo Gym environment for training models to understand images through
reconstruction. A policy studies a reference image and writes Python to recreate it.
The environment executes the code in an isolated renderer, compares the result with the
reference, and returns a configurable reward and optional feedback for revision.

Derived from the [Vision2Code benchmark](https://github.com/image2code/vision2code), this
environment builds on its image-to-code task and domain rubrics to support reward
experiments for reinforcement learning. It provides judge-only rewards, pixel-and-judge
blends, semantic score caps, and a spatial-task reward override. Evaluation uses the same
rendering and verification pipeline to measure the resulting policies.

**Reference image → Python code → isolated rendering → judge and pixel scores → training reward.**

Use your own image manifests with the supported domain rubrics or the `generic` rubric.
The included benchmark examples and test-mini evaluation are starting points; training
data is supplied separately through the manifest and dataset configuration.

| Training control | What it lets you study |
| --- | --- |
| [Reward shaping](#rewards) | Linear or squared judge rewards versus Blend A's semantic and pixel signals. |
| [Capped judge terms](#rewards) | Penalties for semantic mistakes such as wrong molecular structure or circuit connectivity. |
| [Spatial reward override](#rewards) | Squared uncapped judge reward on `spatialvlm_qa`, with capped Blend A on other tasks. |
| [Feedback and revisions](#multi-pass-revisions) | Image, rationale, and numeric feedback; stopping criteria; last-pass or best-pass selection. |
| [Task and prompt configuration](#prepare-tasks) | Training images, rubric variants, and optional generation instructions. |

The environment supplies rollouts and rewards; the connected training framework owns
policy updates, optimization, and checkpointing.

## Rubrics

Choose a rubric for each reference image in your training or evaluation manifest. Domain
rubrics cover chemistry, physics, documents, charts, geometry, graphs, and spatial scenes;
`generic` provides an equal-weight rubric for other images. The judge compares the
reference and recreation using the following criteria:

| Rubric | What it judges |
| --- | --- |
| `ChartQA` | Chart values, structure, labels, and completeness. |
| `dvqa` | Chart values, layout, label accuracy, and legibility. |
| `figureqa` | Chart information, structure, text, and visual completeness. |
| `matplotlib` | Plot layout, plotted elements, ticks, annotations, and styling. |
| `ChemVQA-2K` | Molecular structure, chemical symbols, bonds, and readability. |
| `ChemVQA-2K-spacing-v1` | Chemical fidelity plus clearance between labels, bonds, charges, and subscripts. |
| `DocVQA` | Text content, document layout, legibility, and key visual elements. |
| `EEE-Bench` | Circuit connectivity, components, labels, directions, readability, and completeness. |
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

`DocVQA` and `EEE-Bench` retain the original category definitions and prompt notes.
Two optional variants preserve the modified prompts: `DocVQA-v2` explicitly penalizes
visible layout drift and background/text/graphic color mismatches, with unchanged weights.
`EEE-Bench-v2` adds background-color fidelity at 10%, reduces circuit topology from 35%
to 30%, and reduces readability/completeness from 15% to 10%. Select a variant explicitly
in the manifest's `rubric` field. Existing tasks that should keep the prior modified
behavior must use these `-v2` names before preparation or reverification. These variants
use the same datasets; they are not additional benchmark domains. Gym's configured reward
mode and image-degeneracy checks still apply, so matching category definitions alone does
not establish end-to-end parity with another evaluation setup.

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

[`container/`](container/) pins Python, system dependencies, and rendering libraries:
matplotlib, NumPy, Pillow, pandas, NetworkX, RDKit, Schemdraw, SciPy, Seaborn, Shapely,
svgwrite, CairoSVG, and ReportLab, with DejaVu and Liberation fonts. Rebuild the `.sqsh`
after changing these dependencies. The library list is included in model inputs only
when preparation uses `--include-library-instructions`.

[`enroot/`](enroot/) defines its runtime mounts. Each attempt gets a fresh sandbox with a
read-only rootfs, a private `/workspace`, and network/PID isolation. System libraries and
fonts are readable from the container image; host `/usr`, `/lib`, `/lib64`, and `/etc/fonts`
are not mounted over them, so workers using the same image use the same rendering dependencies.
Reference files and host credentials are not mounted. Enroot does not enforce per-attempt
CPU/memory limits; the worker or scheduler must supply them.

The generated program receives `OUTPUT_PATH`, `TARGET_WIDTH`, and `TARGET_HEIGHT`.
The launcher redirects `plt.savefig()` and `Figure.savefig()` to `/workspace/output.png`,
sets the figure size to the target dimensions at 100 DPI, and overrides explicit
`bbox_inches` and padding arguments. The default output is PNG; an explicit `format=`
argument is preserved. Matplotlib global cropping settings in `rcParams` remain effective
and can change the output dimensions. Other drawing libraries must save their output at
`OUTPUT_PATH` with the target dimensions themselves. Supported single-frame output formats
are PNG, JPEG, WebP, GIF, TIFF, and BMP; filenames do not determine validation or image MIME types.
The program should recreate the background too: transparency is not a
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

By default, preparation uses the short base user prompt without appended dimensions,
sandbox restrictions, or a library list. To append only `Target canvas: W x H pixels.`,
use `--include-canvas-dimensions`. The existing rendering and library additions remain
independently opt-in and can also be combined with that flag:

```bash
python -m resources_servers.vision2code.prepare \
  resources_servers/vision2code/data/manifest.jsonl /data/tasks.jsonl \
  --include-rendering-instructions --include-library-instructions
```

`--include-rendering-instructions` adds target dimensions, output-path requirements, and
sandbox restrictions. `--include-library-instructions` lists the installed libraries and
fonts. The base system prompt still requires code that saves to `OUTPUT_PATH`. Regenerate
existing prepared JSONL files to change their prompts; server config changes do not rewrite them.

Three presets select preparation and runtime settings together:

| Config | User prompt additions | Passes | Reward | Revision feedback |
| --- | --- | --- | --- | --- |
| [vision2code_evals.yaml](configs/vision2code_evals.yaml) | None | 1 | Uncapped Blend A | Single pass |
| [vision2code_feedback.yaml](configs/vision2code_feedback.yaml) | Canvas dimensions | 3, best reward | Uncapped Blend A | Previous render, judge rationale, judge and pixel scores |
| [vision2code_training.yaml](configs/vision2code_training.yaml) | Canvas dimensions | 1 | Capped Blend A; squared uncapped judge on `spatialvlm_qa` | Single pass |

The evaluation user prompt is exactly `Generate Python code to recreate this image. Return code only.`
All presets retain the system prompt's `OUTPUT_PATH` requirement. They use up to three
unchanged judge requests when judgment validation fails. The feedback preset stops at a
perfect final judge score, independently of the blended reward.

Pass the same config to preparation and `gym env start`:

```bash
python -m resources_servers.vision2code.prepare \
  /data/manifest.jsonl /data/tasks.jsonl \
  --config resources_servers/vision2code/configs/vision2code_feedback.yaml
```

Explicit `--include-*` and `--no-include-*` flags override preset preparation options.
Set `VISION2CODE_EVAL_TASKS` to your prepared JSONL when loading the evaluation preset.

Paths in the manifest are relative to the manifest file. Preparation sends the reference
as an embedded image in the policy input; its actual filesystem path is hidden from the
policy model and unavailable inside the rendering sandbox. The resource server keeps the
path, dimensions, and checksum in verifier metadata so it can check the reference and score
the output. Keep those source files accessible to the resource server. Prepared
[example tasks](data/example.jsonl) and [real rollouts](data/example_rollouts.jsonl) are included.

## Training workflow

Choose a reward recipe, prepare a training manifest, and supply the resulting tasks and
environment config to your NeMo Gym training integration. For example:

```bash
export VISION2CODE_TRAIN_TASKS=/data/vision2code/train.jsonl
python -m resources_servers.vision2code.prepare \
  /data/vision2code/train-manifest.jsonl "$VISION2CODE_TRAIN_TASKS" \
  --config resources_servers/vision2code/configs/vision2code_training.yaml
```

Use `resources_servers/vision2code/configs/vision2code_training.yaml` as the environment
config in your training integration, with the renderer and policy/judge endpoints
configured as below. This preset uses capped Blend A on non-spatial tasks and squared
uncapped judge reward on `spatialvlm_qa`; its dataset entry reads `VISION2CODE_TRAIN_TASKS`.
Other reward choices are described under [Rewards](#rewards). Multi-pass revisions are
optional and configured independently of the reward formula.

Keep training and held-out manifests separate. Compare trained policies with a fixed
evaluation reward, judge, prompt, and sampling setup so changes in the measurement do
not obscure the effect of the training reward. Record the reward recipe and evaluation
settings alongside each checkpoint.

## Connect models and collect rollouts

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
retains the reference, previous code, and feedback. By default, each successful render is
also embedded in the next revision message; sandbox files are not shared across passes.
Each pass submits a complete replacement program.

Agent configuration controls the feedback and stopping behavior:

```yaml
include_rendered_image: true
include_judge_scores: false
include_pixel_scores: false
stop_on: reward  # Also: judge_score or final_judge_score
minimum_scores: {}
require_same_dimensions: true
```

Numeric feedback is off by default to avoid encouraging reward hacking. Enable judge
scores to send the raw 0–5 aggregate and category scores, or pixel scores to send foreground
color similarity, foreground overlap, edge overlap, and RGB overlap. Pixel feedback can be
computed with any reward mode; enabling it does not change the reward calculation.
`stop_on: judge_score` stops at a raw judge score of exactly 5, independently of pixel
scores, caps, or reward shaping. `stop_on: final_judge_score` uses the rounded final
judge rating, including rubric caps, before any pixel blending. `stop_on: reward` retains
the reward-1 stopping rule.

To stop when quality thresholds are met, configure `minimum_scores`, for example:

```yaml
minimum_scores:
  overall_rubric_score: 4.5
  foreground_tolerant_f1: 0.9
  edge_tolerant_f1: 0.9
require_same_dimensions: true
```

All configured thresholds must pass. `rgb_overlap` is also supported. Pixel thresholds
range from 0 to 1; `overall_rubric_score` ranges from 0 to 5 and uses the final judge
rating before blending. Thresholds request pixel metrics even when numeric feedback is
disabled, without exposing those metrics to the policy. A perfect score selected by
`stop_on` can stop independently of the thresholds. Empty `minimum_scores` disables
threshold-based stopping.

The agent stops at the configured success criterion, the pass limit, or a masked/structured failure.
`result_selection: last` returns the last attempt. `result_selection: best` returns the
successful render with the highest configured reward; ties select the earliest pass.
The selected response, reward, and artifacts come from the same pass. All token usage
and trajectory turns remain recorded, and `selected_pass` identifies the returned attempt.
Masked and structured failures keep their failure routing. If no render succeeds, the
last failure is returned. The rollout's `attempts` list records each pass's reward,
feedback, and artifact paths. Each extra pass costs a
policy call and render, plus a judge call for a valid image; token limits apply per call.

Malformed judge JSON or invalid category scores trigger up to two additional evaluations
by default. `judge_repair_mode: retry` repeats the original request unchanged, including
its temperature and token budget. Configure `judge_repair_attempts` on the resources
server (set `0` to disable), `repair` to correct the previous response, or `rejudge`
to request a fresh evaluation with a bounded JSON schema. Repair and rejudge reuse the
original reference, render, and rubric at temperature 0; `judge_repair_max_output_tokens`
optionally overrides their token budget. Rejudge requires an endpoint supporting
JSON-schema structured output. Exhausted recovery masks the sample; partial scores are
never accepted. Judge artifacts retain each response and validation error.

Transport retries are separate: `judge_transport_attempts` defaults to three attempts
for HTTP 408, 409, 429, 500, 502, 503, and 504, timeouts, and connection failures.
The first two retries wait five and ten seconds. A transport retry never changes the
request or adds a repair prompt.

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

The [training preset](configs/vision2code_training.yaml) uses capped Blend A on all
rubrics except `spatialvlm_qa`. With `spatial_uncapped_judge_squared: true`, that rubric
instead receives `(L / 5)²`, using the raw weighted judge score without caps or rounding
and without pixel terms. For example, an uncapped judge score of 4/5 gives 0.64 even if
a rubric cap lowers the diagnostic final judge rating. Code/render failures still give
zero, and infrastructure/judge failures remain masked. This does not square the blend
or just its judge term. Prepare the training manifest with this preset, set
`VISION2CODE_TRAIN_TASKS` to the resulting JSONL, and pass the preset to the training
integration's Gym environment configuration. It defines the environment reward, not
an optimizer or a training job.

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

## Benchmark results

The evaluation presets use uncapped Blend A for single-pass or feedback-driven runs.
The historical results below were collected before these presets and the renderer update;
they are retained as reference measurements, not scores for the current presets.

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

Code errors, execution timeouts, and missing/invalid images receive zero reward. Infrastructure,
reference-image, and judge failures are masked and excluded from scored aggregates.
Each attempt saves the Python code, reference, generated image, execution diagnostics,
judge output, and result in its own artifact directory.

Outputs must be regular, readable, nonanimated images of at most 32 MiB and 25 million pixels.
TIFF and BMP artifacts are encoded as PNG for model image inputs; saved artifacts retain
their original format.
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
