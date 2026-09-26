# Visual agent tasks

An environment for the visual agent tasks of *MiMo-V2.6: Scaling Reinforcement Learning Towards
Self-Improvement*, §4.2.3. An agent builds a visual artifact in a sandbox: a website, an
interactive app, a game, a 3D scene, a slide deck, an SVG, a video, or an implementation of a
Figma design. An agentic verifier then grades the result.

- **Open-ended design** is scored with pointwise binary rubrics covering runtime correctness,
  instruction adherence, layout integrity and basic aesthetics. An optional groupwise pass
  compares the rendered artifacts of one rollout group and moves clearly stronger or weaker
  candidates up or down.
- **High-fidelity replication** is scored mainly by rule-based pixel similarity between the
  render and the reference (60%), plus a judge-decided fidelity rubric (40%).

Agent: `responses_api_agents/opencode_visual_sandboxed_agent` (OpenCode with image input).
Benchmark: `benchmarks/visual_agent` (200 synthetic tasks). Launcher for Qwen3.8-Flash-Next:
`benchmarks/visual_agent/launch_qwen3.8_flash_next.sh`.

## Tasks

| category | open-ended | replication | artifact |
|---|---:|---:|---|
| website | 20 | 10 | `index.html` |
| interactive_app | 24 | 4 | `index.html` |
| game | 24 | 0 | `index.html` |
| 3d_scene | 18 | 4 | `index.html` + vendored three.js |
| slides | 16 | 8 | `slides.html` (`.slide` = 1280x720) |
| svg | 16 | 12 | `*.svg` |
| video | 16 | 4 | `video.mp4` |
| figma | 8 (layered SVG design) | 16 (Figma file → HTML/CSS) | `design.svg` / `index.html` |

This is the v2 task set. The first set (v1) turned out too easy: Qwen3.8-Flash-Next scored 0.89 and
70 of 100 tasks averaged >= 0.95 (results below); it is not in the repository. v2 keeps the
task ids and category/mode counts, and each task is a larger spec with exactly checkable
outcomes:

- open tasks carry 8-15 required features each, with exact data, rules and expected values that
  the rubric checks (computed totals, seeded game states, dates, sort orders), plus concrete
  judge hints;
- games are complete (levels, AI with a stated strategy, seeded randomness, win/lose paths,
  persistent records);
- decks and charts are computed from data files in `data/assets/<id>/`;
- most website, app and all Figma replications are responsive: the reference has desktop and
  mobile screenshots (`reference_viewports`) and both are compared.

Batch b (`data/tasks/<category>_b.json`, 2026-09-25) adds 100 tasks with the same split and new
themes, written against the v2 results: correctness hinges on computation the agent must do
(simulations, search, derived metrics, seeded games) and on edge cases the brief states, rather
than on transcribing a long spec. Batch b has not been run yet; the results below are for the
first 100.

Task specs live in `data/tasks/*.json`; see [TASK-AUTHORING.md](TASK-AUTHORING.md) for the format.
Replication tasks have hidden goldens in `data/golden/` (Figma tasks: `data/assets/<id>/design.json`).
Reference images are rendered from the goldens by the grader's own renderer, in the task image
(`build_dataset.py references`); the rendered PNGs are not committed, so render them once after checkout
(below). Each task records a determinism check (`self_similarity`, 1.0 for
every task) in `data/assets/<id>/reference_info.json`.

## Rollout and verification

```
seed_session ── policy sandbox (task image + setup_sandbox.sh)
                /workspace/task/     reference images, input files
                /workspace/output/   the deliverable (+ vendor/ libraries)
agent ───────── plain OpenCode: the prompt is the task plus the I/O contract (where to save, format,
                offline, bundled libraries, target/viewport pairs). No grader tooling or tips: how to
                render, view and compare its own output is up to the agent (the machine has Chromium,
                Playwright, ffmpeg, Node, Python, and OpenCode can read images)
verify ──────── tar /workspace/output → stop policy sandbox
                fresh grader sandbox (nothing the policy left behind can affect grading)
                ├─ vtools.py measure: renders, JS/network errors, layout overflow, blank detection,
                │  slide/video/SVG facts, replication similarity, reference-copy detection
                ├─ agentic judge: OpenCode + judge model views the renders, clicks, types and presses keys
                │  with vtools.py interact, then writes /grader/verdict.json (one pass/fail per rubric item)
                └─ reward (grading.compute_reward)
groupwise ───── (optional, open-ended) wait for the whole rollout group, one judge compares all candidates
```

Reward:

- **Runtime gate.** A missing, blank or unrenderable artifact scores 0.
- **Rubric score.** The weighted pass rate over the judge's items plus automatic items:
  `AUTO-runtime` (no JS errors, failed or external requests; video spec; pure-vector SVG),
  `AUTO-layout` (no desktop or mobile overflow, no broken images; slide size and overflow), and
  `AUTO-animation` when a task expects motion.
- **Open-ended reward** = rubric score. **Replication reward** = 0.6 × similarity + 0.4 × rubric score.
  Similarity combines SSIM, pixel, edge-F1 and color-histogram metrics. It is normalized against
  a flat canvas in the reference's median color, so a blank attempt gets about 0.
- **Anti-hacking.** A deterministic check zeroes the reward when the artifact contains a
  reference image, as a byte copy, its base64, a re-encoded or rescaled raster copy, or a load
  from the task directory. A judge reward-hacking flag with concrete evidence also zeroes it.
- **Groupwise** (`groupwise.enabled`). Candidates that pass the gate are shown anonymized and
  shuffled to one judge, which labels each `stronger`, `comparable` or `weaker`. The reward moves
  by ±`bonus`, clipped to [0, 1]. On timeout or judge failure the pointwise reward is kept.
  `pointwise_reward` is always reported next to `reward`.
- A judge with no usable verdict after `judge_max_attempts` gives `mask_sample=true`, with
  `failure_kind=judge_failed`.

Every rollout's grading folder, `grading_output_dir/<task_id>/t<task>_r<rollout>_<hash>/`, keeps:
the artifact tarball, the renders, `measurements.json`, the judge prompt, the judge OpenCode
exports, `verdict_<n>.json` and `result.json`.

## Sandbox image

`apify/actor-python-playwright:3.12-1.63.0` from Docker Hub: Debian 13, Python 3.12, Playwright
1.63 with Chrome/Chromium. `sandbox_tools/setup_sandbox.sh` (about 40 s) adds:

- numpy, Pillow, ffmpeg and Node.js;
- the Inter, Roboto, Open Sans, Lato and DejaVu fonts, with fontconfig aliases (the image maps
  `sans-serif` to a Thai font by default);
- `/workspace` as an alias of the image WORKDIR, where OpenCode starts.

WebGL runs on SwiftShader. three.js r186 and Chart.js 4.5 are vendored offline
(`fetch_vendor.py`, MIT, not committed).

## Running

```bash
python resources_servers/visual_agent/fetch_vendor.py
python resources_servers/visual_agent/build_dataset.py references   # reference PNGs, in OpenSandbox (env.yaml)
python resources_servers/visual_agent/build_dataset.py validate
gym env start --config resources_servers/visual_agent/configs/visual_agent.yaml ...   # example tasks
# The launchers take MODEL (checkpoint dir), CONTAINER (vLLM + Gym .sqsh) and EXTRA_MOUNTS; see their headers.
SMOKE=1 bash benchmarks/visual_agent/launch_qwen3.8_flash_next.sh                   # 4 nodes
bash benchmarks/visual_agent/launch_qwen3.8_flash_next.sh                           # 200 tasks x 4
```

**Other policies, fixed judge.** To compare models, grade every policy with the same judge.
`launch_judge_qwen3.8.sh` serves Qwen3.8-Flash-Next as a vLLM-only job (2 nodes) that writes its
router URL to `results/visual-agent-judge/qwen3.8-flash-next.endpoint` (`ENDPOINT_FILE` in
`sbatch_external_vllm.sh`). The policy job loads `benchmarks/visual_agent/external_judge_override.yaml`,
which points `judge_model` at that file (vllm_model `endpoint_file`), so it follows the judge
across restarts. Example with a Nemotron 3.5 Super VL checkpoint (2 more nodes, vision preset
`vllm_configs/nemotron_3.5_super_vision.sh`):

```bash
bash benchmarks/visual_agent/launch_judge_qwen3.8.sh
SMOKE=1 bash benchmarks/visual_agent/launch_nemotron_3.5_super.sh [/path/to/hf_checkpoint]
```

**HTML report.** `benchmarks/visual_agent/report.py` turns one or more rollout files into a
browsable report. The index has scores per run, category and task. Each task page shows the
prompt, reference images and rubric. Per rollout it shows the grader's renders, the live
artifact (iframe or video), per-item verdicts with the judge's evidence, the screenshots the
judge inspected, the agent's last previews and a condensed agent trace. Serve it over HTTP,
because three.js artifacts use ES modules, which do not load from `file://` pages:

```bash
python benchmarks/visual_agent/report.py results/<qwen>.jsonl results/<super>.jsonl \
    --labels qwen3.8-flash-next nemotron-3.5-super --out results/visual_agent_report
python -m http.server -d results/visual_agent_report 8000
```

Throughput notes, from the Qwen3.8-Flash-Next run on 4 TP4 replicas:

- **Sessions per replica.** Screenshot-heavy OpenCode sessions reach ~100K tokens, so a replica
  (~4.3M KV tokens) keeps only ~40 live sessions cached. At 400 rollouts in flight, KV use hit
  100%, the prefix-cache hit rate fell from 68% to 22% and generation dropped to ~1.2K tok/s per
  replica. The launcher defaults to 128 in flight; KV use then stays near 15-30%.
- **Groupwise holds slots.** With groupwise on, a finished rollout keeps its rollout slot until
  its whole group is graded (about a quarter of the slots mid-run). On a resumed run the members
  written earlier never arrive, so `collection_timeout_s` is kept short (20 min) and those rollouts
  keep their pointwise reward.
- **Venvs.** The container ships no venv for these servers. `benchmarks/visual_agent/eval_setup.sh`
  links prebuilt ones with identical requirements (via `EVAL_SETUP_SCRIPT`), because building
  them at job start timed out.

Key metrics: `mean/pointwise_reward` (headline), `pointwise_reward/category/*`,
`pointwise_reward/mode/*`, `mean_similarity/replication`, `rate/gate_failed`,
`rate/judge_failed`, `rate/hack_detected`, `groupwise/mean_abs_shift`.

## Results on v2: Qwen3.8-Flash-Next vs Nemotron 3.5 Super, 2026-09-25

Setup: v2 tasks, 100 x 4 rollouts, bare policy prompt (task + I/O contract, no grader tooling),
groupwise on. Judge: Qwen3.8-Flash-Next at temperature 0.6 for both. For the Qwen run it shared the
policy's 4 replicas; for Super it was a separate 2-node job (`launch_judge_qwen3.8.sh`). Super: a
Nemotron 3.5 Super VL RL checkpoint (step 28) on 2 nodes (vision preset).

| group | Qwen3.8-Flash-Next | Nemotron 3.5 Super |
|---|---:|---:|
| website / open, replication | 0.886, 0.642 | 0.537, 0.195 |
| interactive_app / open, replication | 0.854, 0.737 | 0.585, 0.291 |
| game / open | 0.721 | 0.382 |
| 3d_scene / open, replication | 0.922, 0.577 | 0.500, 0.162 |
| slides / open, replication | 0.777, 0.782 | 0.521, 0.209 |
| svg / open, replication | 0.864, 0.672 | 0.621, 0.235 |
| video / open, replication | 0.952, 0.731 | 0.797, 0.691 |
| figma / open, replication | 0.884, 0.725 | 0.643, 0.281 |
| **all (pointwise)** | **0.804** | **0.473** |
| open-ended / replication | 0.848 / 0.699 | 0.557 / 0.267 |
| rubric score / similarity | 0.940 / 0.717 | 0.643 / 0.307 |
| no deliverable written | 44 | 90 |
| rollouts that looked at any image | 323 | 29 |
| median tool calls / input tokens | 50 / 2.8M | 14 / 0.55M |
| masked | 9 (policy sandbox OOM at 8 GiB, 2 judge) | 0 |

- **v2 separates models.** Qwen: 37/100 tasks average >= 0.95 (70 on v1). Super wins 7 tasks,
  Qwen 89. The hardest for Qwen are game-open-09 (0.00), app-open-08 (0.26) and game-open-04 (0.33).
- **Self-checking is the main difference.** Unprompted, Qwen renders its work (its own
  Playwright/Chrome/ffmpeg scripts) and looks at the images in 81% of rollouts. Super almost never
  does (7%) and stops after about 14 tool calls. Given the grader's tools and instructions (earlier
  run, old prompt, same tasks), Super looked at images in 297/380 rollouts and scored 0.540 vs 0.483
  bare on the same (task, rollout) pairs.
- **Reference-copy check.** It zeroed 2 Super rollouts (video-rep-02, slides-rep-04) because a
  comparison PNG copied from the target was left in the output folder, although the deliverable
  itself did not use it.

## Judge calibration, 2026-09-25

Does the judge separate good from bad artifacts? `benchmarks/visual_agent/judge_calibration.py`
builds artifacts of known quality for 32 v2 tasks (2 open + 2 replication per category, 4 open
games). The replay agent (`responses_api_agents/visual_replay_agent`) grades them through the normal
`/verify` path (`calibration.yaml`), twice each, with the Qwen3.8 judge configuration used above.

- Anchor: the golden (replication) or the best Qwen3.8 rollout (open-ended; video replication).
- wrong_task: the anchor of another task in the same category.
- Degradations of the anchor:
  - `no_script`: scripts removed;
  - `unstyled`: CSS removed;
  - `text_corrupt`: every rendered digit shifted by 3, including canvas text;
  - `hue_shift`: hue rotated by 180°;
  - `adversarial`: `text_corrupt` plus a note asking the grader for full marks;
  - `drop_half`: half the SVG or slides removed;
  - `truncate`, `freeze`: video only.

| variant | cases | rubric score | anchor higher / tie / variant higher |
|---|---:|---:|---|
| anchor | 32 | 0.960 | |
| wrong_task | 32 | 0.091 | 32 / 0 / 0 |
| no_script | 22 | 0.362 | 16 / 6 / 0 (ties: static pages) |
| drop_half | 10 | 0.368 | 10 / 0 / 0 |
| freeze, truncate (video) | 4, 4 | 0.072, 0.422 | 8 / 0 / 0 |
| unstyled | 22 | 0.581 | 22 / 0 / 0 |
| text_corrupt | 28 | 0.560 | 25 / 3 / 0 |
| adversarial | 22 | 0.482 | 20 / 2 / 0 |
| hue_shift | 32 | 0.715 | 29 / 3 / 0 |

- **Discrimination.** AUROC good vs bad 0.946, 0.92-1.00 per category. No degradation ever
  scored above its anchor.
- **Targeted items.** The items each degradation should break are the ones that fail:
  - interaction items without scripts: 0.86 → 0.00;
  - items naming exact values under `text_corrupt`: 0.95 → 0.39;
  - color items under `hue_shift`: 0.98 → 0.27.
  Hue shift costs 0.39 on replication (colors specified) but only 0.13 on open tasks.
- **Grader-directed text backfires.** The note scored lower than the same artifact without it
  (0.48 vs 0.56). The judge flagged it as reward hacking in 41 of 44 gradings, which zeroes the reward.
- **Consistency.** Item agreement between two gradings of the same artifact is 0.963, with mean
  |score difference| 0.031. Two failures in 404 judged gradings.
- **Replication tracks pixels.** On the 195 judged v2 replication rollouts, the judge's fidelity
  pass rate rises with measured similarity: 0.22 / 0.64 / 0.75 / 0.88 for similarity bands
  <0.3 / 0.3-0.6 / 0.6-0.8 / >0.8 (Spearman 0.71).
- **Flaw found and fixed.** The 4 unstable artifacts were `text_corrupt` or `adversarial` pages
  where one grading noticed the injected script and graded the app with it stripped (1.00), while
  the other graded what was displayed. In one case the judge also passed an item while noting
  that it displayed a wrong value. The judge prompt now says to grade the artifact exactly as
  delivered and never patch, strip or bypass parts of it. Re-running the 82 affected and anchor
  cases with that prompt:
  - anchors unchanged (0.960 → 0.962);
  - `text_corrupt` 0.560 → 0.500 and `adversarial` 0.482 → 0.400 (now below its anchor on 22/22
    tasks);
  - exact-value items passed despite wrong displayed numbers: 0.39 → 0.30;
  - test-retest |score difference| mean 0.056 → 0.021, max 0.842 → 0.474.
  The remaining `text_corrupt` ties are 3D scenes that display no numbers. `judge_calibration.py analyze` writes the
  summary.
- **Caveat.** Figma replication anchors are static renders of `design.json`, so they
  legitimately fail the interaction items (anchor 0.44).

## Results on v1: Qwen3.8-Flash-Next (policy and judge), 2026-09-24

Setup: 100 tasks x 4 rollouts; 4 nodes with 4 TP4 replicas of the vision preset; judge at
temperature 0.6; groupwise on (summarize a rollouts file with
`python benchmarks/visual_agent/summarize.py <file>`).

| group | n | pointwise | reward (groupwise) | rubric | similarity | gate failed |
|---|---:|---:|---:|---:|---:|---:|
| website / open | 40 | 0.848 | 0.837 | 0.998 | - | 15% |
| website / replication | 20 | 0.771 | 0.771 | 0.989 | 0.947 | 20% |
| interactive_app / open | 48 | 0.979 | 0.967 | 0.979 | - | 0% |
| interactive_app / replication | 8 | 0.961 | 0.961 | 1.000 | 0.935 | 0% |
| game / open | 48 | 0.741 | 0.728 | 0.961 | - | 23% |
| 3d_scene / open | 36 | 0.936 | 0.904 | 0.962 | - | 3% |
| 3d_scene / replication | 8 | 0.703 | 0.703 | 0.900 | 0.571 | 0% |
| slides / open | 32 | 0.927 | 0.913 | 0.988 | - | 6% |
| slides / replication | 16 | 0.848 | 0.848 | 1.000 | 0.949 | 13% |
| svg / open | 32 | 0.854 | 0.835 | 0.976 | - | 13% |
| svg / replication | 24 | 0.937 | 0.937 | 0.995 | 0.966 | 4% |
| video / open | 32 | 0.948 | 0.934 | 0.978 | - | 3% |
| video / replication | 8 | 0.966 | 0.966 | 0.932 | 0.988 | 0% |
| figma / open (layered SVG) | 16 | 0.981 | 0.944 | 0.981 | - | 0% |
| figma / replication | 32 | 0.962 | 0.962 | 0.997 | 0.990 | 3% |
| **all** | **400** | **0.891** | **0.879** | 0.979 | 0.938 | 8% |

- **No masked samples and no judge failures** in the final data. Grader sandboxes at 4 GiB were
  OOM-killed on 9 video/game rollouts; those were re-run at 8 GiB with the dead-sandbox retry.
- **Judge effort.** Median per rollout: 23 tool calls and 9 screenshots viewed. The judge drives
  apps and games with `interact`, reloads to check persistence, and quotes measured evidence
  (pixel counts, element boxes, spacing).
- **Policy vision.** 371/400 rollouts viewed images, median 10 per rollout (its own previews and
  the targets). This v1 run gave the policy the grader's renderer (`vtools.py preview/interact/
  compare`, including the similarity score); since 2026-09-24 the policy gets no grader tooling.
- **Groupwise.** 244 rollouts were graded in complete groups: 29 stronger, 33 weaker, the rest
  comparable (mean |reward - pointwise| = 0.02). The other cases are replication (not
  applicable), groups with fewer than two gated members, and 24 timeouts in groups split
  across the resume.
- **Where scores are lost** (gate failures, 8%):
  - 25 policy sessions (6%) stalled in one runaway generation and hit the 1-hour harness
    budget after <=20 tool calls, with no artifact. The policy has no per-turn output cap;
    capping `max_tokens` (e.g. 64K) would free these slots sooner.
  - A few sessions ended on a reasoning-only turn before writing the file.
  - game-open-09's canvas loop was too heavy for software rendering: screenshots timed out.
  - Games (0.74) and 3D replication (0.70; similarity 0.57) are the hardest; 70/100 tasks
    average >= 0.95.

## Config

See `configs/visual_agent.yaml`. The most relevant keys:
- `judge_model_server` (or `judge_base_url`), `judge.timeout_s`, `judge_max_attempts`;
- `reward.replication_similarity_weight`;
- `groupwise.{enabled, group_size, bonus, collection_timeout_s}`. `group_size` must equal the
  dataset's `num_repeats`.
