# Authoring visual agent tasks

Tasks live in `data/tasks/<category>.json` (a JSON list per category). Replication tasks also have a
hidden golden artifact in `data/golden/<task_id>/`, or for Figma tasks `data/assets/<task_id>/design.json`.
Their reference images are rendered from the golden by `build_dataset.py references` and written to
`data/assets/<task_id>/`. Use the exemplars `website-open-01`, `svg-rep-01` and `figma-rep-01` as models.

```bash
python resources_servers/visual_agent/build_dataset.py validate
python resources_servers/visual_agent/build_dataset.py references --only <task_id> ...   # needs OpenSandbox (env.yaml)
python resources_servers/visual_agent/build_dataset.py rows                               # data/visual_agent_tasks.jsonl
```

## Categories and modes (MiMo-V2.6 §4.2.3)

| category | artifact kind | typical deliverable |
|---|---|---|
| `website` | `html` | marketing pages, portfolios, docs, blogs |
| `interactive_app` | `html` | working tools: editors, trackers, planners, calculators |
| `game` | `html` | playable canvas/DOM games with controls, scoring, game over, restart |
| `3d_scene` | `html` (+ `"vendor": ["three"]`) | three.js scenes, interactive or animated |
| `slides` | `slides` | one HTML deck, each slide a `.slide` element of exactly 1280x720 |
| `svg` | `svg` | icons, logos, illustrations, infographics, animated loaders (SMIL/CSS) |
| `video` | `video` | MP4 made programmatically (Pillow/numpy frames + ffmpeg, canvas recording, ...) |
| `figma` | `html` (replication) / `svg` (open-ended) | implement a Figma file; or design a Figma-importable layered SVG |

`open_ended` tasks ask for an original design that satisfies a request. `replication` tasks give a
visual target and ask for a faithful rebuild; they are scored 60% by pixel similarity to the
reference and 40% by the rubric.

## Task fields

```json
{
  "task_id": "website-open-01",            // <category-short>-(open|rep)-NN; short = website, app, game, 3d, slides, svg, video, figma
  "category": "website",
  "mode": "open_ended",                    // or "replication"
  "title": "Short human title",
  "prompt": "The user request (no harness text; it is added automatically)",
  "artifact": {"kind": "html", "entry": "index.html"},   // file the grader opens inside /workspace/output/
  "viewport": {"width": 1280, "height": 800},              // html only; replication html is rendered at this viewport, full page
  "checks": {"mobile": true, "animation": false},          // extra automatic checks (see below)
  "vendor": ["three"],                     // optional: "three" (three.js r186 + addons) or "chartjs" (Chart.js 4.5)
  "assets": ["data.csv"],                  // optional files from data/assets/<task_id>/ copied to /workspace/task/
  "reference_images": ["reference.png"],   // replication only; rendered by `references`
  "golden": {"entry": "index.html"},       // replication only (not for figma): entry file inside data/golden/<task_id>/
  "reference_frame_times": [0.5, 2.0],     // video replication: timestamps of the reference frames (one per reference image)
  "reference_viewports": [{"width": 1280, "height": 800}, {"width": 390, "height": 844}],   // responsive html replication: one viewport per reference image
  "video_spec": {"width": 1280, "height": 720, "min_duration_s": 5, "max_duration_s": 7, "min_fps": 24},   // every video task
  "slides_spec": {"min_slides": 6, "max_slides": 8},       // slide decks
  "interaction_hints": "Grader-only guidance: which keys/clicks exercise the features",   // apps, games, interactive 3D
  "rubric": [{"id": "R1", "type": "instruction", "weight": 1, "criterion": "..."}]
}
```

`checks.mobile` renders at 390x844 as well, and the automatic layout item then also fails on
horizontal overflow on mobile. `checks.animation` requires the artifact to change on its own over
1.5 s (for scenes, loaders and animated SVG). Set it only when animation is expected with no user
input.

## Writing prompts

- Write the request like a real user with a clear brief: the subject, a named brand or person, exact
  headlines and labels the rubric will check, sections and their order, interactions, and style
  direction. A strong prompt runs 120-300 words.
- Do not describe the harness (paths, offline rule, bundled libraries); `prompts.policy_prompt` appends the
  I/O contract. Do not tell the agent how to check its work either: that is part of the task.
- Never use straight double quotes (`"`) in prompts: OpenCode escapes them. Use single quotes or ‘ ’.
- Everything must work offline with the installed fonts (Inter, Roboto, Open Sans, Lato, DejaVu,
  Liberation). No photos: ask for CSS, SVG or canvas visuals instead.
- Difficulty: aim for tasks a strong model completes with some effort. Each should need several
  distinct, checkable features, not a single widget.

## Writing rubrics (pointwise, binary)

Use 5-10 atomic items. Each item must be decidable by looking at renders and interacting, without
reading the author's mind. Types:

- `instruction`: a specific requirement from the prompt (quote exact strings when the prompt fixes them).
- `interaction`: a behavior that must work when exercised (say what to do and what should happen).
- `layout`: alignment and spacing, no overlap or clipping, responsive behavior, fit inside the frame.
- `aesthetic`: concrete visual quality: palette cohesion, typographic hierarchy, polish. Phrase it
  as what a professional designer would accept.
- `fidelity` (replication): the rebuild matches the target in structure, text, colors, typography
  and shapes.
- `runtime`: only for task-specific runtime needs (e.g. the game loop keeps running after game over).
  Generic checks are automatic, so do not add items for them: JS errors, failed or external
  requests, blank renders, desktop/mobile horizontal overflow, slide overflow and size, video
  duration/size/fps, SVG raster embedding.

Use weight 2 for the core requirements and 1 for the rest. Aim for balance: an artifact
that renders but ignores the brief should score low, and an excellent one should pass
everything.

## Difficulty bar (v2)

v1 was too easy for current models: Qwen3.8-Flash-Next averaged 0.89, and 70/100 tasks scored at
least 0.95. The v1 data is not in the repository. v2 tasks must separate strong models:

- **Breadth and depth.** Each task needs 8-15 independently checkable requirements, not 4-6.
  Apps and sites need real state: multiple views (hash routing), create/edit/delete with
  persistence, undo, search + filter + sort together, keyboard shortcuts and focus handling,
  validation with specific rules and messages, empty/error/loading states, responsive layout.
- **Precision.** Fix exact expected outcomes in the prompt: numbers the page must compute
  (totals, dates, statistics, physics), exact strings, counts and orderings. Rubric items
  check them exactly. Include edge cases (empty input, boundary values, invalid data).
- **Games.** Complete games: several levels or waves, rule-driven AI, combos or special
  rules, win and lose paths, pause and restart, persistent high scores, and a seed or
  deterministic mode so the judge can reproduce states.
- **3D.** Non-trivial systems: procedural generation with stated constraints, picking /
  raycasting interaction, animation systems with exact rates, camera paths, UI panels that
  drive the scene. Stay within software WebGL (modest geometry).
- **Replication.** Denser targets: longer pages, more text and components, charts with many
  labeled points, gradients, layered illustrations. Use responsive targets
  (`reference_viewports` with desktop + mobile references rendered from one golden page with
  media queries) for most website, app and Figma replications; 2-4 slides per deck; 5-6
  frames for videos.
- **Rubrics.** 10-12 atomic items (`validate` caps rubrics at 12), weight 2 on the hardest. Fewer generic aesthetic items;
  when you keep one, make it concrete ("consistent 8px spacing scale, no orphaned single
  words in headings, text contrast at least 4.5:1").
- **Feasibility.** A strong model should be able to finish in under an hour in the sandbox,
  offline, with the installed tools.

## Golden artifacts (replication)

- `html`: a self-contained page at `data/golden/<id>/<entry>`, deterministic (no random values,
  no dates, no animation), system fonts only, no images (CSS/inline SVG only), at most about
  2000 px tall at the task viewport. The reference is a full-page screenshot at the task viewport.
- 3D replication: a three.js page importing `./vendor/three/three.module.js` through an import
  map. Render one static frame with a fixed camera and no animation loop (or call render() once).
- `svg`: a hand-written SVG with a viewBox. The reference is rendered with its long side at 1000 px, a
  non-integer scale: overlap shapes that meet edge to edge by a unit or two, or anti-aliasing leaves a
  hairline seam.
- `slides`: `slides.html` with 1-3 `.slide` sections of 1280x720, one reference per slide
  (`reference_images: ["slide_1.png", "slide_2.png"]`, in order).
- `video`: `make_video.py` (Pillow/numpy + ffmpeg via subprocess, deterministic) writing `video.mp4`
  (`golden.entry`), with 3-4 `reference_frame_times` and matching `frame_N.png` reference names.
- `figma`: only `data/assets/<id>/design.json`, in Figma REST format (see `figma_render.py` for the
  supported nodes and properties). The golden HTML is generated from it. Frames are 390 px wide
  (mobile) or 1280-1440 px wide (desktop); set `viewport` to the frame size. For a responsive
  design, put two top-level frames on the page (e.g. `Desktop` 1280 wide and `Mobile` 390 wide).
  The golden shows each frame at viewports closest to its width. Set `reference_viewports` to
  `[{"width": 1280, "height": <desktop frame height>}, {"width": 390, "height": 844}]` and
  `reference_images` to `["desktop.png", "mobile.png"]`.
- Responsive html goldens (website/app): one page with media queries; list the viewports in
  `reference_viewports` in the same order as `reference_images`.

After rendering, view `data/assets/<id>/*.png` and check that `data/assets/<id>/reference_info.json` reports
`self_similarity` ≥ 0.99.
