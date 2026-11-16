# ClawEval training tasks on the Harbor task shape

ClawEval task packages (video QA, OCR, search, edit, chart, image, webpage and workflow tasks over a fixture video
or image, each graded by its own `claw_eval` grader with an LLM judge) as plain Harbor task folders for the generic
[`harbor` environment server](../../../environment_servers/harbor/README.md), run and trained with Gym's
`opencode_agent` inside each task's image. The entry is `type = "train"`. The packages, the `claw_eval` package and
the judge route are NVIDIA-internal: the prepare script writes the tasks into this folder from a task root and a
ClawEval checkout you give it, and nothing of them is committed.

```sh
python -m benchmarks.claweval.prepare_harbor_env --tasks-root <root> --claw-eval-source <checkout>   # writes the tasks here
gym eval run claweval_train --agent opencode_agent --sandbox docker \
  --config environments/harbor/claweval_train/harbor_train.yaml \
  --model-type vllm_model --model <model> --model-url <url> --model-api-key <key>
```

## The task root

A task root holds one package per folder, `<root>/<task_id>/`, with `task.yaml` (the task id, which is the folder's
name, the category, the prompt, `sandbox_files` the agent sees, `sandbox_grader_files` and `local_grader_files` it
must not, the agent budget `environment.timeout_seconds` and the snapshot commands the grader runs), `grader.py`,
the fixtures and the grader's reference material. `--task-ids FILE` selects packages by id (one per line), for a
root larger than the training set. `--claw-eval-source` is a checkout holding `src/claw_eval`, the package the
graders import; it is copied into every task's verifier.

Each task is written as a `[[steps]]` task with one `solve` step: the prompt is the step's `instruction.md`, the
fixtures and a short `AGENTS.md` (how to look at the media with ffmpeg and the image-reading tool) are the step's
`workdir/`, and `tests/` holds the verifier (`test.sh` runs `judge_claweval.py`, which plays the package's snapshot
commands, runs its grader with the judge and writes `reward.json`), the package's `task.yaml` with its rubric and
reference answer, `grader.py`, the grader files and the `claw_eval` package. The server uploads `tests/` only after
the agent's session has closed, so the agent never sees the grading material. `task_data.json` names the task's
category and the digests of its prompt and scoring inputs (typed by `task_data.py` beside this README). The OpenCode
compaction plugins are staged under `workdir/.opencode-plugins/` for the opt-in overlay below. Fixtures are
hard-linked where they can be (`--copy-fixtures` copies them).

Every task's `environment/Dockerfile` is the same image (the Playwright Python image, so the webpage tasks have a
browser, plus ffmpeg and the Python packages the graders import). It has no `COPY`, so the server treats it as an
overlay and runs its `RUN` lines at every seed, which costs minutes of package installs per episode and needs
network from the sandbox; for a training run, build that image once, load it on the sandbox provider and pass
`--docker-image <ref>`, which writes it as the tasks' `[environment].docker_image` instead. The judge is the NVIDIA Inference Hub Gemini route
(`--judge-base-url`, `--judge-model`), written into each task's `[steps.verifier.env]` with the key as the
`${NVINFERENCE_API_KEY}` template, which the server resolves from its `secrets` (the overlay) or its environment.
There is no reference solution to play: the packages carry a reference answer for the judge, so
`gym dataset validate` scores this entry 0 and is not its check.

## Training

[`harbor_train.yaml`](harbor_train.yaml) is the deployment overlay: the judge key as a verifier-only secret, the
episode cap, `image_input`, training-token capture of OpenCode's model calls and the context and output budgets
OpenCode is told (`claweval_context_tokens`, `claweval_output_tokens`; the defaults are the Super 3.5 recipe's 49,152
and 8,192) with automatic compaction on. A NeMo RL run lists the rows and the server blocks a run of the entry
generates, then this overlay, as the [Agentic-VBench training entry](../agentic_vbench_train/README.md) describes:

```sh
gym eval run claweval_train --agent opencode_agent --sandbox docker \
  --config environments/harbor/claweval_train/harbor_train.yaml --limit 1 \
  --model-type vllm_model --model <model> --model-url <url> --model-api-key <key>
# results/harbor/claweval_train/tasks.jsonl and run_config_opencode_agent.yaml are written.
```

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
      - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
      - results/harbor/claweval_train/run_config_opencode_agent.yaml
      - environments/harbor/claweval_train/harbor_train.yaml
      # optional: environments/harbor/claweval_train/compaction_plugins.yaml
    sandbox: docker
```

To train on these tasks together with Agentic-VBench, Video-MME and video captioning in one run, see
[`environments/unified_video_agentic`](../../unified_video_agentic/README.md).

## Long episodes

The same two provisions as the Agentic-VBench training entry: the shared sandbox supervisor stops the agent and
its descendants at the time limit before the verifier runs, and [`compaction_plugins.yaml`](compaction_plugins.yaml)
(opt-in) loads the staged OpenCode plugins and the model server's compaction-summary recovery.

## What differs from the internal runner

The internal ClawEval RL runner gave OpenCode two sandbox-local MCP tools, `claw_ReadMedia` (bounded video frame
extraction returned as images) and `claw_BrowserScreenshot`. `opencode_agent` accepts only remote MCP servers, so
this entry relies on `image_input` instead: the model extracts frames with ffmpeg and reads the image files, as
the Agentic-VBench tasks do, and renders pages with the image's Chromium. The grader, the judge route and the
reward are the native ones.

## Validation status

Unit tests cover the prepare script (`tests/unit_tests/benchmarks/test_claweval_prepare_harbor_env.py`). Not
recorded yet: a `gym eval run` of the entry against a vLLM policy with token capture on; it will be added here
before the entry is used for a reported run.
