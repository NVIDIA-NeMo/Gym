# Agentic-VBench training tasks on the Harbor task shape

The training counterpart of [`agentic_vbench`](../agentic_vbench/README.md): generated video-editing tasks of the
four benchmark families (Repair, Assembly, Sequencing, Repurpose) as plain Harbor task folders for the generic
[`harbor` environment server](../../../environment_servers/harbor/README.md), run and trained with Gym's
`opencode_agent` inside each task's own image. The entry is `type = "train"`: its rows enter a training split, and
the 100 pinned benchmark tasks, the evaluation set, never do. The tasks themselves are not committed: the prepare
script writes them into this folder from a task root you give it.

```sh
python -m benchmarks.agentic_vbench.prepare_harbor_env --tasks-root <root>      # writes the tasks into this folder
gym dataset validate agentic_vbench_train --sandbox docker \
  --config environments/harbor/agentic_vbench_train/harbor_train.yaml           # plays each task's solution/
gym eval run agentic_vbench_train --agent opencode_agent --sandbox docker \
  --config environments/harbor/agentic_vbench_train/harbor_train.yaml \
  --model-type vllm_model --model <model> --model-url <url> --model-api-key <key>
```

## The task root

A task root has the pinned checkout's layout: `tasks/agentic_vbench_<family>/<task>/`, each task a Harbor task
folder of the benchmark's shape (`task.toml` with one `solve` step, `environment/Dockerfile` working in
`/workspace`, `steps/solve/{instruction.md,tests,solution,workdir}`). Task folder names must be unique across
families; a family folder may hold links to the tasks' real folders, which the prepare script resolves. Nothing
pins the count or revision: `--families` selects families, and the records of what changed land under
`.records/<task>/` as for the benchmark entry. A task whose Dockerfile starts `FROM` an image that is not on a
registry (generated tasks share a local base such as `localhost/avb-sdg-base:1`) needs that image on the sandbox
provider's daemon before the run builds the task images.

Each task is written with the benchmark entry's adjustments (Hub judge routing for Repurpose, the assembly judge
guard, bounded golden-asset retries and exit 75 for verifier-side outages, `/workspace/output` and
`/workspace/materials` as the judged and repair steps' artifacts), plus `task_data.json` naming its family (typed
by `task_data.py` beside this README) and the OpenCode compaction plugins under the step's `workdir/`.

`dataset.toml` gives the verifiers the published 3x allowance (`[gym.defaults] verifier_timeout_multiplier = 3.0`,
5400 s for a task that declares 1800 s) and leaves the agent budgets as the tasks declare them.

## Training

[`harbor_train.yaml`](harbor_train.yaml) is the benchmark entry's deployment overlay (the judge key as a
verifier-only secret, separate verification of the judged and repair tasks, the episode cap, `image_input`) plus
what training needs: training-token capture of OpenCode's model calls (the run-wide `token_id_capture` store and
the harness's `token_id_capture: true`; see
[Train with external agent harnesses](../../../fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx)),
the agent's time ceiling, and the context and output budgets OpenCode is told (`agentic_vbench_context_tokens`,
`agentic_vbench_output_tokens`; the defaults are the Super 3.5 recipe's 49,152 and 8,192) with automatic
compaction on. The model server must return token ids: `vllm_model_for_training.yaml`, or the training
framework's.

A NeMo RL run lists the rows and the server blocks a run of the entry generates, then this overlay:

```sh
# Once: materialize the rows and the run config (one episode runs; `--limit 0` is not a prepare).
gym eval run agentic_vbench_train --agent opencode_agent --sandbox docker \
  --config environments/harbor/agentic_vbench_train/harbor_train.yaml --limit 1 \
  --model-type vllm_model --model <model> --model-url <url> --model-api-key <key>
# results/harbor/agentic_vbench_train/tasks.jsonl and run_config_opencode_agent.yaml are written.
```

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
      - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
      - results/harbor/agentic_vbench_train/run_config_opencode_agent.yaml
      - environments/harbor/agentic_vbench_train/harbor_train.yaml
      # optional: environments/harbor/agentic_vbench_train/compaction_plugins.yaml
    sandbox: docker   # or the provider the run config names; the sandbox provider block goes in the overlay
```

NeMo RL carries its sampling values into every model call the trained agent makes and configures the capture
store itself; the overlay's `token_id_capture` values are the local-run defaults. The harbor server resolves each
row's task folder through the run config's taskset mapping, so the entry folder must stay where it was prepared.

### Harnesses

`dataset.toml` admits four harnesses, each with its own overlay because an overlay names its harness's block:

| Harness | Overlay | Rows and run config a run writes |
| --- | --- | --- |
| `opencode_agent` | [`harbor_train.yaml`](harbor_train.yaml), optionally `compaction_plugins.yaml` and `subagents.yaml` | `tasks_opencode_agent.jsonl`, `run_config_opencode_agent.yaml` |
| `pi_agent` | [`harbor_train_pi.yaml`](harbor_train_pi.yaml) | `tasks_pi_agent.jsonl`, `run_config_pi_agent.yaml` |
| `openclaw_agent` | [`harbor_train_openclaw.yaml`](harbor_train_openclaw.yaml) | `tasks_openclaw_agent.jsonl`, `run_config_openclaw_agent.yaml` |
| `codex_agent` | [`harbor_train_codex.yaml`](harbor_train_codex.yaml) | `tasks_codex_agent.jsonl`, `run_config_codex_agent.yaml` |

For Pi, OpenClaw or Codex, replace `opencode_agent` and `harbor_train.yaml` in the commands and the recipe above with the
harness and its overlay. The tasks, the verifiers, the server block and the capture store are the same; each
overlay sets the same capture opt-in, `image_input`, time ceiling and model budgets
(`agentic_vbench_context_tokens`, `agentic_vbench_output_tokens`) on its harness, where the harness has the key.
What differs:

- Codex and frames. Codex has no `image_input`: its `view_image` tool returns a frame inside the tool result, and
  the model server must send that image on (`inline_tool_output_images` on `vllm_model`, which the Codex overlay
  sets on `policy_model`). Without it the first frame Codex views fails the episode at `stage: agent`. Codex also
  has no per-call output key, so `agentic_vbench_output_tokens` does not reach it.

- Compaction. The compaction plugins and the model server's summary recovery are OpenCode's. The Pi overlay turns
  compaction off (`auto_compaction: false`), so an episode that fills the context ends there. OpenClaw compacts
  when its estimate nears `context_window`, and Codex below `model_context_window`. For all three, give the
  policy its whole training context.
- `subagents.yaml` and `compaction_plugins.yaml` configure OpenCode and load only with it.
- The runtime. Pi, OpenClaw and Codex install Node and their CLI into the sandbox at session start, from nodejs.org and
  npm. A sandbox without that access needs an image that carries the runtime: `sandbox_runtime_dir` in the
  [Pi](../../../responses_api_agents/pi_agent/README.md) and
  [OpenClaw](../../../responses_api_agents/openclaw_agent/README.md) READMEs, and "Runtime and model
  requirements" in [Codex's](../../../responses_api_agents/codex_agent/README.md).

One run trains one harness. The generated run configs name their servers after the taskset
(`harbor_agentic_5fvbench_5ftrain_agent` and `..._environment`), so listing two in one run's `config_paths` makes
the later replace the earlier; a single run that mixes harnesses needs those instances renamed in a copy.

Status: one episode of the first curriculum task ran through each added harness on rootless Podman sandboxes with
a hosted vision model, and each was verified with reward 1.0 with frames in the model's requests (Pi 20 of 23
calls, OpenClaw 6 of 9, Codex 36 of 41 with `inline_tool_output_images`). A training step with token capture
through them has not been recorded here yet.

Per-family numbers for a run of the entry: `python -m benchmarks.agentic_vbench.family_metrics
results/harbor/agentic_vbench_train/rollouts.jsonl --entry environments/harbor/agentic_vbench_train`.

## Long episodes

Two things long video-editing episodes need were validated on the internal training path and are provided here
without new code where upstream already has them:

- **Stopping the agent at its time limit.** The shared sandbox supervisor
  (`nemo_gym/agents/utils/process_supervisor.sh`, used by every sandboxed harness session) stops the harness's
  process group and every descendant it can reach, including OpenCode subagents and tools that detached, and
  confirms termination before the server captures artifacts or verifies; unconfirmed termination fails the
  episode instead of racing the verifier. The verifier therefore grades what the agent had delivered when its
  budget ran out, never a workspace still being written.
- **Compaction continuity**, opt-in through [`compaction_plugins.yaml`](compaction_plugins.yaml): the staged
  OpenCode plugins put each compaction summary before the retained messages and keep the task prompt verbatim
  across compactions, and `vllm_model` bounds the summary call's output and retries a summary it cannot use
  (`compaction_summary_max_tokens`, `compaction_summary_max_attempts`). See
  [`benchmarks/agentic_vbench/opencode_plugins/`](../../../benchmarks/agentic_vbench/opencode_plugins/README.md).

**Delegation**, opt-in through [`subagents.yaml`](subagents.yaml): the primary agent's `task` tool offers one
subagent, `analyst`, the report-only media analyst of the Agentic-VBench subagent SFT traces (10 steps, no further
delegation, must not write the judged output); the built-in `general` and `explore` are hidden. The analyst runs on
the policy with the run's sampling, and each analyst session's model calls are captured with the rollout's and
reported as a child invocation of the primary agent's. Which calls of a child session train, and how its context is
reconstructed, is NeMo RL's decision; Gym's local lineage treats a child's first call as a new root. Pass the
overlay after `harbor_train.yaml` (and after `compaction_plugins.yaml` when both are on);
`agentic_vbench_subagent_steps` changes the analyst's step cap.

Images with OpenCode baked in: `opencode_agent` installs OpenCode into the sandbox at session start unless
`sandbox_executable_path` (with `sandbox_executable_sha256`) names a binary the image already holds; a task image
built with the runtime at that path skips the download (the harness README, "Preinstalled runtime").

## Validation status

Unit tests cover the prepare script's task-root path, the staged plugins (node tests), the model server's summary
recovery, the verifier-only multiplier and the subagent overlay (a session configuration the harness accepts).

`gym dataset validate agentic_vbench_train --sandbox docker` with `harbor_train.yaml` (judge protocol
`nvinference-hybrid`) on a root of eight generated tasks, two per family, from the internal curriculum: 8/8 scored,
0 masked; assembly, repair and sequencing at 1.0, the two repurpose reference edits at 0.956 and 0.953 (the
rubric judge scores the generated reference edit itself below 1; `gym dataset validate` exits non-zero for them,
as it does for any reference solution below 1). Verification ran separately for the repurpose tasks and in the
agent's sandbox for the rest, with the task images prebuilt on a rootless Podman backend and the generated tasks'
shared base image present on it.

A training-shaped `gym eval run` of the same eight tasks with `harbor_train.yaml` against a standalone vLLM 0.26 server
(the curriculum run's start checkpoint, TP4, `--return-tokens-as-token-ids`, `return_token_id_information: true`):
every OpenCode model call was captured with its prompt and generation token ids (18 to 100 calls per episode), and
every row was masked, because Gym's local `fingerprint` lineage could not chain the calls (`parent_digest_mismatch`
on most of them, no terminal attribution): the next prompt OpenCode sends is the chat template's rendering of the
parsed tool calls and reasoning, not the tokens the model generated, so no prompt starts with its parent's tokens.
Keeping history reasoning in the template (`chat_template_kwargs.truncate_history_thinking=false`) did not change
that. This is the case Gym's capture docs route to framework-owned reconstruction (`token_id_capture.parent_resolution:
source_history` with the training framework's external staging), which NeMo RL uses; the standalone run therefore
shows the records, not a training batch.

A NeMo RL smoke (main 0.8.0, `run_grpo_single_controller.py`, five nodes: one vLLM generation node at TP8/EP8 and
four Megatron training nodes at TP8/EP16/CP2, Gym from this branch, the eight generated tasks with two generations
each, token capture with `parent_resolution: source_history` and vLLM-worker external staging as NeMo RL configures
it) trained one GRPO step on this entry: the three Gym servers started under NeMo RL, sixteen episodes ran through the
harbor server, `opencode_agent` and the Podman sandboxes (16 to 49 model calls each, 33 calls per rollout on average,
27k to 65k cumulative tokens), the ledger chained every served call to its parent (no poisoned rollout, invalid-row
rate 0), the environment scored all sixteen (reward mean 0.318, four at 1.0, median 0), and the step ran on 13 valid
sequences and 225k tokens (loss -0.131, grad norm 0.18, token multiplicative probability error 1.06); the three
sequences left out had filled the 65,536-token window (`finish_reason: length`, masked by NeMo RL's overlong filter).
Two settings made that possible, both in the run's config rather than this entry: OpenCode was told the engine's
real window (`agentic_vbench_context_tokens: 65536`) and automatic compaction was off (`compaction.auto: false`). With
the entry's defaults (49,152 tokens, compaction on), a first attempt had every chain break where OpenCode compacted
(41k to 46k cumulative tokens; every call from the compaction summary on was `unresolved_parent`, 11 of 16 rollouts
rejected) and the step then failed in NeMo RL's loss diagnostics on an empty valid mask. Reconstruction across OpenCode
compaction is the training framework's (Gym's capture docs route it there), and NeMo RL main does not have it yet; a
run with compaction on needs that first. One more NeMo RL main gap met on the way: a harbor row's `task_input` is a
single-step `turns` list, which main's `_build_inputs` rejects (the smoke read `turns[0].responses_create_params`
through a local patch).

Subagents (`subagents.yaml`), `gym eval run` on one generated task per family with Gemini 4 through
`inference_provider` and token capture off: the sequencing episode ran three OpenCode sessions, the primary agent made
two `task` calls (127 s and 139 s) that each spawned an analyst session (27 and 25 messages, both ending with status
`failed` in OpenCode's export, not inspected further), reward 0; the repair episode ran two sessions, three `task` calls
(300 s, 255 s, 197 s) to one analyst session (52 messages, `completed`, the follow-ups through the task id it returned),
reward 0.819. Every child session was reported as a child invocation of the primary's (`parent_invocation_id`), with one
`subagent_spawn_ambiguous` gap where the three calls share the child. The assembly and repurpose episodes were dropped by
the same `OpenCode exited with code 1` as the benchmark entry's full-set run on that route.
