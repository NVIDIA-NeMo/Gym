# Agentic-VBench on the Harbor task shape

The 100 pinned [Agentic-VBench](https://github.com/PhiloLabs/agentic-vbench) video-editing tasks (Repair 18,
Assembly 18, Sequencing 28, Repurpose 36) at revision `410a75fe3dae37de4344fc9e5317da089505330a` as plain Harbor
task folders for the generic [`harbor` environment server](../../../environment_servers/harbor/README.md). Gym seeds
each task's own image as a sandbox, a sandboxed harness such as `opencode_agent` works on the clips inside it, and
the task's own `tests/test.sh` writes the reward, so any session harness can run the benchmark and NeMo RL can train
through the Environment Server with any of them.

```sh
python -m benchmarks.agentic_vbench.prepare_harbor_env        # writes the 100 task folders into this folder
gym dataset validate agentic_vbench --sandbox docker          # plays each task's solution/ through its verifier
gym eval run agentic_vbench --agent opencode_agent --sandbox docker \
  --config environments/harbor/agentic_vbench/harbor_sandbox.yaml \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

`dataset.toml` declares no source: the tasks come from the pinned upstream git checkout, not from a Harbor hub, so
the prepare script writes them into this folder (the task folders, `manifest.toml` and the preparation records under
`.records/` are gitignored beside this README), and both commands resolve the entry name to it. The prepare script
fetches the pinned checkout into the Gym cache root, or reads an existing one from `AGENTIC_VBENCH_ROOT`.

## What the prepare script changes

Each task folder is the upstream folder with the three adjustments the `agentic_vbench_agent` applies to a trial,
so the two paths grade the same way. The adjustments and the digests of the files they touch are recorded under
`.records/<task>/` in this folder.

| Tasks | Change | Why |
| --- | --- | --- |
| Repurpose (judged by an LLM rubric) | `[environment.env]` loses the native `ANTHROPIC_API_KEY`/`GEMINI_API_KEY` templates; `[steps.verifier.env]` gains `NVINFERENCE_API_KEY = "${NVINFERENCE_API_KEY}"` and the `AVB_JUDGE_*` route settings of the judge protocol; `tests/judge.py` becomes the agent's `hub_judge.py` (the original is kept as `avb-original-judge.py`) | The judge runs over the NVIDIA Inference Hub with one key, and that key reaches only the verifier process, never the agent's sandbox. `AGENTIC_VBENCH_JUDGE_PROTOCOL` selects `nvinference-hybrid` (default) or `nvinference-native`. |
| Assembly | `tests/judge.py` rejects a non-object `solution.json` with reward 0 instead of crashing | The upstream judge calls `.get` on whatever JSON the model wrote; a bare list raised instead of scoring 0 like every other malformed manifest. Only the exact pinned judges are patched. |
| Repair | the verifier's golden-asset download retries longer (`--retry 12 --retry-delay 20 --retry-all-errors`) | Hugging Face answers bursts from one egress address with 429/500 and the upstream six attempts finish in about 20 s; the verifier then writes no reward. |

`[steps.verifier.env]` templates are filled from the server's `secrets` (the overlay below maps
`NVINFERENCE_API_KEY` from the environment), so the key never appears in a task file or a row.

## Deployment

[`harbor_sandbox.yaml`](harbor_sandbox.yaml) is the deployment overlay: it passes the judge key as a server secret
and bounds concurrent episodes, since every episode starts one sandbox from a task image of a few gigabytes. Pass it
with `--config`. The task Dockerfiles build their images (they download the task's clips from Hugging Face at build
time), so `gym eval run` and `gym dataset validate` build them once at prepare through the top-level `builds:` block
(`tool`, `repository`); with `--sandbox docker` and no repository, `docker` builds them locally.

Each task declares 1800 s for the agent and 1800 s for the verifier; `[gym.defaults] timeout_multiplier` scales both.
The `agentic_vbench_agent` path runs the verifiers with a 3x allowance (`verifier_timeout_multiplier`), so a
comparable run here sets `++...timeout_multiplier` or accepts the task budgets as published.

## What differs from the `agentic_vbench_agent` path

[`benchmarks/agentic_vbench/`](../../../benchmarks/agentic_vbench/README.md) runs the same tasks through Harbor's own
runtime (Harbor 0.6.6 with OpenCode 1.14.39 inside the task container) and is the published baseline; it also
carries the NeMo-RL training variant with token capture, OpenCode compaction and subagent policies. This entry is the
harness-swappable path: the harness is a Gym agent session, OpenCode is the version `opencode_agent` installs (or
the one a `sandbox_executable_path` pin names), and the reward is the task's own `reward.json` read by the
environment server. Expect scores within noise of the baseline for the same harness version, not identical ones.
The leaderboard's equal-family mean (`reward x 25 / family size`, averaged over the four families) is an aggregate
over this entry's rows by task family (the family is the task name's prefix); the environment server reports the
task-weighted mean.
