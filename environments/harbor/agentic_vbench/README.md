# Agentic-VBench on the Harbor task shape

The 100 pinned [Agentic-VBench](https://github.com/PhiloLabs/agentic-vbench) video-editing tasks (Repair 18,
Assembly 18, Sequencing 28, Repurpose 36) at revision `410a75fe3dae37de4344fc9e5317da089505330a` (Apache-2.0) as
plain Harbor task folders for the generic [`harbor` environment server](../../../environment_servers/harbor/README.md).
Gym seeds each task's own image as a sandbox, a sandboxed harness such as `opencode_agent` works on the clips inside
it, and the task's own `tests/test.sh` writes the reward, so any session harness can run the benchmark and NeMo RL
can train through the Environment Server with any of them.

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
fetches the pinned checkout into the Gym cache root, or reads an existing one from `AGENTIC_VBENCH_ROOT`. The
helpers it uses live in [`benchmarks/agentic_vbench/`](../../../benchmarks/agentic_vbench/README.md).

## What the prepare script changes

Each task folder is the upstream folder with three adjustments; the adjustments and the digests of the files they
touch are recorded under `.records/<task>/` in this folder. A task none of them applies to is byte for byte the
upstream folder.

| Tasks | Change | Why |
| --- | --- | --- |
| Repurpose (judged by an LLM rubric) | `[environment.env]` loses the native `ANTHROPIC_API_KEY`/`GEMINI_API_KEY` templates; `[steps.verifier.env]` gains `NVINFERENCE_API_KEY = "${NVINFERENCE_API_KEY}"` and the `AVB_JUDGE_*` route settings of the judge protocol; `tests/judge.py` becomes `hub_judge.py` (the original is kept as `avb-original-judge.py`) | The judge runs over the NVIDIA Inference Hub with one key, and that key reaches only the verifier process, never the agent's sandbox. `AGENTIC_VBENCH_JUDGE_PROTOCOL` selects `nvinference-native` (default: the upstream judge pair, Claude Opus 4.7 and Gemini 3.1 Pro) or `nvinference-hybrid` (Claude Opus 4.8 and Gemini 3.5 Flash). A Hub outage, a route the key cannot reach or a missing key writes `inconclusive.json`, so the episode is masked instead of scoring 0. |
| Assembly | `tests/judge.py` rejects a non-object `solution.json` with reward 0 instead of crashing | The upstream judge calls `.get` on whatever JSON the model wrote; a bare list raised instead of scoring 0 like every other malformed manifest. Only the exact pinned judges (verified by digest) are patched. |
| Repair | the verifier's golden-asset download retries longer (`--retry 12 --retry-delay 20 --retry-all-errors`), and a download that still fails writes `inconclusive.json` | Hugging Face answers bursts from one egress address with 429/500 and the upstream six attempts finish in about 20 s; without the note the verifier would write no reward and the rollout would score 0. |

`[steps.verifier.env]` templates are filled from the server's `secrets` (the overlay below maps
`NVINFERENCE_API_KEY` from the environment), so the key never appears in a task file or a row. The server masks a
rollout whose verifier wrote `inconclusive.json` and no reward (`harbor:verifier_inconclusive`); see
[Task-folder verifiers](../../../fern/versions/latest/pages/data/task-folders/task-folder-verifiers.mdx).

## Deployment

[`harbor_sandbox.yaml`](harbor_sandbox.yaml) is the deployment overlay: it passes the judge key as a server secret
(no empty default: without a key the server refuses to seed a judged task) and bounds concurrent episodes, since
every episode starts one sandbox from a task image of a few gigabytes. Pass it with `--config`. The task
Dockerfiles build their images (they download the task's clips from Hugging Face at build time), so `gym eval run`
and `gym dataset validate` build them once at prepare through the top-level `builds:` block (`tool`, `repository`);
with `--sandbox docker` and no repository, `docker` builds them locally.

Each task declares 1800 s for the agent and 1800 s for the verifier; `[gym.defaults] timeout_multiplier` scales both.
The benchmark's published runs gave the verifiers a 3x allowance; a comparable run here sets `++...timeout_multiplier`
or accepts the task budgets as published.

## Scoring and validation status

The reward is the task's own `reward.json`, 0..1. The leaderboard's equal-family mean is `reward x 25 / family size`
averaged over the four families; the environment server reports the task-weighted mean, so compute the equal-family
mean from the rows by task family (the family is the task name's prefix).

The entry's checks so far: the prepare script writes all 100 tasks, every `task.toml` loads with Gym's Harbor
schema, and the unit tests cover the three adjustments. `gym dataset validate agentic_vbench` (the oracle run of
every `solution/`) and a full-set `opencode_agent` run with per-family scored and masked counts have not been
recorded yet; until they are, treat `allowed_agents` as the harness under validation, not as a baselined one.
`opencode_agent` must run with `image_input` for the model to see extracted frames, and such transcripts do not
export to text-only training rows (see the agent's README); evaluation is unaffected.
