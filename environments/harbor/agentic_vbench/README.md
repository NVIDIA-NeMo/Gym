# Agentic-VBench on the Harbor task shape

The 100 pinned [Agentic-VBench](https://github.com/PhiloLabs/agentic-vbench) video-editing tasks (Repair 18,
Assembly 18, Sequencing 28, Repurpose 36) at revision `410a75fe3dae37de4344fc9e5317da089505330a` (Apache-2.0) as
plain Harbor task folders for the generic [`harbor` environment server](../../../environment_servers/harbor/README.md).
Gym seeds each task's own image as a sandbox, a sandboxed harness such as `opencode_agent` works on the clips inside
it, and the task's own `tests/test.sh` writes the reward, so any session harness the server can run can take the
benchmark. The entry is `type = "benchmark"`: an evaluation dataset, not a training split.

```sh
python -m benchmarks.agentic_vbench.prepare_harbor_env        # writes the 100 task folders into this folder
gym dataset validate agentic_vbench --sandbox docker \
  --config environments/harbor/agentic_vbench/harbor_sandbox.yaml   # plays each task's solution/ through its verifier
gym eval run agentic_vbench --agent opencode_agent --sandbox docker \
  --config environments/harbor/agentic_vbench/harbor_sandbox.yaml \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

`dataset.toml` declares no source: the tasks come from the pinned upstream git checkout, not from a Harbor hub, so
the prepare script writes them into this folder (the task folders, `manifest.toml` and the preparation records under
`.records/` are gitignored beside this README), and both commands resolve the entry name to it. The prepare script
fetches the pinned checkout into the Gym cache root, or reads an existing one from `AGENTIC_VBENCH_ROOT`. The
helpers it uses live in [`benchmarks/agentic_vbench/`](../../../benchmarks/agentic_vbench/README.md).

Both commands take the deployment overlay: it carries the judge key, the episode cap, separate verification of the
judged and repair tasks and the harness's `image_input`, which the tasks need for the model to see the frames it
extracts. The key is required to start the server (below), so Repurpose can only be scored with access to the NVIDIA
Inference Hub; the other 64 tasks need no credentials.

## What the prepare script changes

Each task folder is the upstream folder with three adjustments; the adjustments and the digests of the files they
touch are recorded under `.records/<task>/` in this folder (`judge_protocol.json`, `assembly_judge_guard.json`,
`materials.json`). A task none of them applies to is byte for byte the upstream folder.

| Tasks | Change | Why |
| --- | --- | --- |
| Repurpose (judged by an LLM rubric) | `[environment.env]` loses the upstream `ANTHROPIC_API_KEY`/`GEMINI_API_KEY` templates; `[steps.verifier.env]` gains `NVINFERENCE_API_KEY = "${NVINFERENCE_API_KEY}"` and the `AVB_JUDGE_*` route settings of the judge protocol; `tests/judge.py` becomes `hub_judge.py` (the original is kept as `avb-original-judge.py`); the step declares `artifacts = ["/workspace/output"]` | The judge runs over the NVIDIA Inference Hub with one key that reaches only the verifier. `AGENTIC_VBENCH_JUDGE_PROTOCOL` selects `nvinference-upstream` (default: the upstream judge pair, Claude Opus 4.7 and Gemini 3.1 Pro) or `nvinference-hybrid` (Claude Opus 4.8 and Gemini 3.5 Flash). Before judging, `hub_judge.py` sends one request per route; a route the key cannot use, a missing key, a Hub transport failure or size rejection that persists, or its own deadline (1500 s, below the verifier budget) make it exit 75 without a reward, so the episode is masked instead of scoring 0. |
| Assembly | `tests/judge.py` rejects a non-object `solution.json` with reward 0 instead of crashing | The upstream judge calls `.get` on whatever JSON the model wrote; a bare list raised instead of scoring 0 like every other malformed manifest. Only the exact pinned judges (verified by digest) are patched. |
| Repair | the verifier's golden-asset download retries longer within a bound (`--retry 12 --retry-delay 20 --retry-all-errors --max-time 300 --retry-max-time 1200`), a download that still fails exits 75, and the step declares `artifacts = ["/workspace/output"]` | Hugging Face answers bursts from one egress address with 429/500 and the upstream six attempts finish in about 20 s; without the guard the verifier would write no reward and the rollout would score 0, and without the bound a stalled download would run into the verifier timeout, which is a measured 0. |

`[steps.verifier.env]` templates are filled from the server's `secrets` (the overlay maps `NVINFERENCE_API_KEY`
from the environment), so the key never appears in a task file or a row. The server masks a rollout whose
verifier exited 75 (`EX_TEMPFAIL`) and wrote no reward (`harbor:verifier_unavailable`); a verifier timeout still
wins, and any other exit without a reward is a measured 0. See
[Task-folder verifiers](../../../fern/versions/latest/pages/data/task-folders/task-folder-verifiers.mdx).

Because the exit code comes from `test.sh`, it is only trustworthy where the agent cannot run `test.sh`: the overlay's
`separate_verifier_for_artifacts: true` grades every task that declares `artifacts`, i.e. the judged and repair
tasks, in a fresh sandbox of the task image that receives `/workspace/output` from the agent's sandbox. The agent's
sandbox therefore never runs the process that holds the judge key either. Assembly and sequencing read
`/workspace/materials`, which the image provides, and verify in the agent's sandbox as upstream does.

## Deployment

[`harbor_sandbox.yaml`](harbor_sandbox.yaml) is the deployment overlay: it passes the judge key as a server secret
(no empty default: without a key the value is null, which the server's `secrets` table rejects, so the server does
not start), turns on separate verification for tasks with artifacts and the harness's `image_input`, and bounds
concurrent episodes, since every episode starts one sandbox from a task image of a few gigabytes. Pass it with
`--config`. The task Dockerfiles build their images (they download the task's clips from Hugging Face at build
time), so `gym eval run` and `gym dataset validate` build them once at prepare through the top-level `builds:` block
(`tool`, `repository`); with `--sandbox docker` and no repository, `docker` builds them locally.

Each task declares 1800 s for the agent and 1800 s for the verifier. The benchmark's published runs gave the
verifiers a 3x allowance; `[gym.defaults] timeout_multiplier` scales the agent budget too, so a comparable run is not
expressible yet, and runs through this entry use the task budgets as published.

## Scoring and validation status

The reward is the task's own `reward.json`, 0..1. The leaderboard's equal-family mean is `reward x 25 / family size`
averaged over the four families; the environment server reports the task-weighted mean, so compute the equal-family
mean from the rows by task family (the family is the task name's prefix).

The entry's checks so far: the prepare script writes all 100 tasks, every `task.toml` loads with Gym's Harbor
schema, and the unit tests cover the three adjustments, the Hub judge's failure handling against a local socket
server, and the server's handling of exit 75. Not recorded yet, and owed before review continues:
`gym dataset validate agentic_vbench` (the oracle run of every `solution/`, rewards per family), a full-set
`opencode_agent` run with `image_input` on (scored and masked counts per family by `failure_kind`, Repurpose
`verifier_seconds` against the 1800 s budget, the largest `repurpose.mp4` the Gemini judge inlined, transcripts and
verifier logs read by hand), and one injected-outage smoke (a bad Hub route on a Repurpose task, a blocked download
on a Repair task) showing `harbor:verifier_unavailable` in `rollouts.jsonl`. Until then, treat `allowed_agents` as
the harness under validation, not as a baselined one. `opencode_agent` saves text-only trajectories, so a run with
`image_input` evaluates normally but its transcripts carry no frames.
