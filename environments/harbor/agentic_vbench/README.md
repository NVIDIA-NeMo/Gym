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
extracts. The key is required to start the server with that overlay (below), so Repurpose can only be scored with
access to the NVIDIA Inference Hub. The other 64 tasks need no credentials: prepare them alone and use the overlay
without the judge secret, which keeps separate verification, the episode cap and `image_input`:

```sh
AGENTIC_VBENCH_HARBOR_FAMILIES=repair,assembly,sequencing python -m benchmarks.agentic_vbench.prepare_harbor_env
gym eval run agentic_vbench --agent opencode_agent --sandbox docker \
  --config environments/harbor/agentic_vbench/harbor_sandbox_no_judge.yaml ...
```

## What the prepare script changes

Each task folder is the upstream folder with three adjustments; the adjustments and the digests of the files they
touch are recorded under `.records/<task>/` in this folder (`judge_protocol.json`, `assembly_judge_guard.json`,
`materials.json`). A task none of them applies to is byte for byte the upstream folder.

| Tasks | Change | Why |
| --- | --- | --- |
| Repurpose (judged by an LLM rubric) | `[environment.env]` loses the upstream `ANTHROPIC_API_KEY`/`GEMINI_API_KEY` templates; `[steps.verifier.env]` gains `NVINFERENCE_API_KEY = "${NVINFERENCE_API_KEY}"` and the `AVB_JUDGE_*` route settings of the judge protocol; `tests/judge.py` becomes `hub_judge.py` (the original is kept as `avb-original-judge.py`); the step declares `artifacts = ["/workspace/output"]` | The judge runs over the NVIDIA Inference Hub with one key that reaches only the verifier. `AGENTIC_VBENCH_JUDGE_PROTOCOL` selects `nvinference-upstream` (default: the upstream judge pair, Claude Opus 4.7 and Gemini 3.1 Pro) or `nvinference-hybrid` (Claude Opus 4.8 and Gemini 3.5 Flash). Before its first Hub call, `hub_judge.py` sends one request per route in the judge's own request shapes; a route the key cannot use, a missing key, a Hub transport failure that persists through the retries, or its allowance of time spent waiting on the Hub (1200 s, below the verifier budget) running out make it exit 75 without a reward, so the episode is masked instead of scoring 0. What the agent's output decides never does: a request the route refuses for its size or length is that rubric item's error, as upstream, and the output clip is inlined as an ffmpeg proxy (at most 720p, audio kept) when it is over the 14 MiB inline budget, recorded in the result's provenance. |
| Assembly | `tests/judge.py` rejects a non-object `solution.json` with reward 0 instead of crashing | The upstream judge calls `.get` on whatever JSON the model wrote; a bare list raised instead of scoring 0 like every other malformed manifest. Only the exact pinned judges (verified by digest) are patched. |
| Repair | the verifier's golden-asset download retries longer within a bound (`--retry 12 --retry-delay 20 --retry-all-errors --max-time 300 --retry-max-time 1200`), a download that still fails exits 75, and the step declares `artifacts = ["/workspace/output", "/workspace/materials"]` (the verifier reads the materials the setup step copied into the agent's sandbox) | Hugging Face answers bursts from one egress address with 429/500 and the upstream six attempts finish in about 20 s; without the guard the verifier would write no reward and the rollout would score 0, and without the bound a stalled download would run into the verifier timeout, which is a measured 0. |

`[steps.verifier.env]` templates are filled from the server's `secrets` (the overlay maps `NVINFERENCE_API_KEY`
from the environment), so the key never appears in a task file or a row. The server masks a rollout whose
verifier exited 75 (`EX_TEMPFAIL`) and wrote no reward (`harbor:verifier_unavailable`); a verifier timeout still
wins, and any other exit without a reward is a measured 0. See
[Task-folder verifiers](../../../fern/versions/latest/pages/data/task-folders/task-folder-verifiers.mdx).

Because the exit code comes from `test.sh`, the server honors it only where the agent cannot have produced it: the
overlay's `separate_verifier_for_artifacts: true` grades every task that declares `artifacts`, i.e. the judged and
repair tasks, in a fresh sandbox of the task image that receives `/workspace/output` (and, for repair,
`/workspace/materials`) from the agent's sandbox; a 75 from a shared-mode verifier is ignored and measured. The
agent's sandbox therefore never runs the process that holds the judge key either. Every task's `/workspace/materials`
is filled by the step's `setup.sh` from the image's `/baked` folder in the agent's sandbox, which is why the repair
verifiers, which read it, receive it as an artifact. Assembly and sequencing verify in the agent's sandbox as
upstream does.

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

The reward is the task's own `reward.json`, 0..1. The leaderboard score is
`(mean_repair + mean_assembly + mean_sequencing + mean_repurpose) / 4`, using the unscaled family means;
equivalently, over the complete 100-task set, average `reward_i x 25 / size_of_family_i` across the 100 task rows
(the family is the task name's prefix). The environment server reports the task-weighted mean, so compute the
leaderboard score from the rows. A subset is not the benchmark score, and a masked row is not a 0.

The entry's checks so far: the prepare script writes all 100 tasks, every `task.toml` loads with Gym's Harbor
schema, the unit tests cover the adjustments, the Hub judge's failure handling against a local socket server and the
server's handling of exit 75, and `gym dataset validate agentic_vbench` (the oracle run of every `solution/`, judge
protocol `nvinference-hybrid`) scored Assembly 18/18, Sequencing 28/28 and Repair 18/18 at 1.0 and Repurpose 36/36
at 0.202 with no masked rows: 11 Repurpose verifiers hit the 1800 s budget in the judge's own transcription of long
clips (the published runs gave the verifier 3x), and 8 scored 0 on rubric penalties of the upstream reference edits.
Recorded on the branch head: the injected-outage smoke (`gym dataset validate --sandbox docker` with the overlay on
a copy of the entry whose verifiers point at hosts that do not resolve: a Repurpose task with a bad Hub route and a
Repair task with a blocked materials download both came back masked as `harbor:verifier_unavailable`, with the reason
in the row; the dataset.toml review thread has the details), and a full-set `opencode_agent` run with `image_input` on
against `gcp/google/eccn-gemini-4-argon` (Gemini 4 on the NVIDIA Inference Hub through `inference_provider`), the task
images prebuilt on a rootless Podman backend, 12 concurrent episodes, 3 h 55 min: Assembly 17 scored, mean 0.875;
Repair 16, 0.401; Sequencing 27, 0.647; Repurpose 34, 0.437; no masked row (every verifier returned; Repurpose
verified in its separate sandbox in 154 s median, 301 s max, against the 1800 s budget); equal-family mean of the
scored rows 0.590, which is not the benchmark score because six rows (one Assembly, two Repair, one Sequencing, two
Repurpose) were dropped: the agent server answered 502 `OpenCode exited with code 1` after 8 to 110 tool calls, with
an empty last assistant message and no stderr in the error, and Gym kept the rows out of the score rather than scoring
them 0. The run kept no server log (the harness's `export_dir` was unset), so the cause is open; set `export_dir` to
keep OpenCode's event log for such rows. Treat these numbers as the harness's first baseline, not a baselined
result. `opencode_agent` saves text-only trajectories, so a run with `image_input` evaluates normally but its
transcripts carry no frames.
