# Terminal-Bench-Science v0.1

[Terminal-Bench-Science](https://github.com/harbor-framework/terminal-bench-science)
(Apache-2.0) has 70 research-workflow tasks across five domains. There are 19 life-sciences
tasks, 17 physical-sciences, 8 earth-sciences, 17 mathematical-sciences and 9
engineering-sciences.

Every task is a Harbor package. Its agent works in a container with public network access
for up to 8 hours. A separate verifier container with no network then grades the declared
artifacts.

`terminal_bench_science/miniswe` runs the tasks on the generic
[`environment_servers/harbor`](../../environment_servers/harbor/README.md) environment server,
through the committed entry
[`environments/harbor/terminal_bench_science`](../../environments/harbor/terminal_bench_science/README.md),
as Terminal-Bench 4's mini-SWE preset does. The Codex presets and the deprecated `/run` preset
(`miniswe_legacy_agent`) reuse the pinned Harbor-package runtime in
[`resources_servers/terminal_bench_4`](../../resources_servers/terminal_bench_4/README.md); only
the dataset pin, the task images and the run directories differ from Terminal-Bench 4. The
sections from [Dataset pin](#dataset-pin) through [Compose tasks](#compose-tasks) describe that
runtime; [Profile](#profile) says which of its settings the Harbor preset carries.

## Measurement contract

`terminal_bench_science/miniswe` protocol `0.2.0` selects the 70 pinned tasks through the Harbor
environment server, with incomplete coverage. Preparation skips `inverse-lithography`,
`noisy-blackbox-optimization` and `protein-active-learning` because their Compose settings are
unsupported on this route (see the entry's [Known limits](../../environments/harbor/terminal_bench_science/README.md#known-limits)).
`ode-law-discovery` and `geometric-pharmacophore-alignment` fail at agent seed because their
working directory is `/`. These skips and failures have no reward and are excluded from the
default `mean/reward`: at most 65 tasks can return rewards, even if every other task succeeds
in setup. This is an upper bound, not validated coverage; the two remaining Compose tasks
are untested. The unadjusted mean is not a full 70-task leaderboard score. For full task
coverage, use `terminal_bench_science/miniswe_legacy_agent` protocol `0.1.0`. Harbor-path and
relay scores are separate measurements; the version change records the new adapter and its
coverage limit.

A run is comparable with the [public leaderboard](https://www.terminal-bench-science.ai/) when
it keeps these fixed:

- **Provenance.** The Harbor registry dataset `terminal-bench-science/terminal-bench-science@v0.1`
  (Git tag `v0.1.0`), content hash
  `sha256:91531bf50016a7c64f6cc60794a17c64c6b2c14858a8ae0de39ca16f2abd611a`. `manifest.json`
  pins all 70 task packages by digest, and the resources server checks each package against its
  digest before it allocates a sandbox ([Dataset pin](#dataset-pin)). The Harbor entry pulls the
  same dataset digest and checks each task folder against the content hash its pull recorded.
- **Task contract.** Gym reads each pinned `task.toml` and `instruction.md` unchanged: the agent
  image built from `environment/`, its CPUs, memory, storage, time budget and declared
  artifacts, and a separate verifier image built from `tests/` with its own timeout.
- **Run conditions.** The agent works with public network access for up to 8 hours (28800
  seconds on every task). The verifier then runs in its own sandbox with all egress denied.
- **Scoring.** A trial passes only when the verifier's official reward is 1. The leaderboard
  runs three trials per task, 210 in all (`++num_repeats=3`), and includes every trial in its
  denominator. Gym's default `mean/reward` includes only scored trials; count excluded
  failures and skipped tasks as zero when comparing with the leaderboard, including the five
  tasks above. The resources-server presets report partial credit separately; it never replaces the reward
  ([Scoring](#scoring)).
- **Deviations.** A few deployment settings differ from the packages. Each is listed with its
  reason and its override in [Deviations from the packages](#deviations-from-the-packages).

## Dataset pin

`manifest.json` pins the Harbor registry dataset
`terminal-bench-science/terminal-bench-science@v0.1`. The registry tags `0.1.0` and `latest`
point to the same version, revision 10, with content hash
`sha256:91531bf50016a7c64f6cc60794a17c64c6b2c14858a8ae0de39ca16f2abd611a`.

The manifest also records the 70 task package digests. They match the
`tasks/dataset.toml` of Git tag `v0.1.0` (commit `f81afac4`), and the resources server's
content hash reproduces each of them from that tag.

The `/run` profile's `prepare.py` writes identity rows only. The Codex profiles use
`prepare_codex.py`, which also embeds each package's official instruction as the row's prompt.
The resources server downloads each package and verifies its digest before it allocates a
sandbox. The mini-SWE preset needs neither script: `gym dataset pull terminal_bench_science`
fetches the task folders, and the Harbor environment server reads each instruction from its folder.

## Images

Upstream publishes no images. Each task ships `environment/Dockerfile` for the agent and
`tests/Dockerfile` for the verifier, and the verifier image bakes in `/tests`. You build
both images once per task, using that directory as the build context, and push them to a
registry your sandbox deployment can pull from. The five [Compose tasks](#compose-tasks) also
need their sidecar images.

Then pass them to the resources server as a `task_images` map, as templates, or both:

```sh
'++tbsci_agent_image="REGISTRY/tbsci/{task}:environment-{digest}"' \
'++tbsci_verifier_image="REGISTRY/tbsci/{task}:tests-{digest}"' \
'++tbsci_task_images={mendota-ice-phenology: {agent: "REGISTRY/x@sha256:...", verifier: "REGISTRY/y@sha256:..."}}'
```

Keep the inner double quotes around templates; without them, Hydra's override grammar
rejects the braces.

Templates accept these placeholders:

- `{task}`: the short task name.
- `{org}`: `terminal-bench-science`.
- `{digest}`: the task package's SHA256, without the `sha256:` prefix.

Using `{digest}` in tags ties each image to the pinned package content. An entry in
`tbsci_task_images` takes precedence over the templates. A task with no resolvable image
fails at seed time, before any sandbox is allocated, and the error names the missing role.

### Verifier worker UIDs

Eleven verifiers drop their worker processes to a fixed low UID under a small `RLIMIT_NPROC`:
`baseline-free-localization`, `genomic-model-ranking`, `guided-wave-localization`,
`linked-cell-suppression`, `navigation-sensor-calibration`, `noisy-blackbox-optimization`,
`protein-active-learning`, `rdkit-ic-constraints`, `small-area-equivalence`,
`tess-transit-vetting` and `virtual-baseline-localization`. Linux counts that limit per UID
across the whole node, including the processes of other sandboxes that use the same UID (65532
is distroless's `nonroot`, 65534 is `nobody`). On a shared node, worker launches then fail with
`EAGAIN` ("Resource temporarily unavailable") and correct work is graded 0.

Build these verifier images with a worker UID that nothing else on the node uses, such as a
high value unique to the task. The UID is set in the task's `tests/` files (the Dockerfile's
`useradd` or a constant in the verifier's runtime); change only the UID and keep the limits and
the grading. `small-area-equivalence`'s `tests/data/suite_manifest.json` pins the SHA-256 of
`test_state.py`, so update that pin with it. The UID is per task, not per trial: concurrent
verifiers of one task on one node still share its limit. Each verifier peaks well below it (the
tightest, `baseline-free-localization`, at 1 of 4), so the three trials of a task fit on one
node; many more concurrent trials of one task may not.

Nothing in Gym checks the UID. The images are built outside this repository and CI never
runs them, and a reference-solution run with the shared UID can still pass on a quiet node.
Check the UID when you build each image.

### Check images with the reference solutions

Before you spend model time, run each task's `solution/solve.sh` through the same seed and
verify lifecycle. The runner works like Harbor's oracle agent and needs no model:

```sh
PYTHONPATH=. python benchmarks/terminal_bench_science/oracle.py \
  --output results/tbsci-oracle --category cpu --concurrency 4 \
  --agent-image 'REGISTRY/tbsci/{task}:environment-{digest}' \
  --verifier-image 'REGISTRY/tbsci/{task}:tests-{digest}' \
  --env-file /path/to/private.env   # OPENSANDBOX_* credentials
```

Instead of the templates, `--images map.json` takes the same per-task map as `tbsci_task_images`
(`{"<task>": {"agent": ..., "verifier": ..., "services": {...}}}`).

A task passes when its official reward is 1. The runner writes one line per task to
`oracle.jsonl`, with the reward, `partial_credit`, the test counts, the oracle exit code and
the trial directory.

`--local-docker` runs the same lifecycle on this machine's Docker daemon, using locally built
images. There, the verifier runs with `--network none` instead of OpenSandbox's deny-all
egress policy. Reference solutions can be compute-heavy; `sparse-network-assimilation`, for
example, takes about 17 minutes.

## Execution

The resources server reads each task's budgets and resources from its pinned `task.toml`:

| Setting | Values across the 70 tasks |
|---|---|
| Agent timeout | 28800 seconds on every task |
| Verifier timeout | 120 to 12000 seconds |
| Sandbox start, including the image pull (`build_timeout_sec`) | 300 to 7200 seconds |
| CPUs | 1 to 4 |
| Memory | 2 to 16 GB |
| Storage | 2 to 50 GB |
| GPUs | none |

The longest verifiers are `linked-cell-suppression` (12000 seconds) and `leaky-bloch-meep`
(10800 seconds).

The verifier sandbox denies all egress. Some verifier settings deliberately differ from the
packages; see [Deviations from the packages](#deviations-from-the-packages).

`resources.yaml` keeps the Terminal-Bench 4 deployment settings:

- OpenSandbox credentials, set with `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`, or
  `++tbsci_split_sandbox_endpoints=true` with the split CPU/GPU variables.
- Shared EFS logs, with their fallback.
- TTL renewal.
- A 10-hour seeded-session deadline, which covers harness setup and the 8-hour agent budget.
- An 18-hour per-command ceiling.

Memory limits behave as on Docker, where the leaderboard ran. By default an OpenSandbox sandbox
that exceeds its memory limit is killed as a whole (`memory.oom.group=1`), while Docker kills
only the largest process and the container keeps running. The profile's sandbox metadata
`nemo.nvidia.com/oom-kill: process` asks the cell for the single-process kill, as Terminal-Bench
4's profile does since #4650; the value `pod` restores whole-sandbox handling. Every sandbox a
trial creates carries it: agent, verifier, Compose services and the EFS log helper.

### Deviations from the packages

Gym reads every pinned `task.toml` unchanged. The settings below differ from what a package
declares, or from what Harbor does with it on Docker. Each is set in `resources.yaml` and
reported: an applied verifier timeout is recorded in the trial's diagnostics as
`verifier_timeout_override`.

| Setting | Tasks | Package or Harbor | This profile | Why | Override |
|---|---|---|---|---|---|
| Verifier CPUs and memory | The 58 tasks whose `[verifier.environment]` declares neither | Unset, which is unlimited on Harbor's Docker backend | The agent environment's CPUs and memory | OpenSandbox would apply its default of 1 CPU and 2 GiB. With it, the `highdim-mediation-debiasing` reference solution loses its verifier sandbox mid-run. | `++tbsci_verifier_resources_from_agent=false` |
| Verifier timeout | `finite-free-stam`, `gen-turan-paths` | 600 seconds | 1800 seconds | These verifiers rebuild the submission against a prebuilt Mathlib of about 6 GB of `.olean` files. On sandbox images that are pulled lazily, reading them alone can exceed 600 seconds. | `++tbsci_verifier_timeout_overrides={}` |
| Verifier timeout | `certified-sparse-regression` | 3600 seconds | 7200 seconds | The verifier is CPU-bound. On the agent's 4 CPUs, its check of the reference solution took 4002 seconds; Docker verifiers can use every host core. | `++tbsci_verifier_timeout_overrides={}` |
| Compose hardening | `inverse-lithography`, `noisy-blackbox-optimization`, `protein-active-learning` | `init`, `read_only`, `tmpfs`, `pids_limit`, `cap_drop`, `security_opt` and an internal network | Removed | Separate sandboxes cannot enforce them; see [Isolation this profile drops](#isolation-this-profile-drops). | None |
| Sidecar address | `protein-active-learning` | `submission` reaches `http://assay:8080/status` through `/etc/hosts` | `ASSAY_STATUS` holds that URL, rewritten to the `assay` sandbox's address | OpenSandbox cannot edit `/etc/hosts` in a non-root sandbox, so the `submission` image must read the variable; see [`protein-active-learning` submission](#protein-active-learning-submission). | None |
| Disk limit | Agent, verifier and service environments that declare `storage_mb` (2 to 50 GiB) | `storage_mb`, which Harbor's Docker backend does not enforce | `storage_mb` is reserved; the limit is 100 GiB (`sandbox_min_disk_gib`) | OpenSandbox evicts the whole sandbox, with no "no space left" error first, once it writes past its ephemeral-storage limit. | `++tbsci_min_disk_gib=<GiB>` |
| Shared S3 mount | Every task | No such mount on Docker | Dropped (`nemo.nvidia.com/s3-hostpath: 'false'`) | By default the cell mounts a shared read-only S3 bucket at `/mnt/s3-data`. Its data is unrelated to the benchmark; agents searched it for answers, and whole-filesystem scans stalled in it. | None |

### Compose tasks

Five tasks run sidecar services beside the agent container. Their Compose files build each
sidecar from a Dockerfile in `environment/`:

| Task | Sidecars |
|---|---|
| `inverse-lithography` | `fabricator` |
| `longitudinal-clinical-agent` | `broker` |
| `noisy-blackbox-optimization` | `public-evaluator` |
| `protein-active-learning` | `assay`, `submission` |
| `tamp-skill-planning` | `sim` |

Build and push those images too, then name them by service. `protein-active-learning`'s
`submission` needs a one-line change; see
[`protein-active-learning` submission](#protein-active-learning-submission). Use a template
(`{service}` is the Compose service name) or per-task `services`; `oracle.py` takes the template
as `--service-image`:

```sh
'++tbsci_service_image="REGISTRY/tbsci/{task}:{service}-{digest}"' \
'++tbsci_task_images={tamp-skill-planning: {agent: "...", verifier: "...", services: {sim: "REGISTRY/z@sha256:..."}}}'
```

Each service runs in its own sandbox. Gym starts the service's command itself and adds every
service name to each member's `/etc/hosts`. Harbor's `CPUS` and `MEMORY` Compose variables
take the task's resources.

To start a service, Gym needs its image's startup metadata. `compose-images.json` records it
for the 11 images built from these tasks' Dockerfiles, keyed `<task>/<service>` (`main` is
`environment/Dockerfile`). Each record holds `Cmd`, `WorkingDir`, `User`, `ExposedPorts` and the
platform. Only a Dockerfile's `FROM`, `WORKDIR`, `USER`, `EXPOSE`, `CMD`, `ENTRYPOINT` and `ENV`
instructions set these fields, so the records hold for any linux/amd64 build of the pinned
Dockerfiles. Layers you add on top must not change them. The records come from building just
those instructions over each pinned base image. They were checked against prebuilt images on
OpenSandbox for the user, the default working directory and the environment.

#### Isolation this profile drops

Separate sandboxes cannot enforce some Docker settings in these files. The profile's
`compose_ignored_settings` removes them instead of rejecting the task:

| Setting | Where | Effect |
|---|---|---|
| `read_only`, `tmpfs` | `noisy-blackbox-optimization/public-evaluator` | The evaluator's root filesystem is writable, and `/tmp` is not a size-capped `noexec` mount. |
| `pids_limit: 128` | `noisy-blackbox-optimization/public-evaluator` | No per-service process limit beyond the sandbox's own. |
| `cap_drop: [ALL]` | `noisy-blackbox-optimization/public-evaluator` | The evaluator keeps the sandbox's default capabilities. |
| `security_opt: [no-new-privileges:true]` | `noisy-blackbox-optimization/public-evaluator` | Setuid binaries can raise privileges inside the evaluator sandbox. |
| `init: true` | `inverse-lithography/fabricator`, `noisy-blackbox-optimization/public-evaluator` | No init process wraps the service command. |
| `networks` (the `internal: true` network `task`) | `protein-active-learning` | `assay` and `submission` share the default network with `main` and have public egress. |

The evaluator's `cap_add` (`KILL`, `SETGID`, `SETUID`) is checked, not dropped. The profile
configures a capability probe for each, run as root in the evaluator sandbox, and the task fails
at seed if one is missing. OpenSandbox grants all three to root by default.

As in Harbor, the agent cannot read a sidecar's filesystem, but it can reach every service over
the network.

#### `protein-active-learning` submission

`submission` runs as a non-root user, and OpenSandbox rejects running commands as root in a
non-root sandbox, so Gym cannot add its peers to `/etc/hosts`. The profile's
`environment.compose_service_overrides` instead passes its only peer URL as
`ASSAY_STATUS=http://assay:8080/status`, rewritten to the `assay` sandbox's address, and runs
`submission` as its image's default user (see the
[resources server README](../../resources_servers/terminal_bench_4/README.md#non-root-compose-services)).
Terminal-Bench 4's `medical-claims-processing` is handled the same way, but its Compose file
already passes that address in `BROWSER_URL`. The v0.1 `submission_server.py`
hardcodes it, so build `submission` from a server that reads it from the environment with the
same default:

```python
ASSAY_STATUS = os.environ.get("ASSAY_STATUS", "http://assay:8080/status")
```

To run only the 65 single-container tasks, use `++prepare_script_args.category=cpu`.

## Scoring

The reward is the verifier's official binary `reward.txt`, and a trial passes only when it is 1.
The official protocol is three trials per task, with reward averaged over all 210 trials. Like
Terminal-Bench 4's profile, this one runs one trial per task; add `++num_repeats=3` for the
leaderboard's repeat count. `mean/reward` reports the average over scored trials; the mini-SWE
preset has the coverage limit described in the [Measurement contract](#measurement-contract).

When the agent has no infrastructure failure, a verifier timeout or a missing or invalid reward
after the verifier runs produces a measured zero included in `mean/reward`. Harbor preserves a
valid reward written before the verifier times out; the resources-server presets score that
timeout as zero. Provider or transfer failures that prevent reliable verification remain masked
or unscored and excluded.
The leaderboard's denominator counts every trial, so count excluded failures and skipped tasks
as fails when you compare with it.

On the Harbor route, an incomplete mini-SWE run with a usable trajectory, including a timeout
or a killed runner, still reaches the official verifier. The `/run` relay instead scores a
killed runner as zero without verification.

Every Terminal-Bench-Science verifier also writes a CTRF report to `/logs/verifier/ctrf.json`.
The Codex and `miniswe_legacy_agent` resources-server presets project its test counts
(`num_tests_passed`, `num_tests_total`, `num_tests_failed`
and `num_tests_skipped`) and `partial_credit`, the fraction of tests that passed. Skipped tests
count as not passed.

`partial_credit` is never below the official reward. An official pass keeps full credit even
when pytest skipped some tests, and a verifier without a usable report gets the official
reward. `mean/partial_credit` is reported beside `mean/reward` and never replaces it. Partial
credit is not comparable with leaderboard scores.

For those presets, recognized verifier timeout and missing or invalid reward outcomes carry
0 in both `reward` and `partial_credit` and remain in both means, unless an agent infrastructure
failure takes precedence. Masked infrastructure failures are excluded. The Harbor preset keeps
verifier logs but does not project CTRF test counts or `partial_credit` into rollout rows, so it
does not report `mean/partial_credit`.

## Profile

`terminal_bench_science/miniswe` (`miniswe.yaml`) runs mini-SWE 2.4.6 through the
[Harbor environment server](../../environment_servers/harbor/README.md) on the task folders of
[`environments/harbor/terminal_bench_science`](../../environments/harbor/terminal_bench_science/README.md)
(`gym dataset pull terminal_bench_science` fetches them). The server owns each task sandbox, its
Compose group and the separate verifier, and lends the sandbox to
[`mini_swe_agent`](../../responses_api_agents/mini_swe_agent/README.md#agent-sessions)'s agent
session; each row carries the official instruction and the task's own agent budget. Model calls
go through Gym's model server and are captured; token IDs and logprobs are not (see the agent's
README). `--agent-type` runs another harness on the same tasks, with that harness's own shipped
limits; only the episode bound and the server settings below carry over.

The defaults follow Harbor's built-in `mini-swe-agent`:

- No step limit.
- No cost limit.
- The package's `mini.yaml` prompts.
- mini-SWE's 30-second `LocalEnvironment` command timeout.
- An 8-hour wall-clock limit, every task's own budget.

You can override these with `++tbsci_max_steps=...`, `++tbsci_step_timeout_sec=...` and
`++tbsci_agent_max_timeout_sec=...`, as with Terminal-Bench 4's `tb4_*` settings. Terminal-Bench
4's profile follows Artificial Analysis's methodology instead (500 steps); this one follows
Harbor's agent because the leaderboard ran Harbor. Harbor installs the latest mini-swe-agent
unless you pin a version, so pin it when you compare against Harbor runs.

The public leaderboard uses the Codex, Claude Code and mini-SWE-agent harnesses; there are no
Terminus-2 entries.

How the profile runs:

- **Prompt.** The Harbor environment server gives mini-SWE each task's `instruction.md`, which
  mini-SWE renders with its `mini.yaml` templates. No rows are prepared.
- **Images.** Each task's agent, verifier and sidecar images come from its Dockerfiles, built
  and recorded when the dataset is prepared (see the entry's README); the resources server's
  `tbsci_*_image` settings do not apply.
- **Runtime.** The agent server uploads mini-SWE's pinned runtime into the task sandbox at seed,
  so the sandbox needs no PyPI access for it. It needs glibc 2.28 or newer and about 450 MB
  under `/tmp`; every v0.1 agent image is Debian or Ubuntu based, and none sets `[agent].user`.
- **Run budget.** Each task's own 8-hour budget bounds the agent; `++tbsci_agent_max_timeout_sec`
  can only shorten it.
- **Episode bound.** 46800 seconds (`++tbsci_episode_timeout_sec`): harness setup (900
  seconds, mini-SWE's install allowance), the 8-hour agent budget, the separate verifier's start
  and the slowest verifier (12000 seconds).
- **Verifier settings.** The entry's `dataset.toml` carries the verifier timeouts and the
  verifier CPUs and memory from [Deviations from the packages](#deviations-from-the-packages),
  and each package's no-network verifier restriction (see the entry's README).
- **Deployment settings.** The preset sets the three sandbox labels from [Execution](#execution)
  (`nemo.nvidia.com/resources: custom`, `nemo.nvidia.com/oom-kill: process`,
  `nemo.nvidia.com/s3-hostpath: 'false'`) and the resources server's 1200-second readiness wait
  for lazily pulled images. It does not raise the disk limit: each sandbox's limit is the
  package's `storage_mb`, where the resources server's presets raise it to 100 GiB.
- **Compose tasks.** The Harbor route rejects three of the five tasks during preparation;
  the other two are untested. It does not apply the resources server's per-task Compose
  settings; see the entry's README.
- **Model address.** mini-SWE calls the model server from inside the sandbox at the address the
  Gym servers bind (see [Run](#run)); there is no separate sandbox-side URL.
- **Working directory `/`.** `ode-law-discovery` and `geometric-pharmacophore-alignment` set no
  `WORKDIR`, so their agent starts in `/`. `mini_swe_agent` refuses a working directory that
  contains its session directory under `/tmp`, so these two tasks are refused at seed. Each is
  recorded once in `rollouts_failures.jsonl` as a terminal failure with no reward, so it is left
  out of `mean/reward`; their reference solutions pass on this path. Count them as fails when you
  compare with the leaderboard, or run them with `terminal_bench_science/codex` or the `/run`
  profile.

### `/run` profile

`terminal_bench_science/miniswe_legacy_agent` (`miniswe_legacy_agent.yaml`) is deprecated. It keeps
the previous profile unchanged: [`miniswe_sandboxed_agent`](../../responses_api_agents/miniswe_sandboxed_agent/README.md)
over the `legacy_agent` `/run` relay of the resources server, on identity rows (`prepare.py`,
`data/benchmark.jsonl`), with the instruction delivered by the resources seed. It installs
mini-SWE from PyPI in the sandbox, runs all 70 tasks including the two above, keeps its agent
artifacts under `<tbsci_jobs_dir>/agent`, and accepts `++tbsci_sandbox_model_base_url` for a cell
that reaches the model server only through a proxy. The same `tbsci_max_steps`,
`tbsci_step_timeout_sec` and `tbsci_agent_max_timeout_sec` settings apply. It stays for
comparison until the relay is removed; the two adapters are configured differently, so do not
compare their scores as the same measurement. `data/example.jsonl` and
`data/example_rollouts.jsonl` belong to this profile.

### Codex in the task sandbox

`terminal_bench_science/codex` (`codex.yaml`) runs the Codex CLI inside each task sandbox
through [`codex_agent`](../../responses_api_agents/codex_agent/README.md), composed with the
resources server and `single_agent_turn_legacy`, which seeds both, closes Codex before grading,
then closes the resources session. The resources server lends the task sandbox through
`SandboxAccess` (see its README); grading, partial credit, artifacts and cleanup are the same
as for mini-SWE. Defaults:

- Codex `0.155.1` (`++tbsci_codex_version`), installed in the sandbox with a private Node.
- The task's 8-hour budget as Codex's timeout (`++tbsci_agent_timeout_sec`).
- Codex sees the image's environment and `HOME`, as when Harbor starts it in the container.
- `model` (the name Codex sees, so the catalog entry it applies) defaults to the served name;
  set `++tbsci_codex_model=...`. The model server still substitutes its own model upstream.
- No reasoning effort, unless `++tbsci_reasoning_effort=...` and
  `++tbsci_model_server_forwards_reasoning_effort=true` are both set.
- The CLI's own provider settings: a 300-second stream idle timeout
  (`++tbsci_stream_idle_timeout_ms`) and Codex's default request and stream retries
  (`++tbsci_codex_request_max_retries`, `++tbsci_codex_stream_max_retries`).
- A Codex run that fails or dies while its sandbox still runs is graded as a nonzero agent exit,
  as Harbor grades it (`++tbsci_verify_after_agent_error=false` retries it instead); a lost
  sandbox or a model-endpoint outage is still retried. A provider failure Codex reports itself
  goes to the environment's outcome policy (#4834), which by default grades a rejected request
  (400, 413, 422) and retries a transient one. See `codex_agent`'s `verify_after_agent_error`.
- Codex and its tools get `oom_score_adj` 1000 (0 when the sandbox starts below zero), so an
  OOM kill inside the sandbox picks agent work, not the exec daemon
  (`++tbsci_raise_oom_score_adj=false` turns it off).
- Codex's session rollouts, with per-call token counts and timestamps, are copied to
  `/logs/agent/sessions` (`++tbsci_codex_sandbox_logs_dir`). The resources server keeps them in the trial directory
  (`agent/sessions/`), as Harbor keeps them under `/logs/agent`.

Rows carry the official instruction as their prompt. `prepare_codex.py` reads it from the
pinned packages and checks every selected task's 8-hour budget. Verifying the package hashes
needs `pathspec`, which Gym's own environment leaves out, so the script runs itself in the
resources server's environment, setting that up first as `gym env start` would:

```sh
gym eval prepare --benchmark terminal_bench_science/codex \
  '+prepare_script_args.task_names=[mri-harmonization,eeg-erp-recovery]'
```

### Codex with DeepSeek V4.1 Flash

`codex_deepseek_flash.yaml` is the run config for the leaderboard's DeepSeek V4.1 Flash / Codex /
effort max row with a self-hosted model: Codex `0.154.0` with no model catalog, effort `max` and
the full 8-hour budget. Sampling and output limits belong to the chat-completions server; Codex
sends none.

Gym's `vllm_model` is the policy model server (#5155, with #4880's custom-tool bridge). Codex sends
its Responses requests to it, and it turns them into chat completions for a server that serves the
model as `deepseek-flash` (vLLM with DeepSeek's tokenizer, tool and reasoning parsers). The run
config sets:

```yaml
# vllm_model, the policy model server
custom_tool_strategy: function_bridge      # Codex's freeform apply_patch as a function tool
drop_custom_tools: false                   # dropping would run before the bridge
drop_hosted_tools: true                    # tools the chat server cannot run
forward_unvalidated_reasoning_effort: true # effort max
drop_unrepresentable_request_fields: true  # include and reasoning.context all_turns
inline_tool_output_images: true            # view_image output, for a vision model
stream_keepalive_s: 15                     # under Codex's 300-second stream idle timeout
```

Each call reaches the server with `X-Request-Id: <rollout id>-<8 hex>`, so a router in front of
several replicas can keep a rollout's calls together.

Earlier runs put an `openai_model` relay in front of a separate `vllm_model` with these settings.
On a local test rig the two setups send the chat-completions server the same requests, field for
field apart from per-call ids. That earlier setup scored 31/210 (14.8%) over 70 tasks with three
trials each (`++num_repeats=3`), against the leaderboard row's 33/210 (15.7%).

Pass the server and its key as overrides: `--model-url` is the server's `/v1` base URL, and the
key comes from `POLICY_API_KEY` (or the variable named by `++tbsci_model_api_key_env`). Load the
file with `--config` and no `--model-type`:

```sh
export POLICY_API_KEY=...  # any string if the server has no auth
gym eval run --config benchmarks/terminal_bench_science/codex_deepseek_flash.yaml \
  --split benchmark --model-url http://SERVER_HOST:PORT/v1 \
  --output results/tbsci-deepseek/rollouts.jsonl --concurrency 8 ++default_host=GYM_HOST_IP
```

Codex's requests match those of a plain `codex exec` launch with the same `config.toml` model
settings, as Harbor 0.24 starts it: the agent leaves Codex's feature defaults alone
(`features: {multi_agent: null, code_mode: null}`, so the multi-agent tools stay), and sandbox
sessions run Codex without `--ephemeral`, which keeps its goal tools. Gym's own `config.toml`
keys (provider name, update checks, analytics, history) do not change the requests.

## Run

The `gym` CLI does not read an env file, and the sandbox provider block falls back to `localhost`
when `OPENSANDBOX_DOMAIN` is unset. Export the OpenSandbox credentials first:

```sh
export OPENSANDBOX_DOMAIN=... OPENSANDBOX_API_KEY=...
```

mini-SWE runs inside each remote sandbox and calls Gym's model server from there, so the Gym
servers must listen on an address the sandbox cell can reach. `++default_host` binds them to
that address. `++use_absolute_ip=true` binds to `gethostbyname(gethostname())` instead, which
is `127.0.1.1` on many Ubuntu hosts and unreachable from the cell. With the `/run` profile, if
the cell reaches the model server only through a proxy, also set
`++tbsci_sandbox_model_base_url=...`.

### Harbor preset

```sh
gym dataset pull terminal_bench_science
gym eval run --benchmark terminal_bench_science/miniswe --sandbox opensandbox \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tbsci/rollouts.jsonl --concurrency 8 \
  ++default_host=GYM_HOST_IP
```

The run prepares the pulled task folders first: it builds each task's agent, verifier and
sidecar images and records them, unless they are already recorded (see the entry's
[Images](../../environments/harbor/terminal_bench_science/README.md#images)). `--only-tasks` and
`--exclude-tasks` select tasks. `gym dataset validate terminal_bench_science --sandbox
opensandbox` runs every reference solution through the same separate verifier. The resources
server's `tbsci_*` image, split-endpoint and verifier settings do not apply to this preset.

### Resources-server presets

With `++tbsci_split_sandbox_endpoints=true`, export `OPENSANDBOX_DOMAIN_CPU` and
`OPENSANDBOX_API_KEY_CPU` instead. No task requests a GPU. Only the `/run` profile accepts split
endpoints; the Codex profiles lend sandboxes through one provider block.

A run config keeps the model key off the command line and can hold the per-task image map.
For example, `tbsci-run.yaml`:

```yaml
policy_api_key: ${oc.env:POLICY_API_KEY}
tbsci_task_images:
  mendota-ice-phenology: {agent: "REGISTRY/x@sha256:...", verifier: "REGISTRY/y@sha256:..."}
```

```sh
gym eval prepare --benchmark terminal_bench_science/miniswe_legacy_agent
gym eval run --benchmark terminal_bench_science/miniswe_legacy_agent --split benchmark \
  --config tbsci-run.yaml \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tbsci/rollouts.jsonl --concurrency 8 \
  ++default_host=GYM_HOST_IP \
  '++tbsci_agent_image="REGISTRY/tbsci/{task}:environment-{digest}"' \
  '++tbsci_verifier_image="REGISTRY/tbsci/{task}:tests-{digest}"'
```

`gym eval prepare` writes the `/run` profile's identity rows with `prepare.py`. For Codex, use
`--benchmark terminal_bench_science/codex` in both commands; its rows come from
`prepare_codex.py`, which runs itself in the resources server's environment when Gym's lacks
`pathspec` (see [Codex in the task sandbox](#codex-in-the-task-sandbox)).

Preparation accepts these filters, which combine:

- `++prepare_script_args.task_names=[...]`
- `++prepare_script_args.category=cpu` or `compose`
- `++prepare_script_args.domain=life-sciences` (or any other domain)

For a capped smoke run, add `++tbsci_max_steps=3 ++tbsci_agent_max_timeout_sec=900`.
Capped runs check setup, execution, verification and cleanup. They are not benchmark scores.
They also log `opensandbox.adapters.filesystem_adapter` 404 tracebacks for declared artifacts
the agent never wrote; these are harmless.

When the model endpoint reports no cached-token count (vLLM without
`--enable-prompt-tokens-details`, for example), each agent log shows a LiteLLM
`Error creating standard logging object` traceback per model call. It comes from LiteLLM's
logging callback; the calls and the run are unaffected.

The profile runs one trial per task; add `++num_repeats=3` for the leaderboard's three.

`data/example_rollouts.jsonl` holds one capped trial (3 steps, 900 seconds) of each example
task, run with the `/run` profile on `data/example.jsonl`. Its local paths and the model server address are
placeholders.
