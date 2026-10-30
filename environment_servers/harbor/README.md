# Harbor tasks

Runs tasks written in the [Harbor](https://github.com/harbor-framework/harbor) task format: a folder with
`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`. Harbor is the input
format; Gym owns execution. The supported subset is listed below.

## What the server does

This is an environment server: one `/run` call owns one whole episode, and no resources server takes part.

- Seed: checks the row's digest against the taskset mapping, snapshots and checks the task files once per
  digest, starts the task image as a sandbox through the configured provider, and creates the working
  directory. A declared `docker_image` starts as written; a pull-mode or overlay-mode Dockerfile's base image
  starts pinned to the multi-arch index digest recorded at prepare (`compose-images.json`), so a moved tag
  cannot change the image under the recorded configuration and each platform's sandbox still selects its own
  manifest. Concurrent episodes share the snapshot. Later source edits cannot change an already cached task;
  re-materialize and restart the server to use edited tasks.
- Agent: opens a session on the configured agent server with a `SandboxAccess` for that sandbox (the same
  seed request every sandboxed harness already accepts), posts the task's `responses_create_params` to it,
  then closes the session and keeps the harness's observations for the result. When the agent instance is
  implemented by `harbor_oracle_agent` (the key under its `responses_api_agents` block), the seed first
  uploads the snapshot's `solution/` to `/solution`, as root like its other uploads and readable by the task
  user, and the sandbox access carries the host-resolved `[solution].env` as `env`; that is where Harbor's
  oracle copies and runs the solution. No other harness receives the solution, and a task without
  `solution/` uploads nothing (the oracle reports it unvalidated).
- Verify: clears `/tests` and the grader's own files in `/logs/verifier` (`reward.json`, `reward.txt`, the exit
  record and `test-stdout.txt`; the rest of the folder, a sidecar's input, stays), uploads the snapshot's
  `tests/`, runs `bash /tests/test.sh` in the same sandbox, and downloads the verifier logs.
- Cleanup: stops the sandbox after the episode, whatever happened before. Cancellation and the episode
  deadline also reach this cleanup; a failed stop keeps the sandbox owned by the server for a later retry
  or shutdown, and a redelivered episode cannot provision while an owned sandbox of the same episode remains.

A task that cannot be set up is a handled episode failure with `stage: seed`: terminal when the task or row
is at fault (unknown task, changed folder, unsupported settings), retryable when the sandbox provider failed.
A harness that cannot be seeded, fails its turn or fails to close is `stage: agent` or `stage: cleanup`, with
its observations and any usable response attached. A task that declares `[[environment.mcp_servers]]` or
`skills_dir` is a terminal `stage: agent` failure unless the harness declared `supports_task_tool_context` in
its seed response: running it without its tools would score a silent zero. Verification never runs before the
harness has let go.

A failed agent response that carries `ng_agent_outcome` goes through `agent_outcome_policy`, as on
`single_agent_turn`, before `test.sh` runs: by default an `unknown` or `model_transport_error` cause is a
retryable `stage: agent` failure, `agent_limit` and `agent_invalid_output` are verified, and malformed metadata
fails without retry. Responses without that metadata are verified as before. See
[Add or update a harness](https://docs.nvidia.com/nemo/gym/latest/agent-server/add-or-update-a-harness) for the
policy table and its overrides.

When `test.sh` ran, the sample counts. A missing or invalid reward file scores 0 with a `failure_kind`.
Only a required infrastructure failure (sandbox lost, reward transfer failed) sets `mask_sample`.
Diagnostic logs are best effort: unsafe archive entries are skipped securely, and a diagnostic
transfer failure cannot discard an already measured reward. Verifier timeouts score zero and retain
available logs. An existing `reward.json`
takes precedence even if malformed; `reward.txt` is read only when JSON is absent. Boolean JSON rewards
are numeric, as in Harbor. In `reward.json` a `reward` key is the score and the other keys are metrics, kept in
`verifier_rewards`; without one every key is a component of the score, which is their sum, and the result names
them as `reward_components` (only keys that sum to `reward`, the contract a trainer asserts; a `reward` key with
other keys names none, since those do not sum to it). Components whose sum is not a finite number are an invalid
reward file: measured 0 with `harbor:invalid_reward`. A multi-step task averages components as it averages rewards
when every measured turn named some, else names none. On the single-turn path the check cannot fail (the reward is
the sum of the same keys); when a roll-up's components do not sum to its reward, the episode names none and the
server logs a warning naming the task, the keys and both numbers.

`reward_components` here is Harbor's optional, task-specific reading of `reward.json`, not the shared multi-reward
contract of `BaseMultiRewardVerifyResponse` (a required mapping with the same objective keys on every task of an
environment): a task names whatever keys its `test.sh` wrote, two tasks of one dataset may name different ones, and a
measured row may name none. It is for per-row inspection and for a trainer that reads the mapping row by row and
takes the scalar `reward` alone where a row has none. Gym pads no missing component and renames no key, so a dataset
meant for a fixed-objective trainer (GDPO and the like) must have every task write the same keys without a `reward`
key; that is the dataset's contract to keep, not the server's.

## Supported task behavior

Supported today: single-step tasks with a prebuilt `docker_image`, a pull-mode Dockerfile (`FROM` plus
`WORKDIR`/`ENV`/`USER`/`LABEL`) or an overlay-mode Dockerfile (the same plus `RUN` lines, which the server
runs in the pulled base image at seed, before the healthcheck and the agent, each with the `ENV`, `WORKDIR`
and `USER` in effect at that line; `ENV`, `WORKDIR` and `USER` resolve at prepare from the base image's
recorded configuration the way `docker build` resolves them, so preparing fails when the registry cannot
resolve the base image. A `RUN` line that exits non-zero or outruns `[environment].build_timeout_sec` is a
422 with the command and the tail of its output; a provider that cannot run one is a 503); shared verifier mode, and
separate verifier mode (the agent's `/logs/artifacts` and `artifacts` entries are copied into the verifier sandbox
first, sidecar hooks and artifacts included, each as the archive the agent's sandbox packed, so the verifier sees the
modes the agent left). In separate mode the verifier's image is the one prepare built from
`tests/Dockerfile` when the task ships that file and its effective verifier environment names no image, as Harbor
builds it (see "Dockerfile builds" below); else `[verifier.environment].docker_image`; else the task's own
`[environment].docker_image` (Harbor's effective verifier environment is then a copy of `[environment]`). A
separate-mode task that names no verifier image and ships no `tests/Dockerfile` is refused as Harbor refuses it, and
one whose `tests/Dockerfile` has no recorded build is left out at prepare like an unrecorded `environment/` build;
the server refuses either at seed. Compose environments start as
a sandbox group when a `compose-images.json` with the sidecar images' recorded OCI configuration sits next to
the task folders, written by `gym dataset fetch`; on OpenSandbox the provider block also needs
`networking.enabled: true` (with `loopback_forwarding`) and `runtime_requirements.shm_size_metadata_key` (see
[Compose tasks on OpenSandbox](#compose-tasks-on-opensandbox)); each service's working directory is created once its sandbox is up, as Docker creates a missing `working_dir` or `WORKDIR` at container create. `[environment.healthcheck]` is polled at seed.
A Compose file's `${NAME}` resolves as Harbor's `docker compose` would: Gym's infrastructure variables, the task's
`[environment].env`, the process environment, then `environment/.env`; a name none of them sets is an empty string
at prepare, with one warning. The server's `compose_variables` fill such names at seed, below everything the task
and Harbor set (in a job-level overlay file, above what the task sets: see "Job-level Compose overlay"): each maps a
name to a literal `value`, a process `env` variable, or a `model_server`, whose
sandbox-reachable URL with the episode's rollout prefix becomes the value (a sidecar's model calls are then
recorded with the rollout and never become training tokens). A name still unset at seed fails the episode
terminally, before provisioning. A document holding such values is written to a private folder removed once the
group starts, not under `artifacts_dir`.

```yaml
compose_variables:
  LITELLM_BASE_URL: {model_server: user_model}
  LITELLM_API_KEY: {env: USER_MODEL_API_KEY}
  SIM_USER_MODEL: {value: gpt-5.5}
```
Tasks that declare GPUs run on the `gpu_sandbox_provider` block when one is set. The task's
`[[environment.mcp_servers]]` and `skills_dir` are written to `/tmp/.nemo-gym/task.json` in the agent's
sandbox for harnesses that drive task MCP servers from inside it (mini-SWE does). With `task_mcp_grants:
optional` or `required`, the `streamable-http` URL entries are also granted to the agent as MCP tool
accesses, as declared; a Compose sidecar's URL names its service, which resolves inside the task's group.
A task that declares MCP servers or `skills_dir` is refused to a harness that does not declare
`supports_task_tool_context`, since its tools would never reach it, unless every declared MCP server was granted
with `task_mcp_grants: required` and the task has no `skills_dir`: a harness that cannot establish a required
grant fails its seed or its first turn (Hermes and Pi connect at the first activation) instead of running without
the tools. Harbor's default transport is `sse`, which cannot be granted, so a task meant for such a harness must
say `transport = "streamable-http"` explicitly.
Dockerfiles outside pull and overlay mode are built at prepare, see below.

## Multi-step tasks

A task with `[[steps]]` runs each step as one turn in the same sandbox, as Harbor's multi-step trial does
without `resume_trajectory`: a fresh agent session per turn, with the step's
`steps/<name>/instruction.md` as its only input and the step's `[steps.agent].timeout_sec` (else the task's)
as its budget. `turn_sessions: continue` on the server is Harbor's `--resume-trajectory`: the first turn seeds
one multi-turn continuation session and every later turn is the next activation of it, so the agent keeps its
conversation across steps; each step is still verified on its own, the session is closed once, after the last
turn's activation and before that step's verifier, and its observations and cumulative trajectory are the episode's (`ng_agent_observations`,
`ng_trajectory`; the per-turn results carry none). The result's top-level `responses_create_params` and `response`
are then the last activation's input and its new output alone, produced with the earlier steps in the harness's
own conversation, not a standalone prompt and completion: a continued episode's record is the cumulative
`ng_trajectory` (every trajectory turn names its activation) with, when
[model-call capture](https://docs.nvidia.com/nemo/gym/latest/model-server/model-call-capture) is enabled at the
model server (`observability_enabled`, opt-in outside `gym eval submit`), the captured model calls, which hold what
the model saw and produced. The run is refused before any sandbox when the harness
declares no multi-turn continuation (`GET /v1/agent_sessions/capabilities`; only OpenCode does today), as
Harbor refuses `resume_trajectory` for an agent without `SUPPORTS_RESUME` before any environment spend, and when
the task's steps switch the agent user, which a live session cannot follow. The setting is a no-op for a
single-step task, as in Harbor. One divergence: Harbor's resumed agent has exited between steps, and so has the
continued harness's process (an activation returns once the process it launched has stopped), but between steps
the session keeps its conversation state on disk in the sandbox and the borrowed connection the next activation
reuses, through every step's verifier but the last's; services the task itself started are untouched between steps
in either mode. The task's top-level `instruction.md` is never sent; a multi-step task may keep material there
the agent must not see. After each turn the shared verifier runs the task's `tests/` with
`steps/<name>/tests/` on top, under the step's verifier user and environment over the task's and the step's own
`[steps.verifier].timeout_sec`, else Harbor's step default of 600 s (times a dataset `timeout_multiplier`, like every
budget), never the task's `[verifier].timeout_sec`
(Harbor's `StepConfig.verifier` is a full `VerifierConfig` with that default, so the task's figure never reaches a
step; the loader warns about a task whose `[verifier].timeout_sec` differs and whose steps name none). A turn
whose failed response the harness classified goes through `agent_outcome_policy` first, like a single-step
task's: a retry or fail cause ends the episode at that turn, ungraded. Before
every turn that follows a turn verified in the agent's sandbox (a shared turn), `/tests` and `/logs/verifier` are
emptied and the server's `/logs/agent/trajectory.json` removed, so the next agent does not open beside the previous
turn's tests, reward or transcript; a failure to clear them fails the episode before the next agent session opens.
After a turn verified in a sandbox of its own nothing is reset, since nothing was graded in the agent's sandbox
(Harbor 0.22.0's separate mode never touches its `/tests`). Harbor 0.22.0 clears them only before each step's
shared verifier (Harbor main clears them before the step when the previous step ran a shared verifier there, which
this follows); Harbor also empties (or, for a bind-mounted `/logs/agent`, moves away) the whole `/logs/agent` before
each non-resumed step, here the agent's own files there stay. A step's
`[steps.verifier].user` or `[steps.agent].user` that names a group (`name:group`, `uid:gid`) is refused at seed
on every provider but Docker, as the task's own users are, naming the step, before any sandbox exists. Each turn's
agent runs as the step's `[steps.agent].user` (else the task's `[agent].user`, else the image's `USER`), as Harbor
runs each step: when a step's user differs from the one in place, the sandbox's default exec user, the working
directory's owner and the user the harness is handed switch to it (a numeric uid resolved to its name, as at seed)
and switch back for a later step that sets a different one; the verifier keeps its own user. The oracle gets the
step's own `steps/<name>/solution/` at `/solution` before each turn (falling back to the task's top-level
`solution/`), as Harbor's oracle resolves each step's solution, so `gym dataset validate` runs a task whose
solutions are per step; such a task counts as validatable when every step has a solution (its own or the task's),
and is listed unvalidated otherwise. A step's `steps/<name>/workdir/` is merged into the working directory before
its turn (an upload over what the previous turns left, as root with the folder's own modes, the digest's ignored
names left out; a symbolic link a previous turn's agent planted at a shipped name is removed as root right before
the extraction, so BusyBox tar, which would follow it and write the shipped files through it, lands the merge where
GNU tar and Harbor's `docker cp` do) and its `steps/<name>/workdir/setup.sh`, when the folder ships one, is then run as
`bash <workdir>/setup.sh` from the working directory as the turn's agent user, within `step_setup_timeout_s` (unset
by default: the step's own agent budget, `[steps.agent].timeout_sec` else `[agent].timeout_sec`; Harbor bounds the
script by nothing but the trial's timeout), a bound the episode budget counts for every step that ships the script:
Harbor 0.22.0's `_prepare_step` order (`_upload_step_workdir`, `_run_step_setup`), under the step's agent user,
before the oracle's solution is placed and the agent session opens. A `setup.sh` that exits non-zero (or runs past
that bound) is the step's failure, as Harbor records it in the step's `exception_info`: the turn is recorded
with `failure_kind: harbor:step_setup_failed`, no response and no verifier result, no agent session or verifier
runs for it, and the episode stops after it, scored as a measured 0 for that turn: it counts in `mean` and makes
`final` 0, and the episode carries the step's failure kind and reason. That is a deliberate divergence from Harbor,
whose `_aggregate_step_rewards` leaves a step without a verifier result out of its denominator and whose `final`
then has no reward at all: the script ran in the sandbox the earlier turns' agent had, where a process it left
running or a working directory it corrupted can fail it, and a masked sample (or a mean over the earlier turns
alone) would give an RL policy no signal against wrecking the next step's setup. The mean is still over the turns
that ran: the steps after the failed one never ran and are not in the denominator (Harbor's step loop stops at a
step whose preparation failed and records none after it), so in a task of three or more steps a wrecked second step
scores the first turn's reward over 2, not over the task's step count; the policy loses the step it broke, not the
steps after it. A task whose own step preparation is broken fails the same way in every rollout, the oracle's
included, under that failure kind. An episode whose
first turn fails there has no rollout at all and is a terminal agent-stage failure with that kind (masked: no agent
has acted). An entry of
`workdir/` the sandbox refuses to unpack over what the previous turns left (a non-empty directory at the name of a
shipped file, which the previous turn's agent can plant in the working directory it owns) is the step's failure as
well, `failure_kind: harbor:step_workdir_failed` (Harbor's `docker cp` refuses the same mismatch and its trial
errors; the sandbox is sound, so it is not an outage to retry). Nothing stops what an earlier turn left running
before the script runs: a process it left behind can replace the uploaded `setup.sh` in the working directory it
owned, so a step whose agent is root after a step whose agent was not must not rely on `setup.sh` to undo what the
earlier step did (Harbor runs it the same way; the loader warns about such a task). A `setup.sh` beside
`instruction.md` instead of inside `workdir/` is not a Harbor file: the loader warns and nothing runs it. A step's
`[steps.healthcheck]` is then polled to a pass (after the setup, not after a failed one; Harbor's
`_run_step_healthcheck`), from the working directory as the turn's agent user, with the same Docker-style timing as
the seed's checks (`start_period_sec` of uncounted attempts every `start_interval_sec`, then `retries` consecutive
failures every `interval_sec` fail it; an attempt killed at its `timeout_sec` plus the 5 s exec grace counts as one
failed attempt, as Docker's `HEALTHCHECK` counts it, where Harbor 0.22.0 errors the whole trial on the first attempt
that overruns, since its Docker exec raises and no caller of `run_healthcheck` catches it); the task's
`[environment.healthcheck]` ran at seed and is not run again per
step, as Harbor runs it once at environment start. A check that never passes is the step's failure too:
`failure_kind: harbor:step_healthcheck_failed`, the same failed turn and early stop as a failed `setup.sh`; a
sandbox that cannot run the command (stopped, a provider error) is an outage, retried as a failed `setup.sh` exec is.
The episode budget counts the check's worst case for every step that declares one (the start period, one attempt
that may start just inside it and its `start_interval_sec` sleep, then `retries` counted attempts with an
`interval_sec` sleep between them, each attempt `timeout_sec` plus the 5 s exec grace).
After each turn's agent session and before its verifier, Harbor's per-step collection pass runs, in shared mode too
(0.22.0 `_collect_step_artifacts`: the trial collects before the step's verifier, whatever `StepConfig.artifacts`'s
description says about "after"): the task's `[[verifier.collect]]` hooks and then the step's own
`[[steps.verifier.collect]]` hooks for `main` (Harbor's `_collect_hooks_for`: the task's, then the step's), then
`/logs/artifacts`, the task's `artifacts` and the step's own `[[steps]] artifacts` entries from `main`, then the
sidecars' hooks and entries, each staged under its numbered root in that turn's verifier logs folder
(`artifacts/NN/<destination>`,
the same layout as a separate verifier's staging) with the pass recorded in `artifacts/manifest.json` (one entry per
declared artifact: `source`, `service`, where it was declared as `origin`, its `destination`, its `type` and a
`status` of `ok`, `absent`, `link`, `failed` or `skipped`); the archives a shared turn does not restore are
removed once the pass ends. Nothing is stopped between turns: the sandbox and its sidecars keep running for the
turns that follow, except on the last turn of a task whose last step verifies separately, where the agent's
container is stopped before the sidecar pass and the whole agent side before the verifier, as Harbor's `_run_step`
does on the last step of a separate-mode task. The hooks run once per turn, in that pass, not again inside the
shared verifier.
A collection the sandbox cannot serve masks the turn as an outage; an artifact the agent's tree defeats only warns,
as for a single-turn task. A single-turn task in shared mode still collects nothing (Harbor collects its task-level
artifacts there too; a follow-up).

Each step's verifier mode is its own, as Harbor's `resolve_step_verifier_mode` resolves it: the step's explicit
`[steps.verifier].environment_mode`, else `separate` when the step declares a `[steps.verifier.environment]`, else
the task's mode; steps of one task may mix (each turn reports its `verifier_mode`, the episode's is the one they
shared or `mixed`). A turn that verifies separately runs its `test.sh` in a verifier sandbox of its own, created and
stopped around that turn's verification under the episode's verifier key, the single-step machinery on a per-turn
view of the task: the sandbox comes from the step's effective environment (`[steps.verifier.environment]`, else
`[verifier.environment]`, else a copy of `[environment]`, Harbor's `resolve_effective_verifier_env_config` with a
step), so a declared image runs as it is, a copy of `[environment]` with a `docker_image` runs the task's own image
(overlay and all), and without an image the turn's verifier is the image prepare built from `tests/Dockerfile`
(Harbor starts a step's verifier from `steps/<name>/tests/` when that folder exists, else from `tests/`; prepare
builds `tests/Dockerfile` only, so a step that would be built from its own folder is refused by the loader, naming the
image to declare, and a step built from `tests/` plans that build whatever the task's own mode). The verifier sandbox
receives exactly what Harbor gives it: the turn's collected archives restored at their own paths (`/logs/artifacts`,
the task's and the step's artifacts, as collected from `main` after the agent), the task's `tests/` with
`steps/<name>/tests/` over it unless the image ships `/tests/test.sh`, and the transcript; nothing else from the
agent's sandbox, which no turn's verification stops but the last separate one, since the next turn's agent session
opens on it (Harbor stops the agent environment before the verifier only on the last step of a separate-mode task,
and so does the server). The step's `[steps.verifier]` user, env and timeout
apply over the task's as in shared mode; `separate_verifier_for_artifacts` forces, per turn, the turns whose task or
step declares artifacts into the task image's fresh sandbox. The episode budget counts one verifier sandbox start
(and the overlay build when it applies) per separate turn. A `[verifier]` network restriction needs every step to
verify separately (a shared turn cannot be restricted); per-step `[steps.verifier]` network restrictions stay
refused.

The episode stops after a turn that is masked, writes no valid reward file, or misses the step's
`min_reward` (a number gates `reward`, a table gates each named key). `multi_step_reward_strategy` rolls the
turns up: `mean` (the default) averages every turn that ran, a turn whose `test.sh` produced no result counting 0
with its failure kind on the episode (Harbor leaves such a step out of its denominator; Gym scores a verifier run
that produced nothing, so a broken later verifier cannot lift the mean), and `final` takes the last turn that
ran. The pre-1.3 spellings `version` and `multi_step_reward_mode` (what older multi-step generators such as the
MemoryArena adapters write) are read as Harbor 0.22.0 reads them: `version` is `schema_version`, and
`multi_step_reward_mode = "aggregate"` is dropped and the task rolls up with the default `mean` (or its own
`multi_step_reward_strategy`), as under Harbor, which has no such field; any other value of that key is dropped
too, with a warning naming it, never read as a strategy. The result lists every turn under `turns`; its
`response` is the last turn's whose agent ran, and each turn carries its own session's `agent_observations` and
`trajectory` (one record per step, as Harbor keeps one `steps/<name>/agent/` per step) while the episode-level
`ng_agent_observations` and `ng_trajectory` stay empty when more than one fresh session ran (a single fresh
session is the episode's; merging several is issue #5063). Each turn is seeded with its step's agent budget and hands the verifier an
ATIF transcript of its own request; with `enforce_task_timeouts` the task deadline sums every turn's harness
setup, agent budget, hooks and verifier time. A long task's turns can budget more than
`default_episode_timeout_seconds`. The seed sizes three figures per episode: what the steps need once the sandbox
is ready (every step's harness setup, agent budget, hooks and verifier time), the nominal episode (one
`sandbox_ready_timeout_s`, and the overlay's `build_timeout_sec`, before that), and the worst case the episode
deadline opened before the seed has to cover (the ready timeout and its grace per image pull attempt,
`image_pull_retries + 1` of them), so a seed that needs a pull retry or a slow sidecar start cannot eat the final
turns' budget. With `enforce_task_timeouts` the episode deadline and the sandbox lifetime follow the worst-case
bound when it is above the default (an expired episode is retried, never scored, so the default would cut it off);
the seed logs this once per episode at WARNING level, naming all three figures. With `enforce_task_timeouts: false`
the default is the deadline: a task whose nominal episode does not fit it is refused at seed naming the setting
(raise `default_episode_timeout_seconds` to at least the nominal figure the message gives, the worst-case one to
cover a retried seed, or turn enforcement on), and one that fits only without a retried seed runs, with a WARNING
that an episode whose seed retries its image pulls may time out and be retried. A multi-step task's Compose sidecar without `deploy.resources` runs at the sandbox provider's default size
(under Harbor's `docker compose` it has the host's resources); the seed logs, once per task, which sidecars run at
the default and where a size goes (`deploy.resources.limits` in the task's Compose file), and a sidecar whose
process dies at start fails the group with a bounded tail of its command log, not only `unhealthy: exit status
1`. See the multi-step tasks docs page under
`fern/versions/latest/pages/data/` for the full per-turn contract.

## Dockerfile builds

Prepare decides one of three image modes per distinct Dockerfile.

- Pull. A base-image-only file (`FROM` plus `WORKDIR`, `ENV`, `USER`, `LABEL`). The sandbox starts from the
  base image.
- Overlay. A pullable `FROM` plus `RUN` lines. The server runs the lines in the pulled base image at seed.
  No build and no registry.
- Build. Everything else: `COPY`, `ADD`, heredocs, BuildKit flags, multi-stage files. The task's
  `environment/` folder is the build context. Prepare builds it once, records it, and the task then runs in
  pull mode from the built image. A `build:` the task's Compose file gives `main` (build args, a target, a
  `dockerfile_inline`) applies to that build; at seed the recorded image stands in for it, so the loader drops
  the section, and a `pull_policy: build` naming it, from every service prepare built before the Compose
  adapter, which starts images only, sees the document. A task in pull or overlay mode whose file only restates
  `main`'s build (`context: ${CONTEXT_DIR}` or `.`, `dockerfile: Dockerfile`, `pull`) has that section dropped
  too, as Harbor's own template already supplies it; a build key Gym cannot apply without a build (`args`,
  `target`, `dockerfile_inline`, another `dockerfile` or `context`) leaves the task in `skipped.json` naming the
  key.
  A `pull_policy: build` goes with the build it names and is dropped with it (a built service's, a restated
  `main`'s); every other `pull_policy` is the service's runtime pull policy and reaches the adapter (see the field
  table below). Prepare runs the adapter's field check on each Compose task's resolved document, so a field no
  provider starts (`deploy`, `extra_hosts`, `platform`, a sidecar `build:` prepare recorded no build for) leaves
  that task in `skipped.json` with the service and field named instead of failing every episode at seed. The
  check's first step, the key allow-list, also runs on the raw file of every task about to be built, before the
  build, so a task refused by key is neither built nor pushed and a build that would fail for it cannot stop the
  preparation. A value the check refuses (`count: all`, a `platform` other than the deployment's) is read from
  the resolved document once the tasks load, after their builds, and leaves the task out the same way.

Every other service of the task's `environment/docker-compose.yaml` with a `build:` section (a sidecar) is built
too, whatever `main`'s mode, as Harbor's `docker compose build` builds it: from its own context and Dockerfile
under `environment/`, with its args and target. Its image is named `hb__<hash>-<service>`, the form Compose gives
a service without `image:` under the project name Gym builds with (Harbor's is the per-trial session id, so its
sidecar tags are never published), and a sidecar's own `image:` spelling is accepted and overridden by that name.
Each built service is recorded by name under the context's `built-images.json` entry, and at seed the loader
substitutes the recorded image for every built sidecar and drops their `build:` and a `pull_policy: build`, so the
sidecar starts from the image prepare built with its recorded entrypoint, command and ports like any pulled image,
under its runtime `pull_policy` when it declares one. A sidecar with a `build:` section and no recorded build is
refused by the loader naming it, whatever its `image:` says (under Compose a service with `build:` is built, never
pulled, so that `image:` is only a tag), and so is one with neither key. `main`'s recorded image stands in only when
the task runs from it (build mode): a task that declares a `docker_image`, in `task.toml` or by a dataset override,
keeps that image even when the context's `main` was built for another task sharing it or before the override, as
Harbor's prebuilt mode runs the declared image. The loader hashes `environment/` once per task, when it loads, and
reads the record then; nothing is hashed per episode.

The separate verifier's image is built the same way. Harbor runs a separate verifier in a second environment whose
build context is the task's `tests/` folder and whose configuration is the effective verifier environment
(`[verifier.environment]` when declared, else a copy of `[environment]`); without a `docker_image` there it runs
`docker compose build` on `tests/Dockerfile`, whatever that file holds (a `FROM` plus `RUN` file is built, not
overlaid), and uploads no `tests/` into the image. Prepare plans that build for every task in separate mode with
`tests/Dockerfile` and no declared or inherited `docker_image`: the `tests/` folder is hashed on its own, as Harbor's
environment id hashes it, and the image is named `hb__<tests hash>-verifier` (the hash Harbor exports as
`MAIN_IMAGE_NAME` for its verifier environment, with the sidecar suffix naming the role) and recorded under that
hash as service `verifier` in `built-images.json`, its configuration in `compose-images.json`. The loader reads it,
the server grades in it and skips the `tests/` upload when the image holds `/tests/test.sh` (it uploads the folder
with a warning when it does not). A task whose verifier image fails to build is left out of the preparation like one
whose own image fails, and so is one whose `tests/Dockerfile` has no recorded build (`builds.enabled: false`, or a
record from before verifier builds): the loader refuses it the way it refuses an unrecorded `environment/` build,
naming the file, the record and the two remedies. Harbor's verifier environment dir is always `tests/` (there is no
fallback to `environment/`), so a task in separate mode with neither `tests/Dockerfile` nor a declared or inherited
`docker_image` fails Harbor's environment validation (`require_agent_environment_definition`), and the loader
refuses it with that wording. A `tests/docker-compose.yaml` may declare `main` only (its `build` section is
honoured): Harbor builds and starts any other service as the verifier's Compose project, this server starts a single
verifier sandbox, so such a task is refused by the loader and at prepare before any build. The same file is refused
at load (`gym dataset validate` and the seed alike) when it sets a runtime key on `main`, anything beside `image`,
`build` and `pull_policy` (`environment`, `env_file`, `volumes`, `command`, `network_mode`, `cap_drop`, ...), and,
beside a declared verifier image, `image`, `build` and `pull_policy: build` too; and when it carries a top-level key
beside `services`, `name`, `version` and `x-` fields (`networks`, `volumes`, `configs`, `secrets`, `include`, ...).
Harbor stacks the whole file on its verifier project (`_docker_compose_paths` in docker.py at v0.22.0), so every such
key shapes Harbor's verifier container, while this server starts the verifier from `task.toml`'s environment and the
verifier image and reads the file's `services` alone (`nemo_gym/tasks/harbor/verifier_env.py`); grading it without
the key would differ from Harbor without a word. Among the hub datasets, arc-crm-6's six separate-verifier tasks are
refused this way (their file sets `network_mode`, `pids_limit`, `security_opt` and `cap_drop` on `main`);
cadir-ci-benchmark's two ship the `main`-only shape this rule honours, but they are refused at load before the file
is read, by their `[agent]` allowlist and `[verifier.environment] network_mode = "no-network"` (the same
network-restriction check that refuses every terminal-bench-science task, all 70 of which carry that verifier
setting, the two overlay-mode ones named under "Upgrading from the pre-build layer" below among them).

A task that declares `[environment].docker_image` runs that image, unless `builds.force` is set: then its
Dockerfile is built anyway and the recorded build wins, as Harbor's `--force-build` does (SWE-Interact's
declared images lack the helper its Dockerfiles copy in). The record names the tasks built under `force`
(`forced_tasks`); a recorded `main` is not substituted for a task pinned to an image otherwise.

The image is named the way Harbor names it, `hb__<hash>`, where the hash is Harbor's 32-character content
hash of the `environment/` folder, its environment id (vendored from Harbor v0.22.0 in
`nemo_gym/tasks/harbor/harbor_compat.py`), so identical contexts get the same image name in Harbor and in
Gym. Builds are configured by a top-level `builds:` block in the merged configuration (`env.yaml`, a
`--config` file, or `++builds.<setting>=<value>` on the command line; explicit settings apply over the
defaults below):

```yaml
builds:
  tool: buildah            # buildah, podman or docker (docker compose build); default: the first on PATH that fits
  repository: registry/path  # push built images under this path; default: $NEMO_GYM_BUILD_REPOSITORY
  platform: linux/amd64    # the platform asked of the tool; default linux/amd64
  enabled: true            # false never builds; tasks that need a build are skipped with a message
  force: false             # true also builds the Dockerfile of a task that declares docker_image
```

With `builds.repository` (or the `NEMO_GYM_BUILD_REPOSITORY` variable) the image is named
`registry/path/hb__<hash>` and pushed, which is what OpenSandbox needs. The build then runs with the first
of `buildah`, `podman` and `docker` found on PATH, or the tool named by `builds.tool`.
buildah runs rootless, with the `vfs` storage driver and `chroot` isolation, under a node-local storage
root (`/raid/scratch/<user>/nemo-gym-buildah` when that disk exists, else the system temp folder). Without
a repository the image stays local as `hb__<hash>`, and only `docker` can build it: the Docker
daemon is the one local store a sandbox provider starts images from, so buildah and podman are refused
without a repository. That is the laptop path with the Docker provider.

buildah and podman run with a `containers.conf` override. The host's containers configuration may give every
container a default bind mount (this deployment's `/etc/containers/containers.conf.d/30-mounts.conf` mounts
`/etc/localtime` read-only into each one), and both tools apply those mounts to build steps too, so any
Dockerfile that installs `tzdata` (Ubuntu with `python3` pulls it in) fails with
`mv: cannot move '/etc/localtime.dpkg-new' to '/etc/localtime': Device or resource busy`. The builder writes
`containers.conf` with `[containers]` `volumes = []` under its storage root, once, and sets
`CONTAINERS_CONF_OVERRIDE` to it for every buildah and podman command, the layer containers-common applies over
the host's files. A `CONTAINERS_CONF_OVERRIDE` already set in the environment is left as it is, so a deployment
that needs its own override keeps it.

podman as the build tool needs a working rootless setup on the build host. On the cluster nodes used here, podman's
default `crun` runtime cannot create rootless build containers and `chroot` isolation breaks Dockerfile heredocs, while
`BUILDAH_ISOLATION=rootless` with the `runc` runtime (named in a `containers.conf` `[engine]` block) builds them. Point
`CONTAINERS_CONF_OVERRIDE` at such a file before `gym eval run` when the host's defaults do not work; the builder leaves
a preset override alone. Build contexts must be on local disk for heredoc builds: buildah's heredoc support overlay-mounts
the context, which fails on Lustre.

buildah and podman also run with a `registries.conf` of the builder's and build docker-format images. An
unqualified `FROM` (`FROM oven/bun:1`) resolves to docker.io under `docker compose build`; buildah and podman
resolve it through the host's `registries.conf`, and the stock Debian and Ubuntu file names no
unqualified-search registry, so the build fails with `short-name "oven/bun:1" did not resolve to an alias`. The
builder writes `registries.conf` with `unqualified-search-registries = ["docker.io"]` under its storage root and
sets `CONTAINERS_REGISTRIES_CONF` to it for every buildah and podman command. That file replaces the host's main
`registries.conf` (the drop-ins under `registries.conf.d`, short-name aliases included, still load), so a site
whose main file carries mirrors or blocked registries sets `CONTAINERS_REGISTRIES_CONF` itself and the builder
leaves it alone. The build runs with `--format docker`, the image format BuildKit builds: under the tools'
default OCI format `SHELL` is ignored by the `RUN` steps (`RUN [[ ... ]]` after `SHELL ["/bin/bash", "-c"]`
fails with `/bin/sh: [[: not found`) and `HEALTHCHECK` and `ONBUILD` are dropped from the image. The recorded
configuration in `compose-images.json` carries `Healthcheck` from the image under all three tools, and `Shell`
and `OnBuild` with buildah and docker (podman's inspect does not report them, although its image carries them).
docker needs neither setting.

Dockerfile heredocs (`RUN <<EOF`, `COPY <<EOF /file`, `RUN python - <<'PY'`) need a buildah or podman built with
openshift/imagebuilder 1.2.6 or later (upstream buildah 1.33.0 or later). Ubuntu 24.04's packaged buildah 1.33.7
and podman 4.9.3 are linked against imagebuilder 1.2.5 and read the heredoc body as instructions
(`Unknown instruction: "ECHO"`). Before the first Dockerfile with a heredoc is built, the builder runs a
preflight once: a two-line `FROM scratch` plus `COPY <<EOF /probe` build in a temporary context, no base image
pulled. When it fails, prepare stops with one error naming the tool and its version, the Dockerfile and the
remedy (a qualifying buildah or podman named with `builds.tool`, or `builds.tool: docker`) instead of one
`Unknown instruction` failure per task. A heredoc opened with a space before the delimiter
(`RUN cat > /x << 'EOF'`) is not a heredoc to BuildKit either and fails under every tool.

With buildah and podman, `main`'s `build` section of the task's `environment/docker-compose.yaml` applies to the
direct build the way `docker compose build` applies it over Harbor's template: `context` (a path or the string
form of the section) and `dockerfile` are resolved against `environment/` and have to stay under it, `args`
(mapping or `KEY=value` list; a bare `KEY` takes the variable's value from the build's environment) become
`--build-arg`, and `target` becomes `--target`; a mapping value of `null` (`KEY:`) is read from the build's
environment like a bare list `KEY`, and both are dropped when it is unset, so the Dockerfile's own `ARG` default
stands, as under Compose. The section is interpolated as Compose interpolates it, from the build's environment
(the process environment, the task's `[environment].env` and Harbor's variables, with `environment/.env` under
them), so `${VAR:?message}` with no value fails that task's build naming the task and the variable instead of
passing an empty build arg, while a bare `${VAR}` nobody sets is an empty string with one warning, as Compose
makes it (at prepare; the seed resolves the same name strictly, see "Compose template variables and `env_file`").
The task's Compose file is read at build time, not at planning, and checked on this path as on the
docker path before anything is built, so a file the loader would refuse is never built and pushed under the
`hb__` name, and a file one task gets wrong leaves that task out without touching the rest of the dataset. Each
service with a `build:` section is one `buildah bud` or `podman build` of its own context and Dockerfile, `main`
first and the sidecars in file order, each inspected, pushed and recorded as it finishes, so an interrupted
prepare resumes at the next service. `dockerfile_inline` and `additional_contexts` are refused on the direct
path for any service, before the context's first build (the checked file already says so, so nothing is built or
pushed for a context that is then refused), with `builds.tool: docker` as the remedy; `no_cache` and `pull` are
ignored there. The
image name and hash do not change: the hash is Harbor's hash of `environment/`, which already covers the Compose
file.

A proxy the build tool inherits can break the registry push. The tool reads `http_proxy`, `https_proxy`,
`no_proxy` and their upper-case forms from its environment and from its own configuration
(`containers.conf` `[engine].env`, `~/.docker/config.json` `proxies`, the Docker daemon), and a proxy that
answers 403 to the registry ping fails the push even when the gym process itself has no proxy variables.
`no_proxy` (and `NO_PROXY`) has to include the registry host, or the proxy variables have to be cleared for the
build. Nothing is hard-coded: when a build, push or ping fails with a 403, the builder logs the names of the
proxy variables present in the tool's environment (names only, never values), or says that there are none and
that the tool's own configuration is the place to look, so the cause is visible in the failure.

With `docker`, the build is Harbor's build step. Prepare writes the Compose file Harbor's Docker environment
builds a task with (service `main`, `build.context` the task's `environment/` folder, Harbor's
`pull_policy: build` and keep-alive `command`), adds the task's own `environment/docker-compose.yaml` after it
when the task ships one, and runs
`docker compose --project-directory <environment/> -f <harbor file> [-f <task file>] -f <images file> build`, as
`DockerEnvironment.start()` does (`harbor/environments/docker/docker.py` at v0.22.0, cited by permalink in
`nemo_gym/tasks/harbor/image_build.py`), with one generated override of Gym's own passed last, the way Harbor
passes its own generated overrides after the task's file: `docker-compose-images.yaml`, the `image` of every
service built (`hb__<hash>` for `main`, `hb__<hash>-<service>` per sidecar), so Gym's names win over a sidecar's
own `image:` without a `docker tag` step. One compose build builds every service with a `build:` section, as
Harbor's does; when some of a context's services are already recorded, or `main` is not built, the build names
the ones it builds. The command's environment is merged as Harbor merges it: the process
environment, then the task's `[environment].env` (a whole value `${VAR}` or `${VAR:-default}` read from the
host, as Harbor's `resolve_env_vars` reads it; an unset `${VAR}` with no default fails the build), then Harbor's
infrastructure variables, which win: `MAIN_IMAGE_NAME`, `CONTEXT_DIR`, `CPUS` and `MEMORY` (`<n>M`) from the
task's resources, the `HOST_*_PATH` / `ENV_*_PATH` log mounts, and the two the loader defines for the same file,
`TEST_DIR` and `COMPOSE_PROJECT_NAME` (the table below). So `main`'s build args and target in the
task file, interpolated or literal, apply as they do in Harbor. Two deliberate additions over Harbor's literal
file and command: Harbor's template sets no `image` (its build is tagged `<session>-main` unless the task file
says `image: ${MAIN_IMAGE_NAME}`), and Gym sets `image: hb__<hash>`, the `MAIN_IMAGE_NAME` value, so the result
can be inspected, pushed and recorded; and `DOCKER_DEFAULT_PLATFORM` carries `builds.platform`. `main`'s
`image`, when the task file sets it, has to resolve to `MAIN_IMAGE_NAME` (`${MAIN_IMAGE_NAME}`,
`${MAIN_IMAGE_NAME:-…}` or `${MAIN_IMAGE_NAME-…}`) when `main` is built; a literal other image is refused. A
sidecar-only build (a prebuilt, pull-mode or overlay `main`, or one already recorded) does not read or tag `main`'s
image, so its spelling is not checked, as Harbor's prebuilt mode never reads it.

The task's Compose file is third-party content that `docker compose build` runs on the build host, which often
holds the push credentials, so prepare checks it before the command runs. The check reads the file as Compose
reads it: interpolated against the build's own environment (the merged environment above, with
`environment/.env` under it, as Compose reads `.env` from the project directory), with Compose's `${VAR}`,
`${VAR:-default}`, `${VAR-default}`, `${VAR:?error}`, `${VAR:+alt}` and `$VAR` forms, so `network: ${X:-host}` is
refused like `network: host` and `cache_to: [type=gha,${X:-token}=abc]` like a literal token. `.env` values are
read as Compose's dotenv reader reads them: an unquoted or double-quoted value is itself interpolated against
the build's environment and the lines above it (so `NET=${X:-host}` in `.env` is `host`), a single-quoted value
is literal, and `COMPOSE_ENV_FILES` set in the build's environment fails the build, as it would point Compose at
files the check does not read. A bare `${VAR}` nobody sets is an empty string with one warning naming the task
and the variable, as `docker compose build` makes it; `${VAR:?message}` fails the build with the message. The
loader resolves the same file at seed with the deployment's `compose_variables` in hand, and a name the Compose
file (or an overlay file) still leaves unset then fails the episode terminally before anything is provisioned (see
[Supported task behavior](#supported-task-behavior)), so the file checked is the file built, or the episode is
refused rather than started with an empty value. `environment/.env` is the exception: its values are resolved with
Compose's empty substitution at prepare and seed alike, before the deployment's values are merged, so a `.env` line
that reads an unset name is empty at seed too and never sees `compose_variables`. The build and the loader
read the same sources (the process environment, the task's `[environment].env`, Harbor's variables and
`environment/.env`), so prepare and the environment server have to run with the same variables exported for the
file built to be the file started. Prepare refuses, naming the service and key: build `secrets`, `ssh`, `extra_hosts`, `entitlements` and `privileged` (BuildKit's
`--allow network.host` / `security.insecure`, which let a `RUN --network=host` line use the build host's network), `network: host`, `additional_contexts` other than `service:`
references, `cache_from` / `cache_to` entries carrying credentials, a `build.context` or `dockerfile` that does
not resolve under `environment/` (symlinks resolved; `${CONTEXT_DIR}` is allowed), a top-level `include` and an
`extends` with a `file`. The per-section checks run on every service with a `build:` section, sidecars included;
a sidecar's `image:` is accepted whatever it says, since the build is tagged with Gym's name for it and the seed
document is filled from the record by service name. Harbor
reads a task's Compose file from `environment/docker-compose.yaml` only; the loader also accepts
`docker-compose.yml`, `compose.yaml` and `compose.yml` at run time, and the build refuses those spellings so the
file a task builds with is the file it runs with. `docker build` is never called, so the Compose v2 plugin
(`docker-compose-plugin`) is required and its absence fails prepare with a message that names it. buildah and
podman are the OpenSandbox path: they build the Dockerfile directly and push it.

Images are built for `linux/amd64`, the platform the sandboxes run. The tool is asked for that platform
(`docker compose` through `DOCKER_DEFAULT_PLATFORM`), and an image that comes out for another one is
refused before it is pushed or recorded. Building on an
arm64 host therefore needs qemu with binfmt registered, or an x86 node. `builds.platform: OS/ARCH` overrides
the target; an amd64 micro-architecture level above the baseline (`linux/amd64/v3`) is refused, since images
are recorded by os/arch only and a later `linux/amd64` run would reuse what it built. `builds.enabled: false`
never builds. Tasks that need a build and have none recorded, the separate
verifier's `tests/Dockerfile` included, are then skipped with a message that says so.

The hash is SHA-256 over the sorted relative paths and bytes of the regular files in the build context,
exactly as Harbor computes it: symbolic links and anything under `__pycache__`, `.git` or `.DS_Store` are
skipped, and file modes are not part of it. Two tasks with identical `environment/` folders share one
image and a changed file gets a new one. `built-images.json` next to the task folders maps each context
hash to the images built from it, by service, with the tool that built each and whether it was pushed, and the
tasks that use the context:

```json
"<hash>": {
  "image": "registry/path/hb__<hash>@sha256:...",
  "tasks": ["cb100-015-ember-medical"],
  "tool": "buildah",
  "pushed": true,
  "services": {
    "main":  {"image": "registry/path/hb__<hash>@sha256:...", "tool": "buildah", "pushed": true},
    "world": {"image": "registry/path/hb__<hash>-world@sha256:...", "tool": "buildah", "pushed": true}
  }
}
```

`image`, `tool` and `pushed` at the top level are `main`'s, kept there for older readers; `image` is absent when
`main` was not built (a pull-mode `main` with a built sidecar), and the two flags are then the first recorded
service's. The flags are kept per service because one context's services may be built by differently configured
runs (a sidecar built locally after `main` was pushed, or the reverse), and each service is judged by its own. An
entry with `image` and no `services` was written before sidecars were
built and reads as `main` alone. Deleting a context's entry
from that file forces a rebuild of it at the next prepare; there is no flag for that, since a changed
Dockerfile or context is a new hash and builds on its own. Each built image's
configuration lands in `compose-images.json` beside it, so the loader reads the Dockerfile's final
`WORKDIR`, `ENV` and `USER` from it like any pulled image. A recorded image is reused when it fits the run:
a pushed image serves every run, a local one only runs that push nothing, and the platform has to match, each
service on its own record. Otherwise it is rebuilt and the record replaced, and prepare says why; a service
without a record is built and the context's other services are left as recorded. An image is recorded only after
it was built, pushed and inspected, and each one as soon as it is done, so an interrupted prepare resumes
where it stopped. Both record files are replaced atomically under a sidecar lock file next to them
(`built-images.json.lock`, `compose-images.json.lock`), so two Gym commands recording into one datasets
folder keep each other's entries and a reader never sees a half-written file. A failed build leaves its tasks
out with the reason and the others prepare. Prepare also writes every
distinct image a run of the prepared tasks starts to `results/harbor/<taskset>/images.json`, for pre-pulling: the
task images (built or pulled), the separate verifiers' images (declared, or built from `tests/Dockerfile`; a step's
own separate verifier image included) and the Compose sidecar images as their records pin them (built or from a
registry); the records themselves sit in
`compose-images.json` (and `built-images.json` for the built ones) next to the task folders.

The hash names the image; it is not the whole build. The configuration record of each built image in
`compose-images.json` also carries an `inputs_digest` over everything its build read, beside the
`config_digest` of what came out: every entry of the context tree with its mode (files by content, symbolic
links by target, folders by presence, the names Harbor's hash skips included) and the service's `build` section
of the task's Compose file as the build reads it (the whole section after interpolation, an inline Dockerfile and
labels included, its paths as resolved and its build arguments as the tool is given them), so the images of one
context share the tree and differ in the section. A service that builds from another service's image
(`additional_contexts: {base: service:db}` with `FROM base`, which `docker compose build` honours) also carries
`db`'s inputs digest, transitively, so a change to `db`'s inputs rebuilds `db` and `main` both, as Compose would
build a different `main`. A recorded image is reused only when the context's hash and
its recorded inputs digest both match; a service whose inputs changed without changing the context's hash (a
mode flip, a new empty folder, a different link target, a variable the file interpolates into the build section
or a build argument taken from the environment) is rebuilt under the same name and its records replaced, the
context's other services left as recorded, and prepare says why. With `builds.enabled: false` such a task is
left out with the reason. An image recorded before the digest was kept is rebuilt once.

The build tool pushes with its own login. Pulling a pushed image into a sandbox uses the registry
credentials of the sandbox provider block (`images.auth`), the same way as any private image. The sandbox
is created with the Dockerfile's final `USER` (or `[agent].user`) as the default for every command, on every
provider, not only for the overlay `RUN` lines. An overlay task starts without that default, because its
`RUN` lines may be what creates the user, and switches to it once they have run. A numeric `USER 1000` is
resolved to its name in the image with `getent` first, because OpenSandbox and PTY sessions switch user by
name. When the uid names no user, the number stays and the seed logs a warning: Docker accepts a uid as
it is, the other providers cannot run commands as it.

| Setting or feature | Behavior in this layer |
| --- | --- |
| Image | Prebuilt `environment.docker_image`; a pull-mode Dockerfile (`FROM` plus `WORKDIR`, `ENV`, `USER`, `LABEL`), started from the base image; an overlay-mode Dockerfile (the same plus `RUN` lines, run in the pulled base image at seed, no build); any other Dockerfile (`COPY`, `ADD`, heredocs, BuildKit flags, multi-stage) is built at prepare with buildah, podman or docker as `hb__<hash>` and the task runs from the built image, Compose sidecars with `build:` as `hb__<hash>-<service>`; with `builds.enabled: false` or no record such a task is skipped at prepare. The separate verifier's `tests/Dockerfile` is built the same way as `hb__<tests hash>-verifier`, whatever it holds (see "Dockerfile builds"). |
| Verification | Shared or separate verification, per step for a multi-step task (see Multi-step tasks). A separate verifier grades in `[verifier.environment].docker_image` when the task declares one, else in the task's own `[environment].docker_image` (Harbor's effective verifier environment is a copy of `[environment]`), else in the image prepare built from `tests/Dockerfile`, for the episode-sized sandbox lifetime; a declared or inherited image is never rebuilt from `tests/Dockerfile`, as Harbor's `should_use_prebuilt_docker_image` decides. A separate-mode task with neither is refused as Harbor refuses it; one whose `tests/Dockerfile` has no recorded build is left out at prepare and refused at seed. A `tests/docker-compose.yaml` may declare `main` only. |
| Users | The image's default user and `verifier.user` are supported. Explicit `agent.user` and Dockerfile `USER` overrides are rejected at seed because sandbox access cannot yet carry the agent user. This layer has no oracle-specific exemption. |
| Environment variables | Whole-value `${VAR}` and `${VAR:-default}` templates in environment and verifier variables resolve from the server's host environment at use time. Missing required values fail the episode before provisioning. Resolved values are not added to dataset rows, task snapshots, or the episode result. A task can still print its own environment into verifier logs. |
| Prebuilt-image input files | `environment/` contents are uploaded to the workdir when the task has a prebuilt image and no `Dockerfile` or Compose definition. File modes are preserved. |
| Network restrictions | `[verifier].network_mode` is enforced for separate-mode tasks on OpenSandbox: `no-network`, or `allowlist` with `allowed_hosts` of host names, `*.domain` wildcards and bare IP literals, spelled as Harbor's `normalize_allowed_hosts` admits them (no brackets or zone ids on an IPv6 literal, no wildcard over an IP literal). The separate verifier's sandbox is created under that egress policy (deny by default, the allowed hosts as allow rules in order), which replaces the deployment's `sandbox_provider_options.network_policy` for that sandbox as Harbor's explicit phase policy replaces the environment baseline, except that it may not widen it: under a deny-by-default deployment policy (`defaultAction: deny`, or none named, which OpenSandbox reads as deny), an `allowlist` task is refused at seed naming its hosts, whatever the deployment's own rules allow (there is no host matching; such a task needs an allow-by-default deployment), and `no-network` passes; the result's `verifier_network` reports the mode. The window is wider than Harbor's: Harbor applies the `[verifier]` policy around `verifier.verify()` only and starts the separate verifier environment under its baseline, whereas here the verifier sandbox lives its whole life under the policy, so the verifier image's `ENTRYPOINT` and `HEALTHCHECK` (see the Image ENTRYPOINT row) and the artifact staging run restricted too; a verifier image that needs the network when it starts passes under Harbor's `no-network` but masks as `harbor:entrypoint_failed` here, so such a task must name those hosts in `allowed_hosts`. OpenSandbox cannot switch a running sandbox from public to restricted (its egress API changes rules, never the default action; see the OpenSandbox provider page), so Harbor's `_phase_network_policy` switch has no equivalent in shared mode: the loader refuses a `[verifier]` restriction on a shared-mode task, naming the fix, and `[environment]`, `[agent]`, `[verifier.environment]` and step restrictions stay refused rather than silently ignored. A provider other than OpenSandbox, or an `allowed_hosts` entry the policy cannot express or Harbor would refuse (a CIDR range, a bracketed or zoned IPv6 literal, a `*.` wildcard over an IP literal), is refused at seed, before the rollout. |
| Artifacts and hooks | Verifier logs are downloaded. `[[verifier.collect]]` hooks run in both modes, each in the service it names (a non-zero exit or a hook that runs past its `timeout_sec` warns, a hook the sandbox cannot run masks). In separate mode `/logs/artifacts` and the `artifacts` entries are copied into the verifier, each under its own staging root, with their `exclude` rules applied as Harbor passes them to `tar --exclude` (unanchored: `node_modules` at any depth), replacing what the verifier image ships at those paths. The verifier receives each artifact as the archive the agent's sandbox packed, so read-only, world-writable and set-id modes arrive as the agent left them; the host copy under the staging root is unpacked through a filter (widened owner modes, links kept, nothing that escapes and no device nodes) and is the gate the archive must pass plus a diagnostic, and the archive is removed once restored. A probe the sandbox cannot answer masks rather than counting as an absent artifact. |
| Image ENTRYPOINT | Runs for a plain (non-Compose) task whose image configuration is recorded: a built image, or a pull or overlay Dockerfile whose base image is recorded, unless the Dockerfile says `ENTRYPOINT []`. The sandbox starts with the provider's keepalive; the seed then starts `ENTRYPOINT + sh -c "sleep infinity"` beside it (what Harbor's Compose `command` leaves it with) as the image's `User` in its `WorkingDir` (created for a resolved Dockerfile when the task's workdir is another), after the working directory and `/logs` exist. It then waits as `up --wait` and Harbor's `run_healthcheck()` do: the image `HEALTHCHECK` first, then `[environment.healthcheck]`, both as the image's user (the image check in its `WorkingDir`, the task check in the task workdir; a `CMD-SHELL` check runs through the image's recorded `Shell`, `/bin/sh -c` when it sets none, as the daemon runs it, so `SHELL ["/bin/bash", "-o", "pipefail", "-c"]` fails `false | true` here too), probing the entrypoint's exit status before every attempt and once more before the hand-over. Exit non-zero before the hand-over fails the seed as a task error (terminal, with the output tail), as does a launcher whose own shell fails for the image's user or directory; a provider that cannot run the launcher is retried. Exit zero logs a warning once and the episode continues on the keepalive. At verification the status is read again: in the separate verifier's fresh sandbox a non-zero exit masks the sample as `harbor:entrypoint_failed` (under Harbor the container would have died and the trial errored); in the agent's sandbox it masks only when the exec holding the entrypoint returned on its own with that status, because the runtime folder is world-writable and the agent had the sandbox; otherwise it is recorded as `entrypoint-exit.txt` with an error in the log and the sample is scored. The holding exec lives as long as the provider allows (docker caps it at `exec.default_timeout_s`, OpenSandbox without `operations.background_exec` at `connection.request_timeout_s`); past that the entrypoint is untracked, logged once, and OpenSandbox deployments need `background_exec` for long-lived launchers. An entrypoint that is one `sleep` or `tail` command is not started. A separate verifier that runs in the task image gets that image's entrypoint and `HEALTHCHECK` because Harbor's effective verifier environment is then a copy of `[environment]` (its prebuilt verifier), never the task's healthcheck. A declared `[verifier.environment].docker_image` starts the same way when `compose-images.json` records it, which it does when that image is the base of a task's Dockerfile (this task's own, in pull or overlay mode, or another task's): its own `ENTRYPOINT` as its own `User` in its own `WorkingDir`, its own `HEALTHCHECK`, none of the Dockerfile's `RUN` lines, `USER` or `WORKDIR`, as Harbor's `up --wait` of the declared image leaves it. The separate verifier's sandbox is a plain one whatever the agent's environment: under a Compose task too, the verifier image's entrypoint runs there and its `HEALTHCHECK` gates `test.sh`, as Harbor's `up --wait` of the verifier image does. The output is kept as `entrypoint-stdout.txt` (and `verifier-entrypoint-stdout.txt`) next to the verifier logs. A prebuilt `docker_image` has no recorded configuration yet, so its entrypoint still does not run. |
| Other features | Windows and TPU environments are unsupported. Multi-step tasks run as turns (see above). A task that ships Harbor's seed trajectory (`trajectory.json` beside `instruction.md`, or `steps/<first step>/trajectory.json` for a multi-step task: the prior context Harbor 0.22.0 loads into the agent's session before its first turn, as `--load-trajectory` does, for every agent but the oracle and nop) loads for the oracle alone: no harness here can be seeded with it yet, so `gym dataset validate` runs the task as Harbor's oracle does (which skips the file too) and the environment server refuses it at seed, terminally, for any other agent, naming the file; the loader warns about it, and the file stays in the task's digest and package hash, so a fetched task still verifies. |

Upgrading from the pre-build layer (#4566): before Dockerfile builds, a separate verifier with no declared
image always ran in the task image (an overlay-mode task's `RUN` lines applied in the verifier's sandbox too,
masking as `harbor:overlay_failed` when they could not be), and a task shipping `tests/Dockerfile` was rejected at
seed. Now the task image is used only when Harbor's inherited verifier environment carries
`[environment].docker_image`; the verifier-side overlay and its mask are gone for a task that declares separate
mode (`separate_verifier_for_artifacts`, below, keeps both for the one shape it creates: a shared-mode task with an
overlay-mode Dockerfile forced into a fresh sandbox of the task image gets the agent's `RUN` lines there, within the
episode budget, and masks as `harbor:overlay_failed` when they cannot be applied), so a separate-mode task with an
overlay-mode Dockerfile grades in the image built from its `tests/Dockerfile` (every separate-mode hub task the
15 Oct scan labelled overlay ships one: 20 in tb3-preview-v2, tb4-preview and terminal-bench-4-no-prebuild, and
terminal-bench-science's two, which a `[verifier.environment]` network restriction refuses at load before that) and
is refused without one. A `tests/docker-compose.yaml` with services beside `main`, with a runtime key on `main`
(`environment`, `env_file`, `network_mode`, ...) or with a top-level `networks`, `volumes` or `include` is refused
at load (arc-crm-6's six separate-verifier tasks; see "Dockerfile builds" above), and a `tests/Dockerfile` verifier needs a
record in `built-images.json`, so a dataset prepared before this layer has to be prepared again (`gym dataset
fetch` or `gym dataset validate`) on a host with buildah, podman or docker installed. With `builds.enabled: false`
the tasks that need a verifier build are left out with the reason, so an existing dataset's eligible task set can
grow (verifiers now built from `tests/Dockerfile`) or shrink (shapes now refused) between the two layers.

Shared-mode verification does not stop or quiesce what the agent left running: `test.sh` runs in the same
sandbox beside those processes, as Harbor's shared mode runs it, so a test that talks to a service the agent
started keeps working, and a process the agent left behind can still interfere. Separate verifier mode is the
tamper-proof path: the verifier gets its own sandbox and its own image, and nothing the agent started is there
(see the Verification row for what this layer supports).

Emptying `/tests` and `/logs/verifier` before verification is a deliberate difference from Harbor's single-step
shared mode, which merges tests and reads whatever `reward.json` the folder holds. It prevents agent-planted reward
files, test reports (an all-pass `results.xml` left behind by a solution that exits during pytest's import, which a
`test.sh` that reads the report would score as a pass) or test helpers from being used. Only the inputs the task
declares survive, as regular files at those names: a sidecar that binds the path writes the verifier's input there
(kumo's `verifier` service appends `actions.jsonl` for `main`'s `tests/score.py`), which emptying the folder would
delete, so the dataset names such files in its `dataset.toml` (`[gym.tasks."<task_id>".verifier] inputs =
["actions.jsonl"]`, a Gym extension Harbor has no field for; a path, a grader output name or a repeated name is
refused at load). They survive the reset before every turn of a multi-step task too. Restored artifacts in separate mode are held to the same line: an artifact at `/tests`,
inside or over it, or at or inside `/logs/verifier` is refused when the task is loaded (its source is settled
to one leading slash first), collected on the host but never restored should it reach the server, one that a
link in the verifier sandbox would divert onto either directory is refused once its target is resolved
there, and a restore that covers `/logs/verifier` (`/logs`) is followed by that directory being emptied
again before `test.sh` runs. Linked verification roots (including `/logs`) are unlinked before preparation,
without deleting their former targets. Helpers baked into `/tests` in an image are also removed; ship required helpers in
the task's `tests/` directory instead. Preparation and transfers attempt root, with a fallback to the
default image user when switching identity is unavailable. Target directory ownership and modes are
preserved during extraction. Uploaded test files stay root-owned with the package's modes, as Harbor's
`docker cp` leaves them, and only `test.sh` gets `chmod +x` as root before it runs as the verifier user
(Harbor's `needs_chmod` rule for a `.sh` script), nothing is chowned or widened, so a package built under
`umask 077` behaves as in Harbor: its `0600` files are readable by a root verifier and by no other
`[verifier].user`.

In separate mode a `/tests/test.sh` baked into the verifier image wins, as in Harbor, which never uploads the
package's tests into a separate verifier (`skip_tests_upload`): the package's `tests/` is uploaded only when that
file is absent. The check is on `test.sh` alone, not on the directory. An image that ships a `/tests` folder without
a `test.sh` gets the package tree extracted into it, same-named files replaced and the rest merged in, so only a
baked `test.sh` is protected. To grade a task's own image with the package's tests, drop the baked `test.sh` from
that image, or ship a `tests/Dockerfile` so the verifier gets an image of its own (again, see the Verification row
for what this layer supports).

## Compose tasks on OpenSandbox

A private image (a Compose sidecar on a private registry, a private base image) is recorded at prepare with the
credentials the sandbox block's `images.auth` holds, the same ones the provider pulls it with at seed: prepare
reads `sandbox.<provider>.images.auth` (image prefix to `{username, password}`, resolved so a `${...}`
interpolates from `env.yaml`), keys it by registry host, and records with it, so a private image no longer stops
`gym dataset validate` or `gym eval run` at prepare with an anonymous `HTTP 403`. No `images.auth` keeps the
anonymous reads. Add no new key for this; it is the one place a deployment names registry credentials.

The shipped `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml` (what `--sandbox opensandbox`
loads) has no networking block, and the Compose adapter refuses to start a group without one. Put an overlay
like this one after it in `--config` (or in a `+config_paths` entry). The three blocks are the deployment's:
service-to-service networking, the label the cluster reads a service's `shm_size` from, and the capability
labels and probes a task that needs `SYS_PTRACE` is scheduled with. The optional second provider block shows
how `gpu_sandbox_provider` routes the tasks that declare GPUs to a GPU pool while the rest stay on `sandbox`.

```yaml
sandbox:
  opensandbox:
    networking:
      enabled: true
      loopback_forwarding: true
      # Runs in every container the provider creates, sidecars included: it must no-op where it does not apply.
      setup_command: "command -v python3 >/dev/null 2>&1 || ! command -v apt-get >/dev/null 2>&1 || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3)"
    runtime_requirements:
      shm_size_metadata_key: nemo.nvidia.com/shm
      capability_metadata:
        SYS_PTRACE:
          nemo.nvidia.com/ptrace: "true"
      capability_probes:
        SYS_PTRACE: "awk '/^CapEff:/ { print $2 }' /proc/self/status | { read caps; test $((0x$caps & (1 << 19))) -ne 0; }"

# Optional: a second provider block for GPU tasks, named by the server's `gpu_sandbox_provider`.
sandbox_gpu:
  opensandbox: ${sandbox.opensandbox}

harbor_environment_server:
  environment_servers:
    harbor:
      gpu_sandbox_provider: sandbox_gpu
```

## Configuration and artifacts

See [`configs/harbor.yaml`](configs/harbor.yaml) for this layer's supported settings. Verifier logs default
to `${results_dir}/harbor/verifier/<generated-directory>`, using Gym's absolute results root. An explicit
relative `artifacts_dir` is relative to that root; an absolute path is used directly. The server reserves
a fresh directory for each verification; episode identifiers are never used as host paths.
New directories honor the server process's umask and inherited directory permissions, including group
access for shared results. Local log-directory creation or containment failures exclude the sample
with `failure_kind: persistence_failed` and identify the affected path.
Measured outcomes include an absolute `verifier_logs_dir` naming the collection location; that directory
can be empty if optional log collection fails.

Every run through this server (`gym dataset validate` and `gym eval run <task target>`) also leaves one
per-run summary beside its rollouts, `<rollouts stem>_run_summary.json` (`rollouts_run_summary.json` by
default; `nemo_gym.tasks.harbor.run_summary`), so a results folder explains itself: the command, target,
taskset and agent; the dataset's folder, source and digests; the `task_source` the run config declares for the
taskset (`harbor_<taskset>_agent`, known before dispatch) and, as a cross-check, the one the result rows carry
(`rows_task_source`, `null` without rows or when they disagree); the files of the run under `files` (rollouts,
failures, materialized inputs, aggregate metrics, the health check's `quality_summary.json` and
`rollout_verdicts.jsonl`; the rows with the data processor's `_prepare.jsonl` and `_metrics.json` beside them;
the collator's `<rollouts stem>/preprocessed_datasets/` folder with the `<split>.jsonl` the collector
dispatched and its `_metrics.json`; run config, `skipped.json`, `images.json`, the verifier root), each `null`
when the run did not write it (the aggregate metrics and the two health reports follow a completed collection,
so they are listed only when at least as new as the rollouts file: a run that stops early leaves an earlier
run's, and the health reports keep one name per folder whatever the `--output` stem; a fresh start, the
default `resume_from_cache=false`, clears the rollouts, failures and materialized inputs only once its servers
are up, so a file under those names that is unchanged since before dispatch, as after servers that failed to
start, is an earlier run's too, its rows are not this run's and its tasks are `missing`); the files found under
the run's names that are not its own are listed under `stale_files`, so a `null` beside a file by that name is
explained; and, per selected task
and repeat, its status
(`result`, `failed` or `missing`), reward, mask and `failure_kind`, the verifier's mode, wall time, network
and return code, the paths of its verifier log folder, `test-stdout.txt`, `trajectory.json`,
`entrypoint-stdout.txt` and collected `artifacts/` (per turn for a multi-step task), and the input the row
carried: its task and dataset digests and the `responses_create_params` metadata (`agent_timeout_sec`,
`harbor_step`). Every path is relative to the summary's own folder, so a results folder moved as a whole
still resolves (`--output` outside it gives `../` paths); a folder a row names that is not on disk is listed
under that task's `missing_paths`, never dropped. The summary is written whatever the run did: after
collection, and also when `gym dataset validate` dispatches nothing because no selected task has a
`solution/` (then with no run config and no tasks, beside the `--output` the command asked for). A summary
that cannot be written is logged without
touching the run's outcome; a fresh `gym eval run` reads its selected rows for the summary only after the
collector has run, inside that same guard. No judge runs on this path outside the task's own `test.sh`: a
task that calls one inside the sandbox leaves whatever the judge wrote under `/logs/verifier`, which is the
verifier log folder the summary names.

`gym -v` / `--verbose` (the `verbose` config key, forwarded to every spun-up server) raises the log level to
DEBUG in the CLI, this server and the oracle agent alike. At that level the server traces each episode
(seeded, `test.sh` exit code, wall time, rewards and log folder, final score) and the oracle its `solve.sh`
run (budget, the task user or `the sandbox default user` when the task names none, exit code); without it
both stay quiet below WARNING.

The task's `[agent].timeout_sec` reaches the harness twice: in the agent-session seed as
`agent_timeout_sec`, next to `sandbox_access` (which also names the task's `[agent].user` as `user`, and
for the oracle the resolved `[solution].env` as `env`; the oracle's solve budget is that `agent_timeout_sec`),
and, until every harness reads the seed, as the materialized `agent_timeout_sec` metadata that Hermes
honors today, using the smaller of that deadline and its configured runner limit. The verifier uses
`[verifier].timeout_sec`.

Every episode runs under its task's own time budget (`enforce_task_timeouts: true`). The deadline starts
when the sandbox is ready, after the image pull, its retries and the overlay, and covers
`harness_setup_allowance_s`, `[agent].timeout_sec`, `[verifier].timeout_sec` and `verifier_grace_s`, with a
second sandbox start for a separate verifier; the seed has its own bounds (`sandbox_ready_timeout_s` per
start, `image_pull_retries`) and is never charged against it. An episode that overruns it fails with
`failure_kind: harbor:task_deadline` (retryable), naming the stage it was in. The whole task budget (one
sandbox start, plus `[environment].build_timeout_sec` for an overlay, plus the deadline) never exceeds
`default_episode_timeout_seconds`, which stays the run-wide cap. Sandbox lifetime covers that budget, the
final cleanup budget (`cleanup_timeout_seconds`) and `sandbox_ttl_slack_s`, so setup and queued agent work
also fit; both budgets must be finite. Set `++<instance>.environment_servers.harbor.enforce_task_timeouts=false`
to run with the run-wide deadline alone. Sandboxes are normally stopped as soon as the episode ends.

Before `test.sh` runs, in either mode, the server writes the agent transcript to `/logs/agent/trajectory.json`
in ATIF v1.7 (the version Harbor 0.22 and 0.23 read), exported from the Gym transcript by
`nemo_gym.atif_export.transcript_to_atif`. Harnesses that feed terminal output back as user turns (Terminus)
export with those turns as the previous step's observation. Shapes a model causes still export and are
graded: a turn cut off inside its reasoning (a reasoning item no answer follows) becomes an agent step with
an empty message and `extra.nemo_gym.incomplete: max_output_tokens`, several reasoning segments join with
newlines, empty or encrypted reasoning exports without `reasoning_content`, tool arguments that are not
a JSON object become `{}` with the original string under the call's `extra.nemo_gym.raw_arguments`, and a
refusal part exports as the step's text, flagged `extra.nemo_gym.refusal` with reason `refusal` (Harbor's
Terminus records an empty step there and grades on; the text is kept here); the trajectory then says
`conversion.status: partial` with the `reasons`. A transcript the exporter cannot
represent at all (a copied assistant turn, a later system turn, a tool result for no call or a second
`function_call_output` for the same `call_id`, item types ATIF has no place for) masks the sample with
`failure_kind: harbor:trajectory_export_failed`, so a converter defect can never appear as a reward; a copy
of the file sits in the verification's log directory.

Operator secrets (judge API keys and the like) go in the server's `secrets` block, by environment variable
name. They fill `${NAME}` templates in a task's `[verifier].env`, which is the whole `test.sh` process
environment, and reach nothing else: a secret no template names is passed nowhere (Harbor's verifier
resolves its env the same way), and a `${NAME}` template in `[environment].env` or
`[verifier.environment].env` still resolves from the server's own environment only. Secrets never enter a
task folder or a row.

A sandbox start that fails on an image pull (the agent's sandbox, a Compose group or a separate verifier's
sandbox) is retried `image_pull_retries` times in the server, with jittered backoff and behind a per-server
gate, before the failure is reported as retryable. A slow
agent-session close that fails only by timeout no longer discards the finished turn: the episode is
verified and scored with `ng_agent_observations` unset.

`separate_verifier_for_artifacts: true` verifies every task that declares `artifacts` in a fresh sandbox even
when its `task.toml` keeps Harbor's shared mode, so an agent running as root cannot tamper with the grader. The
forced verifier runs in the task's own image, an overlay-mode Dockerfile's `RUN` lines applied again (counted in
the episode budget), and the sample masks as `harbor:overlay_failed` when they fail there. Tasks without artifacts
keep their declared mode.

Resource settings come from the task. `sandbox_resources_override`, `gpu_sandbox_provider`,
`request_gpu_type` and `verifier_grace_s` belong to later stack layers and are not configuration options
in this PR.

Every sandbox the server creates (the agent's, a Compose group's `main` service and a separate verifier's,
sized by `[verifier.environment]`) is also told its CPU limit through the CPU-count hints `cpu_cap_env`
derives from it: `OMP_NUM_THREADS`, `PYTHON_CPU_COUNT`, `PYTEST_XDIST_AUTO_NUM_WORKERS`, `GOMAXPROCS` and
the other `CPU_CAP_ENV_VARS` in `nemo_gym/sandbox/utils.py`, each set to the limit's whole core count (an
override moves them with the limit). They are always set, with no switch: on many clusters a sandbox sees
the host's cores through `nproc` and `os.cpu_count()`, so build and test pools sized from them fan out far
past a one or two CPU limit and thrash or get OOM-killed, and the hints are the portable cap. They are
hints, not limits: the cgroup limit stays the provider's, and a variable the task declares itself, in
`[environment].env`, in its image's `ENV` or in the verifier environment's `env`, wins on a key clash.

Servers bind loopback (`127.0.0.1`) by default, the head server with its `/global_config_dict_yaml` endpoint
included. A harness that calls the model server from inside the task sandbox (`calls_model_from_sandbox = True`
on its agent class, or `sandbox_model_base_url` in its config) cannot reach loopback there, so
`gym eval run <target> --agent <harness>` sets `use_absolute_ip: true` together with
`absolute_ip_server_types: [responses_api_models]` in the generated run config: only the model server
advertises the node's resolved address; the head server, the agent server and this environment server stay on
loopback exactly as on a run without a sandboxed harness. The oracle (`gym dataset validate`), Terminus 2 and
mini-SWE run their commands in the sandbox but call the model from the host, so nothing moves for them. A
deployment that needs more servers reachable sets `absolute_ip_server_types` (or `default_host`, which applies
to every server) explicitly; a config file or command-line override wins over the generated default.

A registry fetch (`harbor:<dataset>`, below) reuses a task folder already in Gym's pulled datasets folder (`environments/harbor/`) when its
`manifest.toml` pin names the registry's `git_url` and `path` with a full commit SHA, the folder has a
`task.toml`, and no symlink in it leaves the folder. The recorded commit is kept on purpose when the registry's
entry moves (a deleted folder is fetched again at that commit), and the fetch does not hash the folder's content
(Harbor's own download reuses any non-empty folder without a check). Move the folder aside to fetch the
registry's current copy.

The default pulled datasets folder was `datasets/` in the working directory before it moved to `environments/harbor/`.
A folder pulled there keeps its `dataset.toml` and the `manifest.toml` that records which task versions it holds,
and a fresh pull at the new default inherits neither; `datasets/` stays gitignored for a checkout that keeps it.
To move the folder, with those files, run `mv datasets/<dataset> environments/harbor/<dataset>` (or
`mv datasets/* environments/harbor/` for every pulled dataset at once) before the next fetch or run; to leave it
where it is, set `NEMO_GYM_DATASETS_DIR=datasets`, which keeps the old root for every pulled dataset. A later
fetch reads the moved folder's `manifest.toml` as it read the old one and keeps every task folder it records,
fetching only a task folder that is missing or new to the dataset; a task folder edited by hand is not fetched
again over the edit but refused (by the fetch when it rewrites the manifest, and by `gym dataset validate`), and
a changed `dataset.toml` fetches nothing: the pull only refreshes the digests the manifest records for it.

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `environments/harbor/` in the Gym checkout, or under the working directory when Gym is installed as a package). `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`
fetches a package-store dataset instead (for example `harbor:terminal-bench/terminal-bench@4.0.0`): every task
package is downloaded by content hash into `<name>/` (`latest`), `<name>-<tag>/` or `<name>-<12 digest chars>/` (a digest pin), checked against the content hash Harbor published for it and recorded in
`manifest.toml`, whose per-task pins carry the task's package and that content hash, so a later fetch never mixes another package or a moved `latest` into the folder.
`gym dataset pull <reference>` is that fetch on its own: it prints the folder, the task count and the dataset
digest, refuses a stale, edited or foreign folder in the same words as the fetch inside `gym eval run`, and
prepares, builds and runs nothing. See [Committed dataset entries](#committed-dataset-entries) for
`gym dataset pull terminal_bench_4`, the entry-name form. A local task folder, or a folder of task folders, works the same way; task folders may be
grouped in subfolders (the search is recursive, `task_id` stays the task folder's own name and must be unique
across the dataset, and `dataset.toml` and the recorded image configurations sit at the dataset root). The
taskset mapping records the relative path of a task found below the first level under `paths`. The fetch stages its clone in a hidden `.nemo-gym-harbor-*` directory inside the dataset folder, which the search skips. A fetch interrupted under a `main` build from before this change staged under an unhidden `nemo-gym-harbor-*` name; the recursive search sees that clone and refuses the dataset (a duplicate task ID, or a task the manifest does not record) rather than running it. Delete those directories once; nothing in them is needed. The generated run config declares the materialized rows as the agent's
dataset, with the split, license, repeats and catalog metadata of the dataset's `[gym]` table (see
[Training metadata](#training-metadata)), and binds the per-taskset environment server to that agent. `gym dataset validate` reports,
per task, the verifier's wall time (`verify_s`) and the network it ran under (`network`). Running model episodes
requires a harness that accepts `sandbox_access` in its session seed and leaves the sandbox to the server
when its session is closed.

### Committed dataset entries

`environments/harbor/<entry>/` commits what a checkout keeps for one dataset: a README, deployment YAML overlays
(`terminal_bench_4/harbor_sandbox.yaml`) and a `dataset.toml` whose `[gym]` table declares `source`, the hub
reference the task folders come from (`terminal_bench_4` declares `harbor:terminal-bench/terminal-bench@4.0.0`,
`terminal_bench_2_1` the content-hash pin of the store's TB 2.1). `gym dataset pull <entry>` fetches that reference
into the entry's own folder, where the pulled task folders and `manifest.toml` are gitignored next to the committed
files, and a pull of the reference itself (`gym dataset pull harbor:terminal-bench/terminal-bench@4.0.0`, or
`gym eval run` on it) lands in the same folder, so one dataset never lives in two sibling folders. The match is
exact: `harbor:terminal-bench/terminal-bench` (the store's `latest`) is another reference and lands in
`environments/harbor/terminal-bench/` as any reference no entry declares does. `gym dataset validate <entry>` and
`gym eval run <entry> ...` resolve the bare name to that folder without fetching (an entry not pulled yet names the
pull to run; an existing folder of the same name in the working directory, or an explicit path, keeps meaning the
path for these two commands, which take paths, while `gym dataset pull <entry>` takes none and finds the entry from
any working directory), and the manifest check that follows refuses a pulled folder that was edited; task folders
in the entry's folder with no `manifest.toml`, or with one that records another source or a fetch of another
reference than the entry's `source` declares now (the declaration moved to another tag since the pull), are refused
with the pull to run, never run as a local dataset under the entry's name; the pull re-records a reference that
resolves to the tasks already there without fetching them again. The entry's
`dataset.toml` is the dataset's `dataset.toml`: its other `[gym]` keys apply as on any dataset, and the dataset digest covers it, so
`terminal_bench_4`'s identity differs from that of a bare `harbor:terminal-bench/terminal-bench@4.0.0` pull made
before the entry existed, on purpose. An entry is pulled into its own folder when that folder is under the working
root (the Gym checkout of an editable install, the working directory of a package install) and is not installed. An
installed entry, one a package install ships under `site-packages` or another Python installation prefix (shared,
often read-only; a project-local `.venv` under the working directory counts), and an entry committed outside the
working root (one a `NEMO_GYM_EXTRA_ROOTS` root commits) are pulled into `<datasets dir>/<entry>/` instead
(`environments/harbor/<entry>/` under the working directory, the default every other pull uses), by name and by the
reference the entry declares alike, and the installation is left untouched; `NEMO_GYM_DATASETS_DIR` moves every
pulled dataset there. An entry's tasks
then live in `<datasets dir>/<entry>/` with a copy of the entry's Gym layer beside them, its `dataset.toml` and whatever dataset-level `task_data.py` or `verifier.py` it
commits, placed before the first task lands and replaced when the entry's files change (the next pull, validate or
run does it alike and brings the manifest beside the tasks up to date with the copies at once, so the three commands
never disagree about one folder or its digest; a copy edited in place is refused, never overwritten: the entry's
files are where the layer is edited, and the copies are checked against the record of what was placed, a hidden
`.gym-layer.json` beside them that only the placing writes, so a run of the folder by its path in a shell without
the variable, which rewrites the manifest's digests over an edit as for any dataset, does not make the next pull
overwrite it), and the entry name resolves to that folder. That copy folder, told by its `.gym-layer.json`, is never
itself an entry: the name and the reference keep resolving past it to the installed entry, so an edit of the
installed entry's `dataset.toml` reaches the next pull, validate or run, while a folder of the entry's name without
that file (one the checkout commits itself) keeps precedence over the installed one. The record file is what marks the
copy: delete it and the folder becomes a user-authored entry of that name and shadows the installed one again, so move
the folder aside instead. A copy whose installed entry was removed or renamed is no longer reached by the name; run it
by its path as a plain dataset. A checkout that fetched
`harbor:terminal-bench/terminal-bench@4.0.0` before the entry declared it still holds `environments/harbor/terminal-bench-4.0.0/`: the reference now lands in `terminal_bench_4/` (fetched afresh;
no folder is moved), the pull and a run of the reference name the old folder while it is there, and it can be
moved aside or deleted. Runs of that reference now carry the entry's name as the taskset, as the rows'
`task_source` and in the `results/harbor/terminal_bench_4/` path, and a dataset digest that covers the entry's
`dataset.toml`, so they differ from runs of the same reference made before the entry declared it. `dataset.toml` is
not gitignored in any folder under `environments/harbor/`, pulled hub folders included, so one written into such a
folder by hand (the `nextjs-performance` override an earlier TB4 README placed in `terminal-bench-4.0.0/`) now shows
as untracked: move its keys into the entry's `dataset.toml` and delete it.

To move an entry to another version, change `source` in its `dataset.toml`, delete (or move aside) only the pulled
task folders and `manifest.toml` in the entry's folder, and pull again; the README, YAML files and `dataset.toml`
stay. A fetch refused because the folder was written for another version of the dataset, or because a task folder
no longer matches the store, says so in the fetcher's words and, when the folder is a committed entry's own, adds
what to delete there (the same hint follows the manifest check's refusal of a run by entry name; under
`NEMO_GYM_DATASETS_DIR`, or for an entry committed outside the working root, the pull's folder holds nothing git
tracks and the fetcher's words stand alone). A pull whose
fetch downloaded nothing, every folder being in place, still brings the manifest's digest of `dataset.toml` up to
date and announces the change (`Rewrote manifest.toml: dataset.toml changed; dataset digest <old> ->
<new>`, the manifest check's own notice), so the digest a pull reports after an edit of the entry's table is the one
the next run carries, and that run rewrites nothing. A run or pull of an entry whose fetch was interrupted says to run
`gym dataset pull <entry>` again, which finishes it. Publishing a dataset to the hub is not part of this slice.

### Dataset identity

`manifest.toml` beside the task folders holds derived facts only, never a copy of the dataset's name or version
(those live in `dataset.toml`): `[dataset]` `source` (`harbor-registry`, `harbor-package-store`, `git` or
`local`), `ref` (the hub reference the fetch was made from, a fact of the fetch like the pins:
`harbor:<org>/<name>@<tag | latest | sha256:digest>` for the store, `harbor:<dataset>@<version>` with the version
resolved for the registry; absent for a local folder), `digest` (the dataset's identity: a hash over the sorted per-task digests, `dataset.toml` and, when the
dataset ships them, the dataset-level `task_data.py` and `verifier.py`), `dataset_toml_digest` on its own,
`dataset_files_digest` (the two Gym files, written only
when at least one is present, so a dataset without them hashes and reads as before), and per task
`[tasks."<id>"]` `digest` (the folder digest rows carry as `_ng_digest`) with its upstream pin (`git_url`,
`git_commit_id`, `path`, or `package`, `content_hash`). The output is deterministic: fixed key order,
sorted tasks, no timestamps, so equal manifests mean identical inputs and two runs compare by `digest`.
Two datasets that differ only in a `[gym]` timeout override, or only in the shared grader, are different datasets,
on purpose.

Fetches write the manifest and record only the tasks they fetched; a task folder beside them that neither the
source names nor a previous fetch recorded stops the fetch (move it aside). Prepare writes a manifest for a
local folder, unless git tracks the folder (such as `data/tasks/` here), whose identity is computed in memory.
Before any image is built or pushed, prepare recomputes the per-task digests from the folders and refuses a
fetched dataset whose folders no longer match (the folder was edited by hand or deleted, or its source moved):
the error names the tasks and says to delete the folder, or move it aside, and run again. A single task folder
inside a fetched dataset (`gym eval run <dataset>/<task>`, or a task moved deeper under the dataset folder) is
held the same way to the manifest that check holds it to, before any image is recorded or built: the nearest
fetched manifest within the dataset check's own reach (at most six levels up, through visible folders that are
not links; a `local` manifest on the way is passed over, and a manifest further up, another tool's `manifest.toml`
in a project root for one, is not read). Its digest is compared with the one recorded for it, and an edited
folder, one the fetch did not record, or one it never finished, is refused with the same recovery step; a
subfolder of a fetched dataset is no dataset of its own, so preparing it as one is refused, naming the dataset,
rather than writing a `local` manifest inside the fetched tree. Preparing one task or one subfolder is thus no way
around the check (a single task run reports the fetch's source, not `local`; nothing is written beside the task;
a task the dataset check never sees, deeper down, under a hidden or linked folder or inside another task folder,
or one with no fetched manifest in reach, is held to nothing). The manifest holds the task's content, not its settings: a single task
folder target is prepared with the `dataset.toml` and image records beside it, as any single task target is, so a
task moved into a subfolder runs without the dataset's `[gym]` table; to run it under that table, prepare the
dataset folder and select the task. A registry fetch
re-hashes the folders it keeps only when it also adds tasks (a package-store fetch compares every folder it finds
on each run), while prepare recomputes every task digest on each run, so a hand-edited folder a re-fetch kept is
caught at prepare. A manifest naming a
source this Gym does not know is refused too, never read as local. When only `dataset.toml` or a
dataset-level `task_data.py` / `verifier.py` moved (Gym's layer, not fetched content), the derived facts are
rewritten without a fetch and the rewrite says which. A `latest` package-store folder whose manifest was written for another
version of the dataset is refused the same way: the error names the version difference and says to delete the
folder, or move it aside, and run again to fetch the store's `latest`, or to pin the version you have with
`harbor:<org>/<name>@<tag>`. A manifest written by a newer Gym (a `schema_version` this Gym does not read) is
refused untouched, by fetch and prepare alike, until Gym is upgraded or the folder moved aside. The dataset digest is
recorded in the run config (`tasksets.<name>.dataset_digest` and `dataset_toml_digest`), in every materialized
row as `task_data._ng_dataset_digest`, and, because rollout collection copies a row's `_ng_*` task data onto the
record it writes, in every `rollouts.jsonl` row as `_ng_dataset_digest` (beside `_ng_digest`), so every results
file says which dataset ran. Every materialized row also carries `task_source`, the dataset folder's name (the
parent's, when the target is a single task folder inside a dataset, one whose parent holds a `dataset.toml` or a
`manifest.toml`: its table, images and manifest are the parent's; a standalone task folder, whose parent holds
neither, is labelled by its own name, so two unrelated task folders prepared from one scratch directory do not
share its name), which the collector copies onto the results row as well. It is the label a results row and a
trainer read, not a routing key: a materialized row reaches its environment server by its taskset (the target
folder's name), `agent_map` and `fan_out` do not apply to it, and a per-dataset `num_repeats` dict keys on the
taskset.

### Training metadata

The `[gym]` table also declares what a trainer and the catalog need to know about the dataset, and the generated
`datasets:` entry carries it: `type` is the split the rows enter (the declared one, else `validation` with a
`license`, else `benchmark`, which needs no license and never enters a training split; `example` marks committed
smoke-test samples and is refused by `gym eval run` and `gym dataset validate` as soon as `dataset.toml` is read,
before any manifest check, image record, build or rows file, where a harness outside `allowed_agents` is refused
too), `license` the declared one,
`domain` (one of Gym's domain values) and `verified` (a boolean, `false` unless declared, as the pre-commit hook
starts a new resources server) are carried as declared. `[gym.defaults].num_repeats` becomes the run config's
top-level `num_repeats` (not the entry's: collate would duplicate the rows by an entry count and the collector
would multiply on top, and the metrics sidecar beside the rows records the entry). `gym eval run` inherits that
split and repeat count: `+split=` and `+num_repeats=` on the command line replace them. `gym dataset validate`
inherits the split but plays each reference solution once (it adds `+num_repeats=1` unless the command line sets
a count): validation is not an evaluation, and the oracle's result does not vary. None of these keys enters a
task's `_ng_digest`: a table that declares only them leaves every task at its bare folder hash, exactly as no
table does; only `[gym.defaults]` scaling and `[gym.tasks]` overrides fold a slice in. A `domain` outside Gym's
values is refused with the values named, like a `license` outside Gym's, so the dataset does not load until it
is fixed (the generated entry carries the same enum). A benchmark entry names no `prepare_script`; the rows are
written by the run that generates the entry. Gym's data preparation records the entry, with the rows' metrics,
in a sidecar beside the rows (`results/harbor/<taskset>/tasks_<agent>_metrics.json`, under
`results/harbor/validation/` for `gym dataset validate`) and in the collated split's
(`<rollouts stem>/preprocessed_datasets/<split>_metrics.json`). For a committed source dataset such a sidecar is a
source of truth, and a later preparation that computes something else fails with "Found conflicting aggregate
metrics". On this path it is not: the rows are regenerated from the task folder on every command and the output
folder is reused per taskset, so each run removes the previous selection's sidecars before data preparation
(prepare the one beside the rows as it rewrites them, a fresh start the collated one right before it dispatches)
and the processor writes them anew. Growing the dataset, narrowing it with `--only-tasks`/`--exclude-tasks` or
editing `type` or `license` between two runs of one taskset and harness therefore never fails, and
`overwrite_metrics_conflicts` is never needed. A resumed run (`--resume`) keeps the collated sidecar, as its rows
must be unchanged to resume at all; so does a run with `reuse_existing_data_preparation=true` and the collated
input present, which skips data preparation (that run has no rows sidecar of its own: prepare removed the earlier
one and no processor rewrote it). `domain`, `verified` and `allowed_agents` are catalog declarations,
not properties of the rows, and the sidecar leaves them out.

`allowed_agents` lists the harnesses the dataset admits, by implementation name (`responses_api_agents/<name>`,
what `--agent` resolves to); unset admits any. `gym eval run` with another harness fails as soon as
`dataset.toml` is at hand, after a hub fetch and before any image is recorded or built or any server or sandbox
started, with a message naming the harness and the admitted list. The oracle (`gym dataset validate`) plays the
reference solution, not a model, and is always admitted. The list is carried into the entry as declared, and the
collector's pairing guard reads it there: a run config edited to another harness afterwards is refused at dispatch
as well, naming the harness and the dataset (`--allow-unsupported-pairing` waives it, as for a resources server's
`allowed_agents`).

```toml
[gym]
type = "validation"
license = "Apache 2.0"
domain = "coding"
verified = true
allowed_agents = ["terminus_2_sandboxed_agent", "hermes_agent"]

[gym.defaults]
num_repeats = 3
```

### Typed task fields (`task_data.py`)

A task folder may carry `task_data.json`, a JSON object of the task's own field values; the loader writes them into the
row's `task_input.task_data` beside `_ng_digest` and `_ng_dataset_digest`, and the file enters the task digest. A dataset
that ships `task_data.py` beside `dataset.toml`, a module exporting one Pydantic model `TaskData`, types those fields
(`nemo_gym.tasks.harbor.task_data`). Prepare imports it once per dataset and checks every task folder's
`task_data.json` against it before any image is recorded or built (and the loaded tasks again before a row is
written), refusing the dataset, naming the task, when a task's fields do not satisfy the model or carry a key it does
not declare (unknown keys fail whatever the model's `extra` setting: the model is validated with extras forbidden, so
the input names pydantic itself reads, aliases, `AliasChoices` and `AliasPath` included, are the declared ones, and a
key no field consumed, a field's name where only its alias is accepted or a second spelling of a field already given,
is refused rather than ignored); values are validated strictly, as JSON (`"3"` does not pass as an `int` nor `"no"` as a
`bool`: the row and the verifier's `task_data.json` carry the file's values as written, never the model's coerced ones);
a validator that raises something pydantic does not wrap (a
`KeyError` from a lookup) is the same refusal, with the exception named. This server imports each taskset's module at startup
(`tasksets.<name>.folder/task_data.py`) and validates every row at seed: a row the model rejects, or whose fields differ
from the pinned folder's `task_data.json`, is a terminal seed failure (422) before anything is provisioned (the pin holds
for a task without `task_data.json` too: a row carrying keys beside the reserved `_ng_*` ones, which this server used to
accept and ignore, is refused; re-materialize it), and a module
that does not import refuses every episode of its taskset with the import error (logged once at startup). `gym dataset
collate` validates the rows against this server's `task_data.py` (the reserved keys) and the dataset's model in one
report. A dataset without the module is checked for the reserved keys only, as before.

## Authoring Gym datasets

Three authoring shapes sit on top of Harbor's task folder, all rendered to plain files before any server reads them
(`nemo_gym.tasks.harbor.parameterized`, `python_verifier`, `publish`):

- **Parameterized tasks.** A template folder is a task folder with `tasks.jsonl` beside its `task.toml`, one row per
  task (`{"task_id": "...", "task_data": {...}}`, or the fields flat beside `task_id`). Prepare renders it into
  `results/harbor/<taskset>/rendered/<task_id>/`, `gym dataset validate` included (its rows and results live under
  `results/harbor/validation/`, but the rendered folder is the dataset root prepare records into, so one render,
  one manifest and one built image serve both commands): files named `<stem>.template<suffix>` (`instruction.template.md`,
  `solution/solve.template.sh`, `solution/answer.template.md`) have their `{{ task_data.key }}` / `{{ key }}`
  placeholders filled from the row and are written as `<stem><suffix>` (those two spellings only: a placeholder that
  names no field, any other `{{ ... }}` left in a rendered file, such as `{{ q | upper }}` or `{{ task-data.q }}`,
  or a `{{` that never closes (`{{ a }{{ q }}`),
  fails prepare naming the task and file rather than reaching the agent as text; a literal `{{ ... }}` belongs in a
  file without the `.template` marker); every other file is copied, hidden files
  under `environment/`, `tests/`, `solution/` and `steps/` included (`environment/.env`, `.dockerignore`: they enter
  the task digest and reach the sandbox). Left out are the names the digest skips (`__pycache__`, `*.pyc`, editor
  leftovers), the template root's hidden entries (`.git`, `.venv`), `tasks.jsonl`, the dataset-level files, prepare's
  records and the root `README.md`, which describes the template (a per-task README is `README.template.md`). Two
  entries that would land under one name (`instruction.md` beside `instruction.template.md`) fail prepare naming
  both, and so does a `task_data.json` or `task_data.template.json` at the template root: the rendered task's
  `task_data.json` is written from its row, never from a file of the template (a `tests/task_data.json` fixture
  below the root is copied like any other file). The row's fields
  become `task_data.json`, typed by the template's `task_data.py` when it ships one. The rendered `task.toml` is the
  template's with `[task].name` rewritten to `<org>/<task_id>` (the template's org): Harbor keys a dataset on that
  name (`harbor add` keeps one folder per name, trial results and the viewer label by it), so every rendered task
  needs its own. A template without a `[task]` table renders without one (Harbor then names the task after its
  folder, and `harbor add --scan` asks for `harbor task update --scan` to add names); a template name that is not
  `org/name`, a name spelled some other way than a `[task]` table with `name = "..."` on its own line, or a `task_id`
  outside the characters Harbor admits in a name or longer than 255 bytes (a folder name's limit) fails prepare with
  the row named. The rendered folder is the
  taskset this server loads, with the dataset-level files beside it. Rendering is deterministic, so digests are
  stable across prepares; a re-render lands in a hidden sibling first and is swapped in entry by entry, so a task
  folder whose content did not change is left in place (a server still seeding from an earlier prepare of the same
  template never meets a folder being deleted and rewritten), a changed one is replaced by rename, a stale one
  removed, and a render that fails leaves the previous one as it was. Placeholders are inserted verbatim, so a value
  that must survive shell quoting belongs in a
  rendered text file (`solution/answer.template.md`), not inside a quoted literal of a `solve.template.sh`.
  `gym dataset init <name> --parameterized` scaffolds the shape, with the reference answer as `answer.template.md`.
- **Python verifiers.** `tests/verifier.py` grades a task that has no `tests/test.sh`; a
  `verifier.py` beside `dataset.toml` grades every task without a `tests/` folder (its content enters those tasks'
  digests). `tests/test.sh` wins whenever it exists: on the hub a `tests/verifier.py` beside it is the helper that
  `test.sh` runs (whole datasets ship that shape), so such a task is a plain `test.sh` task and the module an ordinary
  test file, uploaded with `tests/`. `verify` declares what it reads, `workspace` (the working directory, a `Path`), `task_data`, `response`
  (the agent's final response) or `state` (`{"workspace": Path}`), and returns a number or a dict with `reward`.
  Helpers live in `nemo_gym.verifiers` (`final_text`, `exact_match`, `numeric_match`) and `nemo_gym.verifiers.files`
  (`read_text`, `file_exists`, `text_file_equals`), standard library only; in the sandbox they always resolve to the
  vendored copy under `/tests/.ng`, which the runner ranks ahead of the verifier's own folder, `/tests` and the
  image's packages (a `nemo_gym` shipped in, or planted into, the task's `tests/` never replaces them). No verifier code runs on the host: at
  verification this server uploads a shim folder over `/tests` after the task's own `tests/` (a `test.sh` that finds
  a Python 3.8 or newer in the image, refusing an older one with the version named, exit 127 and no reward; the
  runner and the vendored helpers keep to 3.8, `.ng/run_verifier.py`, a vendored `nemo_gym/verifiers`, `.ng/task_data.json`,
  `.ng/response.json` and, for a dataset-level module, `.ng/verifier.py`) and runs it under the usual `test.sh`
  wrapper, in shared or separate mode, with the session's working directory as the workspace. In separate mode the
  verifier sandbox is a fresh container, so a verifier that reads `workspace` (or `state`) needs the task's top-level
  `artifacts` to include the working directory (`artifacts = ["/workspace"]`), as any separate-mode `test.sh` that
  reads the agent's files does; without it the workspace is empty there and the task scores 0. The runner writes
  `reward.txt` or `reward.json`; a verifier that raises, asks for an argument the runner does not know or returns
  something else leaves no reward (the usual missing-reward outcome, with its stderr in the verifier log). The shim
  removes any `reward.json` or `reward.txt` already in `/logs/verifier` before the runner starts, and the runner again
  before `verify` runs: Harbor shares that folder with the agent's container for the whole run and reads
  `reward.json` first, so a file planted there would otherwise be the verdict of a published task (this server
  removes the reward files before every upload; Harbor does not). An
  `async def verify` is awaited. Separate mode is Gym-only for a Python verifier: Harbor uploads nothing into a
  separate verifier container (`skip_tests_upload`), so the published shim could never reach it, and
  `gym dataset publish --harbor` refuses such a task (below).
- **`solution/answer.md`.** A task verified by its answer rather than by a script ships the reference answer as
  `solution/answer.md` (or `answer.template.md` in a template). The oracle writes the file's text to `answer.md` in
  the working directory, as the task user, with the command the published `solve.sh` runs under Harbor's oracle
  (`printf '%s' <answer> > answer.md`, `nemo_gym.tasks.harbor.publish.answer_write_command`), and submits the same
  text as its response. The oracle reads the file as bytes (`base64 /solution/answer.md` over the sandbox exec, decoded
  on the host), not as the exec's raw stdout: a provider may rebuild stdout from per-line messages (OpenSandbox's
  foreground exec, its default when `operations.background_exec` is unset, drops a trailing newline, doubles blank
  lines and loses CR characters), and the written file must be the answer byte for byte. An image without `base64`,
  or an `answer.md` that is not UTF-8 text, is `oracle: failed` with `oracle_error_type: answer`, like a working
  directory that refuses the write. The publishable shape is a verifier that reads `workspace / "answer.md"`: it grades the same
  file under `gym dataset validate` and under Harbor. A verifier that reads `response` grades in Gym only (publish
  refuses it). `has_solution` counts the file, so `gym dataset validate` plays it.

`gym dataset publish --harbor <target> --output <dir>` writes the dataset as plain Harbor task folders: the rows
expanded and checked against the dataset's `task_data.py` first (the check prepare runs, so a row `gym eval run`
refuses is refused here too, with the task named, before anything is written), every Python verifier wrapped in a
`tests/test.sh` shim (the same files as above, without the response,
the module itself under `tests/.ng/verifier.py` whether it was task-level or dataset-level, so the published `tests/`
holds one grader and the folder loads back into Gym as a plain `test.sh` task), every `answer.md` wrapped in a
`solution/solve.sh` that writes it to `answer.md` in the working directory (the file Gym's oracle leaves too), and
`dataset.toml` written without its `[gym]` table (Harbor's own tables byte for byte: the license, the defaults and the
per-task overrides are Gym's and stay with the source); what remains has to be a manifest Harbor reads, since
`harbor add` takes the dataset branch as soon as the file exists: a `[dataset]` table with an `org/name` `name` is
required (refused otherwise, with or without the `harbor` package), Harbor's own `DatasetManifest` reads the published
text as part of the format check, and a `dataset.toml` that held `[gym]` alone is not published at all (the report says
so; the output is then a plain folder of task folders). `tasks.jsonl`,
`task_data.py` and a dataset-level `verifier.py` stay behind. A task folder holding a symbolic link is refused naming
the link: publish does not follow links (a copy would ship the target's bytes, host files outside the dataset
included), as the renderer and the hub fetcher refuse them. Two Python-verifier shapes are refused with the task
named, since they could not grade under Harbor: a `verify` that declares `response` (Harbor's `test.sh` sees only the
container), and a task whose verifier runs in separate mode (`[verifier] environment_mode = "separate"`, or a
`[verifier.environment]` table, which implies it): Harbor uploads nothing into a separate verifier container, so the
shim would never run there. Both stay Gym-only; a `test.sh` task in separate mode publishes as it is. A multi-step
task (`[[steps]]`) is copied as it is, with no shim even under a dataset-level `verifier.py`: Gym grades each step by
a `test.sh` (its own or the task's), never by a Python verifier, and a shim `tests/test.sh` would become Harbor's
fallback grader for every step without its own script. Two folders that share a `[task].name` are refused naming
both (Harbor would keep one of them). The folders
are written into a hidden staging folder beside `<dir>`, Harbor's own loader (`harbor.models.task.task.Task`) checks
every one of them there when the `harbor` package is installed, and they are moved into `<dir>` only once every task
published and passed: a publish refused by Gym or by Harbor leaves nothing under `<dir>`, so a retry into the same
folder needs no cleanup. Gym does not depend on Harbor, and the report says how to install it otherwise. A
`harbor` that is installed but does not import (or a module of that name that is not Harbor, such as
`environment_servers/harbor` when that folder is on `sys.path`) is an error naming the import problem, not a skipped
check, and nothing is published. The
command exits 1 when Harbor refuses a folder (the report names each one with Harbor's message), 2 for a publish error. Six worked examples live under `environments/hello_*` (a task,
a dataset, an MCP tool, a Python verifier, verifier reuse and a parameterized dataset with `task_data.py` and
`dataset.toml`), each with a `solution/` that `gym dataset validate` plays.

## Example data

`data/tasks/` contains five small Apache-2.0 terminal tasks: copying a file, numeric sorting,
deduplicating lines, uppercasing text, and selecting the first line. Each uses `ubuntu:24.04`, a
60-second agent limit, and a 30-second verifier limit. The verifier compares the output byte for
byte with a fixture held outside the agent workdir.

`data/example.jsonl` contains their materialized rows. `data/example_rollouts.jsonl` records actual
Qwen2.5-0.5B-Instruct commands, execution results, and verifier outputs; `data/example_metrics.json`
records their measured scores and runtime. These are smoke examples from a single-command harness,
not a Hermes benchmark. They used llama.cpp on CPU and Enroot sandboxes with an in-process handoff
because Enroot cannot serialize sandbox access. Provider expiration and the distributed handoff
were not exercised by these examples.

To run the bundled tasks with Hermes and a configured sandbox provider and model:

```bash
gym eval run environment_servers/harbor/data/tasks --agent hermes_agent --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The loader generates the task mapping and dataset rows for that run.

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches the task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `environments/harbor/` in the Gym checkout, or under the working directory when Gym is installed as a package). `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`
fetches a package-store dataset instead (for example `harbor:terminal-bench/terminal-bench@4.0.0`): every task
package is downloaded by content hash into `<name>/` (`latest`), `<name>-<tag>/` or `<name>-<12 digest chars>/` (a digest pin), checked against Gym's task digest and recorded in
`manifest.toml`. `gym dataset pull <reference | entry>` is the fetch alone (see
[Committed dataset entries](#committed-dataset-entries)). A local task folder, or a folder of task folders, works the same way.

## Compose template variables and `env_file`

A task's `environment/docker-compose.yaml` may use Compose interpolation. Prepare resolves it the way Harbor's
`docker compose` resolves it, and the server reads the file through the same resolver at seed. Variables come
from these sources, the first that defines a name wins:

1. Gym's infrastructure variables, defined for every task (the table below);
2. the task's `[environment].env`, each value read as Harbor reads it: a whole value `${VAR}` or
   `${VAR:-default}` is taken from the process environment, the default standing in when the variable is unset,
   and one with neither fails prepare naming the task and the variable (Harbor fails the trial the same way);
   any other value is literal;
3. the process environment of the command;
4. `environment/.env` next to the Compose file, read as Compose's dotenv reader reads it (unquoted and
   double-quoted values are themselves interpolated against the sources above and the lines before them,
   single-quoted values are literal). A `.env` value is resolved this way at prepare and seed alike, with an unset
   name empty and warned about once; it never sees source 5;
5. at seed, the environment server's `compose_variables` (a literal `value`, a process `env` variable or a
   `model_server`'s sandbox-reachable URL; see "Supported task behavior"): the deployment's values, below every
   source above, so they fill only the names the task and Harbor leave unset. Prepare has none. In a job-level
   overlay file they stand above sources 2 to 4 instead (the overlay files are the operator's, like the values; see
   "Job-level Compose overlay"), for the pass-through forms of a service's `environment` (a bare `NAME` in the list
   form, `NAME:` with no value in the mapping form) as for `${NAME}`;
6. the default in the expression (`${VAR:-default}`, `${VAR-default}`);
7. otherwise, at prepare, an empty string with one warning naming the task and the variable, as docker compose
   does; at seed, with the deployment's values in hand, a name the Compose document or an overlay file still
   leaves unset fails the episode terminally before anything is provisioned (a `.env` value's unset name stays
   empty, as source 4 says). `${VAR:?message}` and `${VAR?message}` fail prepare, or the seed, with the message
   instead.

Source 3 is a trust boundary. `compose_variables()` includes the whole environment of the process that runs
prepare or the server (Harbor parity: Harbor interpolates with `include_os_env=True`), so a task's
`docker-compose.yaml` can reference any variable of that process, an API key or `HF_TOKEN` among them, and the
value lands in the sandbox environment and in the resolved `compose.yaml` written under `artifacts_dir`. For a
third-party hub dataset, run prepare and the server with a scrubbed environment that exports only what the
dataset needs; an allowlist of variable names is a later option.

| Variable | Value |
|---|---|
| `MAIN_IMAGE_NAME` | the resolved task image |
| `CONTEXT_DIR` | the task's `environment/` folder |
| `TEST_DIR` | `/tests`, where the task's tests land in the sandbox (Harbor leaves it empty; Harbor's own terminal-bench tasks read it) |
| `COMPOSE_PROJECT_NAME` | `hb__<hash>`, a stable per-task placeholder (Harbor's environment id of `environment/`, also the `main` image name). It is not the runtime project name: Harbor's `docker compose -p` sets that to the trial's session id, and Gym's Compose adapter names each episode's project `compose-<uuid>`. A value derived from it is consistent within one document but matches no Compose-generated container, network or project name. |
| `PREBUILT_IMAGE_NAME` | `[environment].docker_image`, when declared |
| `CPUS`, `MEMORY` | `[environment].cpus` and `memory_mb` (as `<n>M`, Harbor's spelling), when declared |
| `HOST_VERIFIER_LOGS_PATH`, `HOST_AGENT_LOGS_PATH`, `HOST_ARTIFACTS_PATH` | `/logs/verifier`, `/logs/agent`, `/logs/artifacts` |
| `ENV_VERIFIER_LOGS_PATH`, `ENV_AGENT_LOGS_PATH`, `ENV_ARTIFACTS_PATH` | the same `/logs` paths |

The forms `${VAR}`, `${VAR:-default}`, `${VAR-default}`, `${VAR:?error}`, `${VAR:+alternative}`, `$VAR` and `$$`
are supported, and a default may itself contain an interpolation. A dollar that is not an interpolation is
literal. The process environment is not part of the task digest: run prepare and the environment server with
the same variables exported, as a Harbor run does in its one process, or the document the sandbox starts
differs from the one prepare validated.

A task whose Compose file does not resolve, or whose image fails to build, is left out of the prepared rows
and recorded in `skipped.json` with the reason, like a task that does not load; the rest of the dataset still
prepares. `gym eval run` lists every task left out with its reason and stops, unless `--skip-unloadable` says
to run the loadable subset.

At seed `main`'s environment is layered as Harbor layers it: the image's `ENV` (baked into a prebuilt or built
image, or the Dockerfile's `ENV` lines the loader applies for a pull or overlay task) at the bottom, `main`'s
Compose `environment` over it, and the task's `[environment].env`, resolved on the host, over both, as Harbor's
env override file is (`harbor/environments/docker/docker.py` 363-370 and `environments/base.py` 278-285 at
v0.22.0). So a `${HF_TOKEN:-}` that interpolated to empty at prepare does not hide the host value, and a key the
file sets still beats the image's `ENV`. A sidecar's `${VAR}` is resolved again at seed with the same
host-resolved `[environment].env`, as Harbor interpolates the file for every service against it (`docker.py`
`_compose_env_vars`), so hil-bench's `ask-human` runs with the deployment's backend and keys rather than the
dataset-time defaults; only the sidecars' `environment` values change, never the structure prepare validated.
A `[environment].env` value with no default (N2) still fails prepare.

A service's `env_file` entries are read at prepare, after interpolation, so a path may use `${CONTEXT_DIR}`.
The file must sit under the task's `environment/` folder: an absolute path outside it, a path that climbs
out, or a symlink fails prepare with the path. Its `KEY=value` lines are inlined into the service's
`environment` and taken literally, quotes and inline comments handled as Compose does, and a `$` in a value
is never interpolated. The service's own `environment` wins over its files. Every service's `environment` becomes a mapping of
strings. A bare `NAME` entry in a list-form `environment` (or `NAME:` with no value in the mapping form) takes the
task variable of that name, in the order of `${NAME}` for the file that wrote it, and is
dropped when there is none. A missing
file fails prepare with its name unless the entry says `required: false`. Error messages name the file and
the line number and never repeat a line's content.

## Compose service fields

Beyond images, commands, `environment`, `depends_on`, `healthcheck`, `expose`/`ports`, `shm_size` and
`cap_add`, a task's Compose services may use the fields below. Prepare gives each its meaning (or fails naming
the field) before any sandbox starts, and the server re-reads the file through the same check at seed. Error
messages read `<task>: environment/docker-compose.yaml: service '<name>' <field> ...`.

| Field | Meaning in a sandbox group |
|---|---|
| `deploy.resources` | `limits.cpus`/`limits.memory` size the service's sandbox (then `reservations`, then the legacy `cpus`/`mem_limit`). `reservations.devices` with the `gpu` capability and a numeric `count` (or `device_ids`) becomes the service's GPU count; `count: all`, other capabilities, `limits.pids` and device `options` fail prepare. `replicas` must be 1; `restart_policy` is recorded like `restart`; every other `deploy` key names swarm scheduling and fails prepare. |
| `restart` | Accepted (`no`, `always`, `on-failure[:n]`, `unless-stopped`) and recorded on the collection (`restart_policies`). Nothing restarts: the group starts each service once, and an exit is reported through the dependency gates, as before. |
| `pull_policy` | `always`, `missing` (also `if_not_present`, `daily`, `weekly`, `every_<n>`, since the group keeps no pull history) or `never`. Applied per sandbox where the provider can (OpenSandbox's image pull policy extension, Docker's `--pull`) and recorded on the collection (`pull_policies`) otherwise, so the deployment's default pull behaviour applies. `build` is dropped with the build it names (a service prepare built, a pull or overlay `main` restating Harbor's template) and fails prepare on any other service. |
| `platform` | Must match the deployment platform, which prepare and the server read from the same setting: the environment server's `sandbox_provider_options.platform` (`{os, arch}`, as `++harbor_environment_server.environment_servers.harbor.sandbox_provider_options.platform.arch=arm64` names it) when set, else `linux/amd64`. `x86_64` and `aarch64` are the usual aliases. Image configurations are still recorded for `linux/amd64` only, so a Compose task runs end to end on amd64 deployments today. |
| `networks` | Custom networks map onto the group's one pod network as name scopes: a service resolves itself, the services sharing a network with it (by name and by their aliases on that network) and nothing else. Network declarations take `name`, `driver`, `external`, `attachable` and `labels` with no further meaning; `internal: true`, `ipam` with a configuration, `driver_opts` and `enable_ipv6: true` ask for isolation the pod does not provide and fail prepare, as do static `ipv4_address`/`ipv6_address` entries on a service; `internal: false` and `enable_ipv6: false` (Compose's defaults written out, as every kumo task does) ask for nothing and are accepted. |
| `ports` | Short (`"8080"`, `"8080:80"`, `"127.0.0.1:8080:80/udp"`) and long syntax take one shape at prepare, so the adapter reads mappings. A service's ports are endpoints the provider allocates: a container port alone (or `published: 0`) is accepted; a fixed published port, a `host_ip`, a protocol other than TCP, a `mode` other than `ingress`, a port range or an unknown option fails prepare naming the service, the refusal the adapter used to make at seed for the long syntax only (a short `"8080:8080"` passed prepare and failed every episode). A service that only needs to be reached by its peers lists the port under `expose`. |
| `extra_hosts` | Hostname-to-IP entries (list or mapping form) added to the service's hosts. `host-gateway` fails prepare. A sidecar whose image runs as a non-root user gets no host injection (the group resolves its peers through its environment URLs instead), so it cannot take them: prepare fails naming the service once the image configurations are recorded, and the adapter refuses the same document at seed. |
| `post_start` | Hooks run once the service's command has started, in order, before the group waits for health (as Compose runs them). `command` (string through the shell, or argv), `user` (default: the service's), `working_dir` and `environment`. A hook that exits non-zero fails the seed with its index and output. `privileged` hooks and `pre_stop` fail prepare. |
| `tmpfs`, `ulimits`, `sysctls`, `security_opt` | Behind a provider capability check (next paragraph). |

GPU sidecars route per service: a service whose `deploy.resources` reserves GPUs runs on the `gpu_sandbox_provider`
block when one is set, with that block's `default_metadata` and, with `request_gpu_type`, the task's
`gpu_types[0]`, exactly as the main service does when `task.toml` declares GPUs. The task's own services keep
following the task's declaration, so a CPU task with one GPU sidecar puts only that sidecar on the GPU block. Both
blocks must be able to reach each other's sandboxes on the network. Everything about a sidecar on another block
goes through that block's provider: its sandbox, its `network_mode` forwarding relay, and its volumes, which are
seeded and mounted on that provider's `shared_storage`, so a volume shared with a service on the task's block
needs the same root on both.

**Capability check.** `tmpfs`, `ulimits`, `sysctls` and `security_opt` need a provider that implements
`SupportsSandboxServiceOptions` and names the field in its `SERVICE_OPTION_FIELDS`. Prepare reads the deployment's
provider names from the keys of the `sandbox:` block the command names (the `--sandbox` provider file and the
configuration's own block) and, when `harbor_environment_server.environment_servers.harbor.gpu_sandbox_provider`
names a GPU block, from that block's keys too (values are never resolved, so credentials are not needed). Each
service is checked against the provider it will run on, the GPU block's for the services of a GPU-declaring task
and for a sidecar that reserves GPUs, and prepare fails with
`service '<name>' <field> is not supported by the <provider> sandbox provider` when that provider lacks it. Docker
applies all four as `docker run` flags; OpenSandbox applies none (a pod exposes no tmpfs, ulimit, sysctl or
security option through its API). When prepare cannot tell the provider, the adapter runs the same check against
the live provider before creating any sandbox.

## Compose volumes

A service's `volumes` entries, short (`SOURCE:TARGET[:ro]`) or long syntax, are resolved at prepare and run
in one of three ways, with the same planner at prepare and at seed:

| Volume | How the sandbox group runs it |
|---|---|
| Bind mount of a task file or folder (`./seed.json:/app/seed.json`, `${CONTEXT_DIR}/data:/data`) | Uploaded into the service once it starts, before its command runs, so the process finds the file where the Compose file mounted it. The receiving folders are created when missing, and the target, every folder a file lands in (pre-existing or not: a `0700` folder the image ships at the target becomes readable, `0755`), the folders created for the upload and the uploaded files are made readable by every user (`chmod a+rX`, never recursive); what the image already holds under the target and the upload did not write keeps its mode; `ro` is not enforced: the copy is an ordinary tree in the service, owned and moded as the provider's upload leaves it, where Harbor's `:ro` bind is a mount the kernel keeps read-only for root too. Prepare warns once per such bind, naming the service, the source and the target (a real read-only mount is tracked in issue #5062). An agent that can write the copy (root, or a user who owns the folder above it and can rename it aside) can therefore change any task data bound `:ro` into `main`, whatever folder it comes from (`environment/`, `tests/`, `solution/`, `steps/` or a top-level task file), and the verifier can read the changed copy in either mode: in shared mode `test.sh` runs in `main` itself after the agent's turn; in separate mode it runs in a fresh single sandbox that receives no Compose upload at all (the bind's copy is not there, only what the verifier image itself ships at that path, unless collected) and sees the agent's tree only through `/logs/artifacts` and the task's top-level `artifacts` entries (the `[verifier]` table carries no such key; a misplaced one is ignored), restored at their own paths with the modes the agent left, so a `:ro` target under a collected artifact path (`./reference:/app/reference:ro` with `artifacts = ["/app"]`) reaches the verifier as the agent left it, where Harbor restores artifacts the same way but its kernel-enforced `:ro` mount keeps the collected copy pristine. What `test.sh` must trust it reads from `/tests`, which is emptied and re-uploaded from the task's `tests/` after the agent's turn in shared mode and uploaded into the fresh sandbox in separate mode (when its image ships no `/tests`); a bind of a `tests/` path at another target (`../tests/golden:/golden:ro`) is an ordinary upload like the rest. A folder's tree is recreated, empty folders included, over whatever the image already has under the target (a merge; a bind mount would hide the image's folder, so a task that relies on hiding, say an `/etc/nginx/conf.d` with a default site, behaves differently), and the names the task digest skips (`__pycache__`, `*.pyc`, `.DS_Store`, editor swap and backup files) are not uploaded, so what reaches the service is what the digest covers. A task file or folder bound into several services is uploaded into each service separately: the copies are independent, so a write in one service is not seen by the others, where a Docker bind mount would share one host folder. A source the task folder does not contain, a link leaving it, a link with an absolute target (the server plans on a copy of the task, where it would leave the folder), a folder that contains a link, or a source outside what the task digest covers (`environment/`, `tests/`, `solution/`, `steps/` and the top-level task files, free of editor-leftover names such as `__pycache__`; a `../data/seed.json` could change without changing the task's identity) fails prepare with the path. The target must be an absolute path below `/` with no `..`. |
| Named volume declared under top-level `volumes:` (shared between services or not) | A folder on the provider's shared storage (`shared_storage` in the OpenSandbox block), mounted into every service that names it, copied up from the image's own content unless `nocopy`, removed when the group stops. When the `sandbox:` block's provider configures no shared storage (no `shared_storage.host_path`), a task that mounts one is left out at prepare naming the setting and the block (a GPU-reserving sidecar's volumes are answered by the `gpu_sandbox_provider` block), rather than failing at every seed; prepare reads the answer from the block's one provider key, so a block it cannot read leaves the decision to the seed. |
| `external: true` volume, or a bind mount of a host path outside the task | Resolved through `volume_sources` in the `sandbox:` block: a mapping of the volume name or host path to a path under the shared storage root. A volume with no entry fails prepare with its name. Only the entries a task's Compose file resolves this way reach the sandbox adapter, so an ordinary named volume that happens to share a mapped name stays the group's own, isolated per episode. |

```yaml
sandbox:
  opensandbox:
    shared_storage: { host_path: /mnt/shared }
  volume_sources:
    pgdata: datasets/pgdata        # volumes: { pgdata: { external: true } }
    /var/lib/models: models        # - /var/lib/models:/models:ro
```

`gym eval run` and `gym dataset validate` read `volume_sources` from the `--sandbox` provider file and the
configuration the command names; `prepare_target(..., volume_sources=...)` takes it directly. Without one,
every external volume and foreign bind mount is reported at prepare. When the configuration sets the server's
`gpu_sandbox_provider` (`harbor_environment_server.environment_servers.harbor.gpu_sandbox_provider`), a task
that declares GPUs is checked against that block's `volume_sources` instead, the block the server seeds it
from, and so is a sidecar that reserves GPUs through `deploy.resources` (the server routes it to the GPU
block, and its external volumes are seeded and mounted through that block's mapping), so a mapping has to be
present in each block whose services need it; `prepare_target(..., gpu_volume_sources=...)` takes that mapping
directly. A bind of Harbor's `/logs` tree (`${HOST_VERIFIER_LOGS_PATH}:/logs/verifier`) onto `main` is not a
mount: the sandbox adapter creates the path in the agent's sandbox before `main`'s command, hooks and
healthcheck run, as Docker creates a bind target at container create, and a different target
(`${HOST_ARTIFACTS_PATH}:/app/output`) becomes a link to it, since under Harbor both are one host folder. The
link is made after the overlay's `RUN` lines and before the uploads, where Harbor's bind sits: a folder the
image ships at the target (`RUN mkdir -p /app/output`) is replaced, with a warning in the server log when it
was not empty, as the bind would hide it, and a task file bound under the target lands in `/logs/artifacts`.
A sidecar that binds a path of the tree (kumo's `verifier` service binds `${HOST_VERIFIER_LOGS_PATH}:/logs/verifier`
and appends `actions.jsonl` there for `main`'s `tests/score.py` to read) shares it with `main` the way Harbor's one
host folder is shared: the path becomes a named volume on the provider's shared storage (declared as
`nemo-gym-logs-verifier`, `nocopy`, created and removed with the group), mounted at the sidecar's target and at the
path itself in `main`, whether or not `main` binds it, since that is where the verifier collects from; `main`'s
own instruction for the path stays, so the adapter still creates and widens it before the commands. Without shared
storage the task is left out at prepare naming the setting, like a named volume: every service the volume is
mounted in needs the storage of its own block, `main` (the task's block) as well as the binder (the
`gpu_sandbox_provider` block when it reserves GPUs), and the refusal names the block and service that lack it.
A target that nests its source fails prepare, however the target is spelled (`/logs/verifier:/logs/.` names
`/logs`), and so does a logs path with a `..` segment (`/logs/../etc`), on
`main` or a sidecar, since the path is used as written (the adapter's `mkdir -p`, the shared volume's name and
target); a sidecar's logs path with a character outside letters, digits, `_`, `.` and `-` (`/logs/my dir`) fails
prepare too, since the shared volume's name is made from it and the adapter mounts only Compose volume names.
Anonymous volumes, `tmpfs`, bind `propagation`, volume drivers and
other unsupported options fail prepare naming the field.

The `x-sandbox` key the planner writes for the sandbox adapter (`uploads`, and the sidecar adaptation's
`hosts`/`resolve_environment`) is Gym's output, never task input: a Compose file that carries one fails
prepare, and the server, naming `x-sandbox`. Upload sources are read in the server's process, so the adapter
only accepts sources under the task copy it is handed.

## Job-level Compose overlay

Harbor's job config can attach services and tools to every task of a run without touching the tasks:
`environment.extra_docker_compose` files are passed to every task's `docker compose -f ...` command after the
task's own file, `agents[].mcp_servers` entries are merged by name with the task's `[[environment.mcp_servers]]`,
and an agent's `prompt_template_path` is rendered around the instruction. The memory team's five backends (Redis,
Moraine, MAMS, Hindsight, Graphiti) attach this way. This server reads the same three things from one block,
`compose_overlay`, and prepare reads it too (from the configuration the command names, like `builds:` and the
provider blocks), so the overlay's images are recorded with the tasks' and its merge is checked per task before a
sandbox starts. No task file changes; swapping the block swaps the backend.

```yaml
harbor_environment_server:
  environment_servers:
    harbor:
      compose_overlay:
        files:
          - /path/redis-agent-memory-docker-compose.yaml         # the standalone stack
          - /path/redis-agent-memory-docker-compose.harbor.yaml  # the glue: `ports: !reset []`, attaches `main`
        mcp_servers:
          - {name: redis-memory, transport: sse, url: "http://mcp:9000/sse"}
        prompt_template: /path/memory_prompt.md.j2
```

**Files.** Each file is merged onto the task's Compose document, in order, the way Compose merges `-f` files
(`nemo_gym/tasks/harbor/compose_overlay.py`): mappings key by key; sequences appended, the keyed ones (`ports` by
host, published port, target and protocol; `volumes` by target; `expose`, `cap_add`, `env_file`, ... by value)
unique on their key with the later entry winning; `environment`, `labels`, `extra_hosts` and `sysctls` as mappings
whatever their written form; `depends_on` and a service's `networks` as mappings; `command`, `entrypoint` and
`healthcheck.test` replaced whole; every other scalar by the last file; a key written with no value in a later file
(a bare `memory_net:` under `networks` re-declaring one an earlier file configured) keeps the earlier value, as Compose
keeps it. The `!reset` tag removes the attribute the earlier files set (`ports: !reset []` drops a standalone file's
host ports, `networks.<name>.name: !reset null` drops its global network name so each episode's group gets the
project-scoped default, `environment: {DISABLE_AUTH: !reset}` unsets one variable) and `!override` replaces a
mapping or sequence instead of merging into it, as Compose defines both. The merge is by service name across every
file, the task's own included: a task whose file declares a name an overlay file declares (`memory:` with an
`entrypoint` of its own, under an overlay that gives `memory` the deployment's key in `environment`) keeps its
definition under the overlay's fields, as Compose keeps the fields a later `-f` file does not set, and prepare and
the server warn once per task and service, naming the overlay file. Tag an overlay service that carries the
deployment's values `!override` (`memory: !override {image: ..., environment: ...}`), so the merged document starts
the overlay's definition alone whatever a third-party task declared under that name; a later overlay file that merges
into a service an earlier one replaced that way (the glue file onto the standalone file's `!override` service) merges
into the overlay's definition and is not warned about; `main` is the exception, the
task's own service that the glue files extend by design. The warning is not a refusal: an overlay written for one
dataset may customize a service it knows by name, as Harbor's extra `-f` files do. Top-level `x-*` extension keys are
dropped after their anchors expand, as Compose ignores them. Every file is interpolated with the task's variables
([Compose template variables](#compose-template-variables-and-env_file), the deployment's `compose_variables`
included, since Harbor's one `docker compose` command sees one environment), so `${REDIS_VERSION:-8.2-alpine}` and
`${NGC_INFERENCE_API_KEY:?...}` resolve as they do under Harbor, with one difference in precedence: in an overlay file
the deployment's `compose_variables` stand above the task's own sources (its `[environment].env`, `environment/.env`
and the process environment), below Gym's infrastructure variables, where in the task's file they fill only the
names the task leaves unset. The overlay files are the operator's, like the values, and a task is third-party here
(the `env_file` containment, the private merged document): a task that sets `MEMORY_BASE_URL` in its
`[environment].env` cannot redirect where an overlay sidecar sends the deployment's `OPENAI_API_KEY`. The pass-through
spellings of an overlay service's `environment` (`- MEMORY_BASE_URL` in the list form, `MEMORY_BASE_URL:` with no
value in the mapping form) follow the same order: they are resolved with the overlay's variables before the merge,
where the task's own pass-through entries are read after it, in the task's order. Under Harbor
the task's `[environment].env` wins over the operator's process environment in every `-f` file (`merge_compose_env`
puts the task's env over `os.environ`); a run that needs that reading keeps the name out of `compose_variables`, since
the process environment keeps Harbor's order. A name both the task and `compose_variables` set resolves to the
task's value at prepare and the deployment's at seed, so it must stay out of the overlay's image references (below).
Prepare, though,
runs without the server's `compose_variables` (a name only they set is empty at prepare, or its `${X:-default}`
default), and records the images the overlay then names: an image reference, or a variable in one, that only a
seed-time `compose_variables` value sets is recorded at its default and refused at seed as unrecorded, which
preparing again cannot change. Image references and the variables they use must resolve at prepare: set such a
value in prepare's process environment too (source 3 above), or keep it out of image references. Its relative
paths resolve against the Compose project directory, the task's `environment` folder, never against the overlay
file's own folder: Compose resolves the relative paths of every `-f` file against the project directory ("Paths in
override files are not resolved relative to the override file's own location"), and Harbor runs `docker compose
--project-directory <task>/environment -f <task file> -f <extra> ...`, so an overlay's `./x.env` or `./conf:/etc/api`
is the task's `environment/x.env` or `environment/conf` under Harbor and here alike (an existing `environment/conf`
is uploaded as the task's own would be; a missing one is refused as a path the task folder does not contain). Write
an absolute path, or a variable with one, for a file that lives beside the overlay: the memory team's
`${GRAPHITI_CONFIG_PATH:-./config.yaml}` and `${MAMS_HARBOR_MCP_BUILD_CONTEXT:-./mcp-server}` defaults land in the
task folder under both and need the variable set, as the Harbor job sets it. An overlay's `env_file` entries may
name any readable file (an absolute path anywhere, a relative one wherever it lands): the overlay files are the
operator's, so a task file's rule (under the task's `environment` folder, part of the digest) does not apply to
them, as it does not under Harbor. A bind source outside the task (`../../../shared/conf` from the project directory,
an absolute path, a `~` path) is mapped under the provider block's `volume_sources`; a bare name in short syntax is a
named volume. The env files of the merged document are read once, after the merge, the merged
`environment` winning over their values, as Compose reads env files after merging the `-f` files: a later file's
`env_file` does not override a variable an earlier file or the task set in `environment`, `env_file: !reset []` in
a later file drops the earlier files' lists (the task's included), `env_file: !override [...]` replaces them, and a
plain entry appends to them, unique on the resolved path. A task that declares only an image gets Harbor's
one-service group first (`main`, the task image with Harbor's keep-alive command) for the overlay to extend, as
Harbor's own base files define `main` for every task. The merged document then goes through the same field check,
volume planner and image resolution as a task's file: a service field the group cannot run, a `build:` on an
overlay service, a bind source no `volume_sources` entry maps, a `secrets:` block (the adapter has no secrets; the
memory team's Redis stack reads its key from one, so a third glue file resets it and sets the variable instead) or an
overlay that removes the `services` mapping (`services: !reset`) fails prepare with a reason that names the task's
Compose file and the overlay files merged onto it in order (`environment/docker-compose.yaml merged with Compose
overlay files [...]`, or `Compose overlay files [...] (the task declares no Compose file of its own)` for an
image-only task), since the service or the field may come from any of them; an overlay file that does not parse,
interpolate or merge is named alone. The
overlay's services start with the task's group and stop with it; a named volume they declare lives on the
provider's shared storage like a task's; an extra network they declare maps onto the group's pod network as a name
scope. The merged document carries the operator's values, so it is written to a private folder per episode
(removed once the group starts), never under `artifacts_dir`, as a document with `compose_variables` is. The
environment server logs the merged group's services and MCP names once per episode, at INFO level (the servers log at
WARNING by default; `verbose: true` in the run config raises them to DEBUG), since the document is resolved per
episode. A server started with a
`compose_overlay` the dataset was not prepared with has no record of the overlay's images (`compose-images.json`
next to the task folders, or no such file for an image-only dataset): the seed refuses every episode of the run
terminally, before any sandbox, naming the services without a record and saying to prepare the dataset again
with the block, rather than failing inside the group's start and being retried. The seed then runs every check
prepare ran, on the merged document with this deployment's providers, `volume_sources` and image records: the field
check, the volume planner, the services the task's `artifacts` and `[[verifier.collect]]` hooks name present in the
merged document, and no `extra_hosts` on a sidecar whose recorded image runs as a non-root user. So an overlay file
changed since prepare into something the group cannot run (a `privileged` service, a bind source no
`volume_sources` entry maps, a removed `services` mapping, a `kafka: !reset` against a task whose collect hook runs
in `kafka`, host entries on a non-root sidecar) is refused the same way, terminally and with the reason prepare
gives, not retried for every episode.

**MCP entries.** Merged by name with the task's `[[environment.mcp_servers]]` as Harbor's trial merges the job's
`agents[].mcp_servers` with the task's (`{name: server}` over the task's entries then the job's): an overlay entry
replaces a task entry of the same name in its position, a new name goes last. The merged list is what reaches the
harness, in `/tmp/.nemo-gym/task.json` and, with `task_mcp_grants`, in the one grant list of its session seed
(`streamable-http` URL entries; `sse` entries stay in the context file: a task's own `sse` entry is warned about at
every seed, an overlay's once per run, since it is the operator's choice for the whole run). A harness that does not
declare `supports_task_tool_context` is refused for an overlay entry exactly as for a task's own.

**Prompt template.** A Jinja2 file that must use `{{ instruction }}` (refused when the block is read otherwise, as
Harbor's `render_prompt_template` refuses it), read and compiled once when the block is read and rendered with
`StrictUndefined` around the instruction of the
episode's first turn: the first user message's text of a single-turn row, or of the first step of a `[[steps]]`
task; later steps are sent as written (wrapping every turn is a parked follow-up). No episode reads or parses the
file (an edit to it is seen when the server is started again). The server renders it before
the sandbox is provisioned, so the harness and the first turn's ATIF transcript (the one its verifier reads) carry
the rendered instruction, and so does a single-turn result's `responses_create_params`; a multi-step result
records the last turn's params, as the multi-turn record does, so its rendered first turn shows in that turn's
transcript only (recording each turn's params on the turn result is a follow-up). A template that does not render
(an undefined variable) is a terminal seed failure with no sandbox. Paths in the block must exist when it is read; the memory team's `${job_config_dir}`
paths are absolute, keep yours absolute too. The template is compiled with Jinja2 wherever the block is read: this
server's requirements install it, but `gym dataset validate` and `gym eval run` read the block in the environment
`gym` runs in, which has Jinja2 through the `ray` extra (`nemo-gym[ray]`, part of `all`) and not through the base
install; without it, prepare
stops with a message naming the install (`pip install 'jinja2>=3.1'`), so install it there before preparing a dataset
with a `prompt_template`.

## Where settings live

Four homes, one owner each. A setting sits on this server's config only if every dataset on the same
deployment wants the same value; a unit test pins that list.

| Home | File | Owner | Holds |
|---|---|---|---|
| Task | `task.toml` | task author | image, resources, timeouts, env, verifier, agent user (read-only once fetched) |
| Dataset | `dataset.toml`, `[gym]` table; `task_data.py` | dataset author | `[gym]` `license` (one of Gym's dataset license values, required for `train` and `validation`), `type` (`train`, `validation`, `example` or `benchmark`; unset means `validation` with a license, else `benchmark`), `allowed_agents` (harness implementation names the dataset admits; unset admits any), `domain` (one of Gym's domain values) and `verified` (boolean); `[gym.defaults]` `timeout_multiplier`, `resource_multiplier`, `num_repeats` (rollouts per task, below `+num_repeats=`); `[gym.tasks."<id>"]` `environment` and `verifier` overrides in Harbor's own fields (`env` merges, other fields replace) and `artifacts` additions appended to the task's own. `task_data.py`: the `TaskData` model every task's `task_data.json` fields are validated against (materialize, seed, collate) |
| Deployment | the `sandbox:` provider block | operator | image rewrites and registry credentials (`images`), root setup commands (`setup`), networking, shared-memory labels, shared storage and `volume_sources` for Compose volumes |
| Run | run config | the run | provider selection, GPU routing, repeats, output paths, `sandbox_resources_override` for parity runs, `compose_overlay` (the job's extra Compose files, MCP entries and prompt template) |

`gym dataset init <name>` writes a dataset in this shape, and `gym dataset validate <folder>` with no extra
config is the test that a dataset is self-contained.
