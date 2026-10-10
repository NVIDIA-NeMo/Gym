# NOOA benchmark baselines

Run the full public SWE-bench Pro (731 tasks), Terminal-Bench 2.1 (89 tasks),
and GDPVal (220 tasks) through Gym's native environment lifecycle. The same
entrypoints accept an existing OpenAI-compatible endpoint or a separately
served checkpoint. Provider credentials, model aliases, images, filesystem
paths and cluster allocations belong to the caller.

The baseline uses temperature 1, top-p 1, a 262144-token context and 32768 maximum
output tokens. Task prompts, task-specific deadlines and verifier settings
remain in the benchmark recipes. This is a documented baseline, not a claim
of exact reproduction of a colleague's private run. Use a model that supports
these settings, or record an explicitly changed experiment configuration.

## Install and prepare the exact inputs

Use a committed checkout and an owned output directory outside the repository.
Install Gym with its dev/sandbox extras and the NOOA revision required by the
agent's `requirements.txt`. Follow the benchmark READMEs for provider/image
prerequisites and access to datasets. Neither these scripts nor the dataset
preparers provision a shared inference service automatically.

```bash
export RUN_ROOT=/absolute/path/to/new-owned-run
mkdir -p "$RUN_ROOT/data"
python - <<'PY'
import os
from pathlib import Path
from benchmarks.swebench.pro.prepare_nooa import prepare_native as swe
from benchmarks.terminal_bench_2_1.prepare_nooa import prepare_native as tb
from benchmarks.gdpval.prepare_nooa import prepare_native as gdp
root = Path(os.environ['RUN_ROOT']) / 'data'
swe(output=root/'swebench-pro-nooa-731.jsonl', max_output_tokens=32768)
tb(output=root/'terminal-bench-2.1-nooa-89.jsonl')
gdp(output=root/'gdpval-nooa-220.jsonl')
for name, count in [('swebench-pro-nooa-731.jsonl',731),
                    ('terminal-bench-2.1-nooa-89.jsonl',89),
                    ('gdpval-nooa-220.jsonl',220)]:
    with (root/name).open('rb') as stream:
        assert sum(1 for _ in stream) == count, name
PY
```

TB preparation also preserves its pinned task checkout and verifier files.
Ship that checkout with the controller; when deploying elsewhere, use
`deployed_repository_path` in its preparer. Do not replace task identities with
filesystem paths. Keep GDP's canonical flat input for later judging.

## Optional BenchAgent baseline

The default TaskAgent uses NOOA revision
`19caab169b018476ac433d040f6ae3f06aeff101`. Run BenchAgent in a new output directory
and separate environments so existing TaskAgent results and dependencies remain
reproducible. The optional profile pins **all three** packages, `nooa`,
`nooa-cli` and `nooa-bench`, to
`19caab169b018476ac433d040f6ae3f06aeff101` in
`responses_api_agents/nooa_agent/runtime/benchagent-requirements.txt`.

First complete standard serial `gym env prefetch --config "$BENCHMARK_SETUP_CONFIG"`
using your resolved benchmark setup config. Its server set must match the run,
with `uv_venv_dir: RUN_ROOT/server-venvs/swe` for this SWE example (use `tb` or
`gdp` for the others). Then install the profile into both the private controller
and NOOA agent-server environments. From the Gym checkout:

```bash
GYM_ROOT="$PWD"
PROFILE="$GYM_ROOT/responses_api_agents/nooa_agent/runtime/benchagent-requirements.txt"
CONTROLLER_PYTHON="$(command -v python)"
AGENT_PYTHON="$RUN_ROOT/server-venvs/swe/responses_api_agents/nooa_agent/.venv/bin/python"
(
  cd "$GYM_ROOT/responses_api_agents/nooa_agent"
  for interpreter in "$CONTROLLER_PYTHON" "$AGENT_PYTHON"; do
    uv pip install --python "$interpreter" --override "$PROFILE" \
      -r requirements.txt -r "$PROFILE"
    "$interpreter" -c 'import nooa, nooa_cli, nooa_bench; from nooa_bench.bench_agent import BenchAgent'
  done
)
```

Retain the standard setup completion markers and verify imports in every server
environment before reuse. The explicit override keeps the core, CLI and BenchAgent
packages on the same NOOA revision in these new environments. After this installation, omit `--prefetch` from
the launch: ordinary prefetch installs the default agent requirements again.
The launcher must find every server environment complete to reuse them.

Pass the matching overlay to **both** canary and full phases:

```bash
python -m benchmarks.nooa_baselines.run_eval \
  --run-root "$RUN_ROOT" --benchmark swe --phase canary \
  --config benchmarks/nooa_baselines/benchagent-swe.yaml
python -m benchmarks.nooa_baselines.run_eval \
  --run-root "$RUN_ROOT" --benchmark swe --phase full --concurrency 16 \
  --config benchmarks/nooa_baselines/benchagent-swe.yaml
```

Use `benchagent-tb.yaml` or `benchagent-gdp.yaml` with the corresponding benchmark;
`run_parallel` accepts these through its per-benchmark `--swe-config`,
`--tb-config` and `--gdp-config` options. Provider configuration and runtime
prerequisites still apply, including GDP's optional `--prepare-apptainer`.

Each overlay selects `nooa_bench.bench_agent:BenchAgent`, the
`responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent` adapter,
and optional `runtime_requirements_file`. That file selects dependencies for
Gym's normal sandbox-runtime preparation; it does not install server packages
or replace the runtime bootstrap. Omitting it preserves the default TaskAgent
profile. The adapter preserves the task working directory, upstream summarizer
and delegation defaults, structured return, and shared Gym model-call accounting
and capture. Both TaskAgent and BenchAgent recipes use `max_policy_calls: null`,
so model calls are uncapped; a positive value opts into a shared rollout limit.
Time and context limits still apply. Direct invocation keeps exceptions visible
to Gym for failure reporting. Each new agent/provider combination still requires
a real canary before full dispatch.

## Use an existing endpoint

Set the API key in an environment variable without putting it in configuration
or shell history. For a caller-owned unauthenticated local service, explicitly
set `POLICY_API_KEY=EMPTY`. Registration records only the environment-variable
name. The preflight then makes two small real tool-calling requests; it is a
transport check, not a benchmark result.

```bash
python -m benchmarks.nooa_baselines.serving endpoint \
  --run-dir "$RUN_ROOT/model" --base-url "$POLICY_BASE_URL" \
  --served-model "$POLICY_MODEL_NAME" --api-key-env POLICY_API_KEY
python -m benchmarks.nooa_baselines.serving preflight --run-dir "$RUN_ROOT/model"
```

The preflight is bound to the exact model manifest hash. Changing an alias,
endpoint or local allocation requires a new manifest and preflight directory.
Credentials are not stored in those receipts; the two harmless preflight
responses are stored. Existing receipts are never overwritten.

## Run a canary, then all tasks

The included SWE/TB recipes use OpenSandbox: supply `OPENSANDBOX_DOMAIN` and
`OPENSANDBOX_API_KEY` in the controller environment. The controller must be
reachable from those sandboxes. GDP uses an audited Apptainer image supplied
as `GDPVAL_CONTAINER_PATH`; match its CPU architecture to the controller.
Provider overrides can be supplied in a caller-owned Hydra YAML with `--config`.
The GDP recipe preserves a six-hour outer episode deadline and explicit
preparation/cleanup deadlines; its whole-agent exec default is unlimited so a
provider's short command timeout cannot kill a legitimate episode.

```bash
python -m benchmarks.nooa_baselines.run_eval \
  --run-root "$RUN_ROOT" --benchmark swe --phase canary --prefetch
python -m benchmarks.nooa_baselines.run_eval \
  --run-root "$RUN_ROOT" --benchmark swe --phase full --concurrency 16
```

Choose concurrency
for the actual endpoint and sandbox capacity. Replace `swe` with `tb` or `gdp`
for the other benchmarks. GDP on a non-setuid Apptainer or root container should
also pass `--prepare-apptainer`: the existing helper validates the installation
and, when needed, changes only session capacity in an owned private config.
It never changes host/shared software or bypasses unsupported setuid behavior.
The HSG root-in-Pyxis profile uses an explicit child user namespace; other
placements may need their documented provider settings via an overlay.

`--prefetch` uses standard serial `gym env prefetch` with private per-benchmark
venvs/cache. An incomplete install is revalidated without deleting shared
files. A completion marker is only installation metadata: the actual server
startup and real canary still have to pass imports, model calls and cleanup.
Run first canaries sequentially with prefetch to avoid a simultaneous package
installation storm. Fully prepared benchmarks can then advance independently:

```bash
python -m benchmarks.nooa_baselines.run_parallel --run-root "$RUN_ROOT"
```

This entrypoint reuses successful existing canaries after the full launcher
checks their hashes; otherwise it runs each real canary first. Pass
`--prepare-apptainer` for the GDP placement described above. A failed canary blocks only its
own full dispatch. `--swe-model-dir`, `--tb-model-dir`, `--gdp-model-dir` select
separate preflighted replicas. Per-benchmark `--gdp-config`/`--gdp-concurrency`
(and SWE/TB equivalents) pass explicit settings. There are no private external
provider-check scripts required by this entrypoint.

Full collection expands Gym's preserved canary materialized schedule to the
entire prepared input and reuses its completed outcome. Input, model, source
and overlay hashes must match. A correctly graded zero-reward canary passes;
a masked infrastructure error does not. A reviewed source-only transition
requires the explicit prior `--canary-source-manifest`. Do not use a blanket
resume to regenerate model failures or overwrite old capture files.

For fixed parallel shards, create disjoint native input files and pass each with
`--input` in its own run directory. Preserve every canonical `task_id` and its
original unique, nonnegative integer `_ng_task_index`; sparse global indices
are supported and must not be renumbered per shard. Use the same full shard file
for its canary and full phases, then reconcile the union against the original
731, 89 or 220 identities. A separate validation-only input belongs in a separate
run directory and does not count toward that union.

Each phase saves its input/config/source provenance, controller log, model
captures, native results and failure sidecar. Immediately after Gym exits, it
fsyncs `native-exit.json` before reading any results. Reporting reads physical
LF records, preserving Unicode characters inside JSON strings. `completion.json`
separates attempted coverage from valid grades/generation and ungraded tasks;
complete coverage does not imply every task was solved or graded. Missing
completion with a present native-exit receipt is a reporting failure to inspect,
not a reason to rerun inference. Neither receipt claims independent external
sandbox deletion beyond the native lifecycle; keep provider cleanup evidence.

GDP generation exports are masked/ungraded, durable candidate artifacts.
Score them afterward through the separately pinned canonical AA-v2 judge-only
recipe and `benchmarks/gdpval/prepare_nooa_judging.py`; see the GDP README.
Keep failed/missing exports explicit and never synthesize losses or ELO.

## Optional HSG checkpoint serving example

`serve_hsg.sbatch` preserves the evaluated Nemotron-family serving settings:
four GPUs, TP4/expert parallelism, FP8 KV cache, 262144 context, 8192 batched tokens,
32 sequences, `qwen3_coder` tool and `nemotron_v3` reasoning parsers. It loads
checkpoint custom code with `--trust-remote-code`, offline, and records its
hashes. This example is not a generic configuration for arbitrary architectures.
Choose compatible approved checkpoint/image paths; pass account and any
partition/QoS/resource overrides explicitly to `sbatch`.

```bash
mkdir -p "$RUN_ROOT/model"
cp benchmarks/nooa_baselines/{serve_hsg.sbatch,serve_in_container.sh,serving.py} "$RUN_ROOT/model/"
sbatch --account="$SLURM_ACCOUNT" --chdir="$RUN_ROOT/model" \
  --output="$RUN_ROOT/model/serve-%j.log" "$RUN_ROOT/model/serve_hsg.sbatch" \
  "$RUN_ROOT/model" "$CHECKPOINT" "$VLLM_IMAGE" "$POLICY_MODEL_NAME" "$PYXIS_MOUNTS"
```

`PYXIS_MOUNTS` must cover the checkpoint, image and owned run directory at the
same absolute paths. GPU jobs use a clean environment without mounting home.
`model.json` records the actual image hash, checkpoint metadata and weight
sizes/timestamps, allocated GPU/driver identity and exact server arguments;
weight contents are explicitly not hashed. It is not a readiness receipt.
After vLLM is ready, run the same `serving preflight` command from the controller
network with the explicit `--job-id` assertion for that allocation.

`bootstrap_hsg.sbatch RUN_ROOT RUNTIME_IMAGE PYXIS_MOUNTS` installs only a private
controller environment from `RUN_ROOT/Gym`. `evaluate_hsg.sbatch RUN_ROOT
RUNTIME_IMAGE PYXIS_MOUNTS PRIVATE_RUNTIME_ENV [run_parallel options...]`
uses that environment and sources the caller's trusted private shell file for
endpoint/provider credentials. Restrict that file's permissions; never commit
it. The CPU examples default to 16 CPU/64 GiB setup and 32 CPU/128 GiB evaluation,
zero GPUs; supply account/partition/QoS and adequate memory/time for your
chosen concurrency. Apptainer task processes consume controller resources;
OpenSandbox task resources are separate from the Slurm allocation. Release
only the allocations you created after all controllers finish.
