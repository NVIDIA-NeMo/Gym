# DeepSWE

This response API agent runs one [DeepSWE](https://github.com/datacurve-ai/deep-swe)
task per `/run` request through [Pier](https://github.com/datacurve-ai/pier). Pier owns
Claude Code setup, task execution, patch capture, verification, reward parsing, and ATIF
trajectory generation. NeMo Gym owns orchestration and the sandbox provider boundary.

Pier runs in an isolated, auto-installed `0.3.0` environment. This is deliberate:
Pier's broad LiteLLM dependency currently requires a newer OpenAI Python package than
Gym's schema-compatibility pin. Process isolation preserves both projects' supported
dependency sets without weakening either pin. Runtime layout v3 exposes the current Gym
checkout or installed package root to the child interpreter through a `.pth` import root;
it does not reinstall Gym or its dependencies into Pier's environment. Runtime installation
is serialized across processes, built as a relocatable environment, and published atomically
only after the installed `pier` executable reports version `0.3.0`.
The complete child-runtime dependency resolution is pinned in
`pier-0.3.0-constraints.txt`. Pier itself is installed from the peeled v0.3.0 Git commit,
and its installed PEP 610 `direct_url.json` must identify that exact repository and commit.
The constraint SHA-256, direct-URL SHA-256, source URL/commit, and Modal SDK version are
copied into every rollout's benchmark metadata and retained runtime provenance.

The benchmark corpus is fetched rather than vendored because the upstream DeepSWE
repository does not currently declare a repository-level license. Two explicit profiles
are provided:

- `configs/deep_swe_v1_1.yaml` selects the current v1.1 task implementation at commit
  `8cae5984d5dd0ee37445beff0e928dc10c331116`;
- `configs/deep_swe_aa_v1.yaml` selects the public v1 tag at commit
  `c33fa70e68d11d85f9e58abcd5d78643705e916e` for comparison with the older Artificial
  Analysis runs.

Both commits contain the same 113 task identifiers. They therefore share one generated,
ID-only benchmark JSONL while resolving instructions, images, tests, and verification
semantics from the profile's pinned checkout at runtime. Preparation validates clean Git
trees for both commits and records each task-tree SHA-256 in every input row. It does not
copy upstream prompts or hidden tests into the Gym repository.

The two profiles intentionally preserve different verifier semantics. The AA v1 profile
uses the historical shared agent/verifier sandbox encoded by public v1; a "clean" AA run
means complete, harness-error-free attempts with captured trajectories and intact evidence,
not a claim of pristine verification. The current v1.1 profile is separately labeled and
uses a fresh pristine verifier sandbox. Mixing v1.1 verifier semantics into AA v1 would no
longer be a reproduction of the public historical profile.

Each profile sets `num_repeats: 3`, matching Artificial Analysis's three attempts per
task. The prepared JSONL still contains exactly 113 rows; rollout collection applies the
three repeats.

## Quickstart

DeepSWE requires Python 3.12, `uv`, Git, Modal authentication, and a model endpoint that
implements Anthropic Messages at `/v1/messages`. Install Gym's development and sandbox
dependencies:

```bash
uv sync --extra dev --extra sandbox
source .venv/bin/activate
```

Create an owner-only `env.yaml` in the Gym root. Keep the API key in the process
environment rather than writing its value into YAML:

```yaml
policy_base_url: https://inference-api.nvidia.com
policy_api_key: ${oc.env:INFERENCE_API_KEY}
policy_model_name: nvidia/zai-org/evals-glm-5.2
```

`policy_base_url` is the service host root; Claude Code appends `/v1/messages`. Export
`INFERENCE_API_KEY`, then authenticate Modal with a local profile or its standard token
variables:

```bash
export INFERENCE_API_KEY=...
modal token set --token-id ... --token-secret ...
# Alternatively export MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
```

Prepare the pinned 113-row AA profile:

```bash
gym eval prepare \
  --config responses_api_agents/deep_swe/configs/deep_swe_aa_v1.yaml \
  --config nemo_gym/sandbox/providers/modal/configs/modal.yaml
```

The one-shot evaluation command starts the Gym agent server, collects three attempts for
every task, and retains the rollout JSONL plus per-trial Pier evidence:

```bash
mkdir -p results
gym eval run \
  --config responses_api_agents/deep_swe/configs/deep_swe_aa_v1.yaml \
  --config nemo_gym/sandbox/providers/modal/configs/modal.yaml \
  --agent deep_swe_aa_v1 \
  --output results/deep_swe_aa_v1_rollouts.jsonl \
  --split benchmark \
  --num-repeats 3 \
  --concurrency 4
```

With the one-shot command, `--split benchmark` selects and preprocesses the
benchmark dataset declared by the loaded agent config. An explicit `--input`
is useful only with `--no-serve`, when collecting against servers that are
already running.

For server debugging, start the same merged config with `gym env start` in one terminal, then
add `--no-serve` to the `gym eval run` command in another. A five-task pipeclean may use
`--limit 5`; Artificial Analysis comparison evidence requires all `113 * 3 = 339`
attempts with `claude_code_install_method: pier`.

## Sandbox design

The default profile uses Gym's provider-neutral `AsyncSandbox` through a Pier
`BaseEnvironment` adapter. Include exactly one provider config along with the agent
and model settings. For Modal:

```bash
gym env start \
  --config responses_api_agents/deep_swe/configs/deep_swe_v1_1.yaml \
  --config nemo_gym/sandbox/providers/modal/configs/modal.yaml
```

This follows the same Gym Sandbox API lifecycle used by
[PinchBench PR #1810](https://github.com/NVIDIA-NeMo/Gym/pull/1810): the benchmark
adapter owns task orchestration, while `AsyncSandbox` owns provider-selected create,
exec, transfer, status, and teardown operations. Modal is the shipped provider for
DeepSWE, not a direct Pier-native sandbox bypass.

For the AA profile, use `configs/deep_swe_aa_v1.yaml` instead. Set `policy_base_url` to
the inference service's host root; Claude Code appends `/v1/messages` itself. The AA
profile also supplies the non-secret NMP principal and batch-priority request headers
and disables nonessential Claude Code traffic so a model-only egress allowlist stays
quiet.

### Agent configuration

| Field | Default or shipped value | Constraint |
|---|---|---|
| `model_base_url`, `model_api_key`, `model_name` | Required | Anthropic Messages host root, secret, and exact served model ID. |
| `claude_code_version` | `2.1.153` | Exact `x.y.z` version, verified against the live sandbox executable. |
| `claude_code_install_method` | `pier` | `npm` is diagnostic-only and invalid for AA evidence. |
| `claude_code_kwargs`, `claude_code_env` | `{}` | Non-secret agent options; secrets belong in Gym or Modal secret handling. |
| `benchmark_git_url`, `benchmark_git_commit` | DeepSWE plus profile pin | Checkout must be clean and exactly pinned. |
| `benchmark_expected_task_count` | `113` | Preparation and startup fail on mismatch. |
| `benchmark_cache_dir`, `benchmark_path` | Private cache, `null` | A supplied path must be a clean pinned Git checkout. |
| `sandbox_provider` | `sandbox` | Must resolve to exactly one Gym Sandbox provider. |
| `sandbox_required_provider` | `modal` | Shipped profiles reject another resolved provider. |
| `sandbox_spec` | Profile-defined | Image/runtime settings passed through the provider-neutral API. |
| `sandbox_supports_*` | Profile-defined | Capability claims must reflect enforcement by the selected provider. |
| `sandbox_preinstall_agent_in_image` | `true` in profiles | Installs Claude Code in a cached image layer before runtime egress restriction. |
| `sandbox_transfer_timeout_s` | `1800` | Positive finite deadline for each provider upload or download operation. |
| `work_root` | UID-private temp root | The root and created job directories must be current-user-owned mode `0700`; ancestors must be root- or current-user-owned and not attacker-writable. |
| `pier_runtime_dir` | UID-private runtime cache | Its private parent must be current-user-owned mode `0700`. Runtime entries must be current-user-owned and not group/other-writable; executable and read-only entries deliberately use modes such as `0755` and `0444`. |
| `pier_cancel_grace_s` | `120` | Positive bounded subprocess cleanup interval. |
| `max_concurrent` | `4` | Concurrent Pier trials per agent process. |
| `max_concurrent_assembly` | `2` | Concurrent bounded artifact assembly workers. |
| `artifact_max_files` | `2048` | Maximum files inventoried per trial. |
| `artifact_max_file_bytes` | `64 MiB` | Per-file read/hash ceiling. |
| `artifact_max_total_bytes` | `256 MiB` | Maximum declared and hashed artifact-set size per trial. |
| `trajectory_max_bytes`, `patch_max_bytes` | `64 MiB`, `16 MiB` | Maximum trajectory and patch returned inline; each must not exceed the per-file ceiling. |

The Pier launcher does not inherit host-level Claude, Anthropic, Bedrock/AWS model,
or `PIER_*` behavioral controls. In particular, host effort, turn, thinking-token,
adaptive-thinking, model, and auth fallbacks are removed before Gym injects only the
configured `ANTHROPIC_AUTH_TOKEN`. Ordinary process essentials such as `PATH`, `HOME`,
locale, and proxy settings remain available, as do Modal credentials. Set intentional
agent behavior through `claude_code_kwargs` or non-secret `claude_code_env`; those values
are serialized explicitly in the per-trial Pier config.

Each DeepSWE v1.1 rollout uses one isolated agent sandbox followed sequentially by one
fresh pristine verifier sandbox. Its verifier Dockerfiles are an audited template: one
concrete `FROM`, four hidden-test `COPY` instructions, and a `chmod`. The adapter starts
that base image and uploads the hidden tests only to the verifier sandbox. It fails closed
if a future task adds an unrecognized Dockerfile instruction.

Gym does not treat Pier's requested agent metadata as proof of the installed executable.
The agent sandbox runs Pier 0.3.0's native verification command,
`export PATH="$HOME/.local/bin:$PATH"; claude --version`, at startup and requires exactly
the configured `x.y.z` version. After the agent exits, the adapter runs that command again
immediately before archiving `/logs/artifacts`. Once the bounded archive is downloaded and
validated, Gym atomically replaces any sandbox-supplied claim with an owner-only local
`gym-agent-version.json` containing the expected version, observed version, agent name,
and verification command. Pier manifests that host-generated file afterward. A missing,
ambiguous, or mismatched post-run version aborts artifact collection. Verifier sandboxes
have no agent install specification and do not emit this proof.

The configured provider must genuinely enforce `allow_internet=false` while allowing
only the model endpoint. Do not mark `sandbox_supports_disable_internet` or
`sandbox_supports_filtered_egress` true for an unrestricted runtime. In particular,
the current Apptainer provider shares host networking and is not a faithful DeepSWE
runtime without cluster-level egress enforcement.

The shipped profiles additionally require the resolved provider name to be `modal`.
Modal does not expose a Sandbox disk request in SDK 1.5.1, so the adapter omits Pier's
20 GiB request and checks the live root filesystem before every trial. A sandbox whose
actual free capacity is below the task requirement fails before task setup and execution.
With `sandbox_preinstall_agent_in_image: true`, a cold cached-image build can install
Claude Code during sandbox creation before that live capacity check runs.

Task `docker_image` references are currently passed to the provider exactly as pinned
in the DeepSWE source tree. Those references are opaque registry tags, not digest-qualified
names. A point-in-time tag-to-digest inventory can establish availability, but it does not
prove which digest a later sandbox pull used; retain provider-resolved image provenance or
rewrite to audited digest references before claiming image-byte parity.

Gym records a provider-issued sandbox ID, requested image reference, environment role, and
session identifier in an owner-written artifact for every agent or verifier environment.
Modal 1.5.1 does not expose the registry image digest selected for a Sandbox, so the
`resolved_image_digest` field is deliberately null. These observations correlate lifecycle
events; they do not upgrade a mutable image tag into image-byte provenance.

Every rollout also records the current Gym origin URL, full Git commit, `uv.lock` SHA-256,
and whether the tracked and untracked (but non-ignored) working tree is clean. A wheel installation or non-Git checkout may report
unavailable source fields for ordinary use; strict reproduction evidence must run from the
clean pinned checkout and reject any unavailable or mismatched field.

Every Gym rollout uses the provider-neutral Sandbox API; the agent has no native Pier
environment bypass. Standalone Pier runs remain useful as an external parity reference,
but they are not Gym rollout evidence.

The AA profile keeps `claude_code_install_method: pier`, which preserves Pier's native
Claude Code installer. For local diagnostics on emulated x86 Docker hosts whose CPU
lacks the AVX features required by that Bun installer, `claude_code_install_method:
npm` selects the same exact Claude Code version from its npm package. Runs using that
portability fallback record `agent_install_method: npm` and must not be mixed into the
AA reproduction evidence.

## Output, errors, and evidence

A completed, safely finalized Pier trial returns:

- binary `reward` and the structured Pier verifier result when verification ran;
- the full ATIF trajectory in `raw_rollout.trajectory` when it is present, valid, and
  within the configured inline limit;
- `model.patch` when the selected corpus produces one;
- Pier's structured trial result and exception classification;
- SHA-256 and size for every file in the retained Pier trial directory;
- pinned benchmark, Pier source/runtime lock, Claude Code install method and version,
  model, sandbox runtime policy, task checksum, and trial provenance.

For completed structured `/run` responses, the status fields distinguish benchmark
failures from harness failures:

| `status` | Meaning | Evidence fields |
|---|---|---|
| `success` | Pier produced a finite binary verifier reward and a valid ATIF trajectory with at least one structured agent step. | Trial, verifier, trajectory, patch, and manifest are returned within configured limits. |
| `error` | The Pier trial ran but reported an exception, omitted a verifier result, or produced an invalid reward. | Safely finalized partial trial and trajectory evidence may still be present; exposed `reward` is `0.0`. |
| `harness_error` | Launch, integration, trajectory parsing, artifact limits, or evidence finalization failed. | Exposed `reward` is `0.0`; partial fields are returned only when bounded and safely scrubbed. |

Request-validation failures such as a missing task ID, and caller cancellation, can end
at the transport or collector layer without a structured rollout response. Diagnose
those through collector failure output and the Gym logs.

Use `error_type` and the redacted `error_message` first. When `benchmark_metadata.job_dir`
is present, inspect `gym-pier-stdout.log`, `result.json`, and the manifest. An
`UnsafeArtifactError` means scrubbing or permission finalization failed: Gym removes or
quarantines the job under its private root and deliberately returns no job path, trial,
verifier result, raw rollout, or artifact manifest. If both deletion and quarantine
fail, the unreferenced directory remains inaccessible below that private root and its
path is still withheld.

The full job directory remains under `work_root`. Depending on how far the trial ran,
it can contain Pier's `config.json`, `lock.json`, `result.json`, logs, trajectory,
patch, CTRF, raw verifier output, and framework-native reports. Default checkout,
runtime, and job caches live below a
UID-scoped owner-only temporary root; configured job roots must also be real,
current-user-owned `0700` directories. Finalization rejects symlinks, hardlinks,
special files, unsafe names, more than 2,048 files or 8,192 total entries, and configured
file/total byte overflows. It scrubs every retained regular file, hashes that same stable
byte view into the manifest, then seals directories `0500` and files `0400` before
assembling the response. A binary verifier reward is not accepted as a successful
rollout unless the ATIF trajectory is parseable and has at least one structured step
whose source is `agent`.

Sandbox archive transfer uses the same configured evidence budget: each member is capped
by `artifact_max_file_bytes`, cumulative expanded payload by
`artifact_max_total_bytes`, and raw/member counts by `artifact_max_files` plus bounded tar
overhead. The adapter enforces these limits while streaming before extraction, so a
highly compressible model-controlled archive cannot expand to a separate multi-gigabyte
intermediate budget.

The retained directory is evidence, not a standalone replay bundle. Pier's persisted
environment config points to a secret-bearing `sandbox-runtime.json` that exists only
during launch and is deleted afterward. Direct `pier run` replay is therefore expected
to fail. Rerun through Gym with the same input row, pinned benchmark and agent config,
model ID, Gym commit, `gym-runtime-provenance.json` policy, and fresh
`INFERENCE_API_KEY`/Modal authentication. After finalization, copy the directory to
content-addressed storage if immutable evidence is required; do not mutate the active
job tree.

Before launching Pier, Gym removes inherited variables whose names look like tokens,
keys, passwords, credentials, or authentication values. Ordinary non-sensitive parent
variables are inherited. The explicit model token and environment-sourced Modal token
pair are passed when needed and included in artifact redaction. Do not store secrets
under unrecognized variable names or in `claude_code_env`.

## Artificial Analysis comparison

Artificial Analysis reports DeepSWE scores for Claude Code with GLM-5.2, GLM-5.1,
DeepSeek V4 Pro, and Kimi K2.6. Those runs predate the v1.1 corpus publication, so the
AA comparison profile deliberately uses the public v1 tag. Artificial Analysis does not
publish the exact source commit or every generation setting; evidence must identify the
public v1 pin used here, record deltas, and avoid claiming an unpublished exact match.
The v1.1 profile is current-benchmark support, not evidence for reproducing the older AA
numbers.
