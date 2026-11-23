# Adapter contract and family recipes

The contract every adapter follows, then three recipes, one per piloted family. Each recipe names its image,
instruction, verifier, file map and the probes it needs. Start from the pilot adapter of your family and change only
what your environment does differently. Paths are relative to the repository root. See
[Known deviations](#known-deviations-of-the-pilot-adapters) for where a pilot adapter still differs from the contract.

## Adapter contract

Files under `resources_servers/<server>/harbor/`: `adapter.py`, `task-template/` (from `gym dataset init <name>
--parameterized` when tasks differ only in data; one folder per row when each row has its own image), `README.md`,
`configs/harbor_arm.yaml`, `configs/harbor_arm_harness.yaml`, `configs/native_arm.yaml`, `configs/parity_ids.txt`
(when the rows are public); and `resources_servers/<server>/tests/test_harbor_adapter.py`, with an empty
`tests/__init__.py` if the folder has none, so two servers' `test_harbor_adapter.py` cannot clash on module name.

git does not track empty directories: if the adapter writes into a template folder that has no tracked file (for
example `environment/` when the Dockerfile is generated), create it in `convert()` (`mkdir(exist_ok=True)`); a
`.gitkeep` would be copied into every task. Test a conversion from a template copy without that folder.

CLI:

| Flag | Meaning |
| --- | --- |
| `--rows FILE` | Required. The native rows, streamed. |
| `--output-dir DIR` | Required. Absent or empty, else exit 2. |
| `--ids-file FILE` | The `subset` output, `#` comments, order kept: values of the row field `--id-field` named, or 0-based line indexes for `_index`. A duplicate or missing id exits 2. |
| `--native-rows-out FILE` | The selected raw lines, in `--ids-file` order: the native arm's input. |
| `--id-map-out FILE` | JSON `{native key: task id}`, sorted keys. The native key is the 0-based line in `--native-rows-out`, which is what native rollouts carry as `_ng_task_index`. |
| `--native-config FILE` | When the adapter reads settings: a `gym env resolve` dump or the pairing YAML. |

Refuse with exit 2 every native setting the task cannot reproduce (math: judge on, format-tolerant extraction,
another entrypoint or `system_prompt_mode`).

The id map pairs the arms: `harbor_parity --native-key _ng_task_index --id-map <map>` and `check --native-rows
<native-rows-out> --native-id _index --id-map <map>`. When the rows carry an id field, that field can key both
instead (`--native-key <field>`, which harbor_parity finds through the materialized inputs, and `--native-id <field>`),
with a map only when the adapter renames tasks.

Deterministic bytes: sorted keys, no timestamps, tar and gzip mtime 0. Two conversions must `diff -r` empty.

`<output-dir>/README.md`: rows basename, sha256 and line count, selected count, ids-file sha256, Gym commit (or
`unknown`; the adapter must run outside a git checkout), each vendored file's sha256, and the regeneration command
with inputs as basenames and the output as `<dir>` (a real path differs between two conversions and leaks local
paths).

Vendored files are copied unchanged (an `ast` segment copy only when `app.py` imports `nemo_gym`; see
[Vendoring a grader](#vendoring-a-grader)), with the header
`Vendored unchanged from <repo path | URL> at <Gym commit | upstream SHA>; source sha256 <hex>; license <SPDX id>`.
Third-party code also goes in the PR body (AGENTS.md "Code provenance").

Run configs:

- `harbor_arm.yaml`: harbor server and deployment settings, passed to every Harbor `gym dataset validate` and
  `gym eval run`: `separate_verifier_for_artifacts: true` whenever the agent could write the reward; for a Compose
  family also `sandbox.opensandbox.networking.enabled: true` and `task_mcp_grants: required`. The README says the
  dataset is tamper-safe only when run with it.
- `harbor_arm_harness.yaml`: the harness settings that match the native arm (`thinking`, `sandbox_model_id`,
  `argv_prompt_quoting`, `max_output_tokens`, permissions), passed to `gym eval run` only. The pilots kept these two
  files apart: an agent block in a `gym dataset validate` config was never run.
- `native_arm.yaml`: `workspace_root`, `HOME`, model routing, `use_absolute_ip`, `token_id_capture`, the `extra_body`
  sampling ([native-baseline.md](native-baseline.md)).

Adapter tests: two runs byte-identical, a non-empty output refused, each native-setting refusal, the id map matching
`--native-rows-out`, the vendored grader equal to the server's on fixed cases or byte for byte, with cases whose
rewards cover 0, 1 and any partial credit, the instruction equal to the native prompt with no hidden value in it or
in `environment/`, and a generated Dockerfile's embedded payload decoded back (`reasoning_gym/tests/` has each).
Remove a generated module from `sys.modules` (`monkeypatch.delitem`) before importing it, so a test does not reuse a
cached copy. They need `nemo_gym` and the server's dependencies in one interpreter: run them with
`gym env test --resources-server <server>`, which builds the server's venv and runs `pytest` in the server folder,
collecting every `test_*.py` under it. When you mutation-test in that venv, its editable `nemo_gym` finder resolves
`resources_servers.*` to the checkout even when a mutated copy comes first on `PYTHONPATH`: print the mutated
module's `__file__` before you trust a result.

## Vendoring a grader

Copy grading code unchanged. Three shapes occurred:

- **Module functions with no `nemo_gym` import**: copy the file.
- **Functions and server methods in an `app.py` that imports `nemo_gym`** (math): generate the file at conversion
  time with `ast`. `grader_source()` in `math_with_judge/harbor/adapter.py` copies the named module functions and
  the named methods of the server class, by source segment, into a stand-in class of the same name, and fails when
  a name is gone. The copied functions then call `LibraryJudgeMathResourcesServer.<method>` as before.
- **A method that reads Responses objects by attribute** (`response.output[i].type`, `item.content[0].text`): a
  Python verifier receives `response` as a dict. Copy the method into the stand-in class and give it an attribute
  view of the dict through a short shim, recorded as glue in the differences list:
  ```python
  from types import SimpleNamespace


  def as_attributes(value):
      """Dicts become attribute objects, recursively; the grading logic itself is not touched."""
      if isinstance(value, dict):
          return SimpleNamespace(**{key: as_attributes(item) for key, item in value.items()})
      if isinstance(value, list):
          return [as_attributes(item) for item in value]
      return value
  ```
  Check that the grader calls no pydantic method (`model_dump`, `model_copy`) on the object; if it does, restate that
  call in the shim and record it. Prove the copy with fixed cases graded by the server's own method and the copy.

## Text answer graded from the response (math_with_judge)

Pilot: `resources_servers/math_with_judge/harbor/`, converting `environments/opencode_math` (DAPO-Math-17k rows,
`math_with_judge` with the judge off, OpenCode as a local CLI on the agent-server host).

| Native | Task |
| --- | --- |
| Rows with one user message; `question`, `expected_answer` | Parameterized template + `tasks.jsonl`; `task_data.py` = the fields the server's verify request reads |
| Local CLI: one argv message `f"{system_prompt}\n\n{user}"` | `instruction.template.md` = the config's `system_prompt` (read from the YAML, trailing newline kept) + `\n\n{{ task_data.question }}` |
| No image (host) | Overlay `environment/Dockerfile`: `FROM` a public Python image pinned by digest + one `RUN uv pip install --system --no-cache <grader deps at uv.lock versions>` |
| `verify()` joins the `output_text` of every assistant message, then last-`\boxed{}` gate + math-verify in a forked child with a timeout | `tests/verifier.py` (`verify(task_data, response)`) repeats only that glue; `tests/math_with_judge_grader.py` is generated: the adapter copies the grading functions out of `app.py` with `ast`, with app.py's sha256 in the header, and writes the config's timeout beside them |
| Reference answer | `solution/answer.template.md` = `\boxed{{{ task_data.expected_answer }}}` |
| Verifier on the server host | Separate verification through `artifacts = ["/logs/artifacts"]` in `task.toml` + the server option `separate_verifier_for_artifacts: true` |

Rules this recipe adds:

- A Python verifier that reads `response` (the agent's final Responses API response) grades exactly what native
  `verify()` grades. Check whether native joins all assistant messages or only the last. Such tasks run under Gym
  only: `gym dataset publish --harbor` refuses them.
- Generate the vendored grader from the server's source at conversion time instead of hand-copying it; a rename then
  fails the adapter loudly. Refuse settings the copy does not reproduce (judge on, format-tolerant extraction).
- A task-level `environment_mode = "separate"` is refused for an overlay image without a prebuilt `docker_image`, so
  the server option is the only registry-free route to separate verification.
- Native oracle and negative controls: `fake-model` behind `policy_model` ([native-baseline.md](native-baseline.md#model-free-native-runs)).
- Probes: P1, P4 (PyPI from both sandboxes).

Run configs from the pilot (every harness key was found by diffing captured first requests). `harbor_arm.yaml`:

```yaml
harbor_environment_server:
  environment_servers:
    harbor:
      separate_verifier_for_artifacts: true
```

`harbor_arm_harness.yaml`:

```yaml
opencode_agent:
  responses_api_agents:
    opencode_agent:
      thinking: false                          # native value; the session default is true
      sandbox_model_id: ${policy_model_name}   # OpenCode picks its system prompt by model id; default "dummy_model"
      argv_prompt_quoting: true                # the local CLI quotes the argv prompt; the session uses stdin
      max_output_tokens: 32000                 # the local path leaves OpenCode's 32000 default
      opencode_config:
        permission: {bash: allow, edit: allow, webfetch: allow, skill: allow}  # native's block; skill undoes the default deny
```

## SWE: per-row image, patch capture, hidden tests (scale_swe, swe_rebench)

Pilots: `resources_servers/scale_swe/harbor/` (Python/pytest images) and `resources_servers/swe_rebench/harbor/`,
converted from this skill alone as a dry run (one image per row, 7 languages). Start a new SWE server from
`swe_rebench/harbor`: it also covers images without Python and a seed that calls the server's own code. Both
adapters grew out of an earlier SWE rows-to-Harbor converter script (pull request #5789) and are the SWE starting
point; [reward-safety.md](reward-safety.md) has the failure modes a SWE task folder must close.

| Native (`app.py`, episode session) | Task folder (one per row) |
| --- | --- |
| Seed starts the row's image with the config's resources and `env` | Overlay `environment/Dockerfile`: `FROM <image>@<digest>`, `RUN` 1 unpacks a deterministic base64 tar.gz into `/var/lib/nemo_gym`, `RUN` 2 runs its `setup.sh`; `cpus`, `memory_mb`, `storage_mb`, `env` from the server config on both sandboxes; `build_timeout_sec` = sum of the seed step limits + unpack (not the 600 default) |
| Seed steps (anti-cheat scrub, git identity, untracked-file list) with their limits, tolerance and retries | `setup.sh` finds a Python (`python.sh`) and runs `seed.py`, which calls the server's vendored seed functions unchanged; a step the server treats as fatal exits non-zero (a loud seed failure) |
| The row's user message | `instruction.md`, byte for byte (Gym reads it with LF; Step 5) |
| `capture_model_patch` against the row's base commit | `[[verifier.collect]]` runs `capture.py` (vendored `swebench/patch_capture.py` through `local_sandbox.py`); the base commit is on the hook's command line in `task.toml`, never in the agent's sandbox |
| Grading in a fresh sandbox (`verification.py`) | `environment_mode = "separate"` on the plain pinned image; `tests/test.sh` finds a Python and runs `grade.py`, which calls the vendored verification module |
| No verdict, unknown parser: masked | exit 75 without a reward |
| Golden-patch mode | `solution/solve.sh` runs `git apply` with the eval script's flags on the golden patch |

Reuse unchanged:

- from `scale_swe/harbor/task-template/tests/nemo_gym/grade.py`: `_privilege_dropper`, `_give_root_ownership`,
  `_protect_reward_path`, `_write_output`, `_no_verdict`, `_verdict`, `_capture_failed` and their constants
  (`NO_VERDICT`, `REWARD_OWNER_UID`, `PROTECTED_DIRS`, `KEPT_CAPABILITIES`, the `prctl` numbers); the capture-status
  rule (only `ok` is graded, every other outcome is a completed 0); the overlay bundle and the TOML writer with
  `_resource_keys` in `adapter.py`;
- from `swe_rebench/harbor/task-template/`: `tests/nemo_gym/local_sandbox.py` (its `exec` honours `cwd` and it has
  `upload`, which the vendored `swebench/anti_cheat.py` needs; Scale-SWE's silently ignores `cwd`),
  `tests/nemo_gym/python.sh` and the shape of `test.sh` and `setup.sh`.

Adapt per server:

- `grade()`: mirror how the server builds its verification inputs (`_inputs`: install and test commands, parser,
  test-patch handling) and call its verification functions with the same arguments.
- `seed.py`: call the server's seed functions (swe_rebench: `apply_anti_cheat_setup(..., required=True)`,
  `prepare_git_for_commits`) rather than restating them in bash; a strict outcome check is easy to get subtly wrong.
  Restate only code that imports `nemo_gym` (swe_rebench: `_pristine_untracked_files`, one command).
- The capture's diff base: the row's base commit (swe_rebench) or a HEAD recorded at seed (scale_swe), as the server
  does.
- Data files the vendored module reads by a path relative to itself (swe_rebench's Maven `settings.xml` and Gradle
  `init.gradle`, found through `Path(__file__).parents[2]`): copy them byte for byte to that relative path.
- A grader dependency the server fetches at run time (swe_rebench's `log_parsers.py` clones SWE-rebench-V2 at a
  pinned commit into a cache under HOME): vendor the files the module imports, plus the upstream LICENSE, at that
  commit into the cache layout the module already reads, so it never clones; the adapter takes the checkout and
  checks its HEAD. Record the source, commit and license in the file headers and the PR body. For the native arm,
  point the server's cache option (`log_parser_cache_dir`) away from HOME.
- Refusals: every server setting the task does not reproduce (swe_rebench: golden-patch mode, an agent egress block,
  a `provider_options.network_policy`, `derive_cpu_env: false`, `legacy_container_setup`, `use_maven_mirror: false`,
  a resource other than cpu, memory_mib and disk_gib, a cpu that is not a whole number, `evaluation_timeout: null`),
  and every row it cannot write safely (tools or sampling, a prompt that is not one user message, an image name with
  whitespace or a control character, an instance id that repeats or is not a folder name, text that is not UTF-8).

Languages (one image per row; each language is a different toolchain):

| Language | What ran | Notes |
| --- | --- | --- |
| Python | oracle 1.0, controls 0 (scale_swe; swe_rebench 1 row) | Some rows' tests download data the sandbox cannot reach (ctapipe: native golden 0, excluded in Step 3) |
| JavaScript, TypeScript, Java, Go | oracle 1.0, controls 0 (swe_rebench, 1 row each); model smoke on JS, Java, Go | Maven hosts answer 404 or 429 to a bare root: reachable |
| Rust, Dart | oracle 1.0 only (1 row each) | Dart images have no Python: `python.sh` installs a sha256-pinned python-build-standalone release (egress to GitHub releases) |
| Any other (PHP, Ruby, C/C++, Kotlin, Scala, C#, Swift, Elixir, Lua, ...) | nothing | Run a golden sweep and the probes before trusting it |

Rules this recipe adds:

- Before choosing ids, run the server's golden-patch mode on every candidate row (Step 3): it is the SWE form of
  grading every reference answer. Rows it cannot award on your infrastructure go to the `--exclude` list.
  ```bash
  gym env start --config <golden-patch-only config> --config <provider config>
  python resources_servers/<server>/apply_golden_patch.py +training_jsonl=<rows.jsonl> \
      +output_jsonl=<out.jsonl> +concurrency=4 ++head_server.port=<head port>
  ```
  The provider config (`nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`) is not in the golden
  config. Lower the golden server's `num_workers` on a small machine. For the empty control, set `patch` to `""` in
  a copy of the rows.
- No Python 3.10+ in an image (P3 fails): `python.sh` uses the image's own when it has one, else installs the pinned
  standalone Python at seed (agent sandbox, off `PATH`) and in `test.sh` (verifier, before any agent code; a failed
  install exits 75). Record it as a difference; it needs egress to GitHub releases (P4). Bound the download so the
  worst case (attempts x `--max-time` + retry delays) stays inside the build and verifier budgets: swe_rebench uses
  `curl --retry 2 --retry-delay 10 --max-time 150` (470 s) against a 600 s share, with a unit test that parses the
  flags.
- Grep the eval output of every golden run for `Permission denied`: the dropped capabilities can break a build, and
  an image built as another user needs `_give_root_ownership` (beets: 107 entries owned by `testuser`). None in the
  7 swe_rebench golden runs, one per language.
- The verifier's install step (`npm ci`, `pip install -e .`, `mvn compile`) runs agent-written code before any test,
  inside the same privilege drop. Probe it per language: `controls --plant-conftest` (pytest) and `--plant-hooks`
  (npm lifecycle scripts, a Go test `init()`, a `conftest.py`); Maven plugins and Rust `build.rs` have no probe yet.
- The collect hook runs a Python the agent can replace (`--plant-reward`'s `python3` stub replaced it on an image
  whose Python is `/usr/local/bin/python3`: a completed 0). Natively only git runs at capture: record it.
- Dataset ships as one huge JSONL: sample rows with HTTP range requests, or the Hugging Face datasets-server
  `/filter` endpoint per language, instead of downloading the file.
- Probes: P2, P3 and P4 per image family or language, in one combined dataset
  ([gates.md](gates.md#infrastructure-probes-p1-p6)).

## Tools with per-session state (workplace_assistant)

Pilot: `resources_servers/workplace_assistant/harbor/`: the tools run in a Compose sidecar. Tools in the agent's own
image would need a registry; no adapter builds that shape.

| Native | Task (parameterized template) |
| --- | --- |
| `seed_session` builds fresh tables per session id | One Compose service `tools` per episode: one fresh process = one fresh state |
| Tools served by the Resources Server (catch-all route, MCP exposure) | `environment/tool_server.py`: low-level MCP server, stateless streamable HTTP, JSON responses, the same 27 schemas; it restates the 8-line route handler and Gym's MCP rendering; the tool modules and data are copied unchanged under their original package path |
| `verify()` replays the response's `function_call` items with `is_correct` | The tool server appends every call (name + JSON arguments, fsync'd, under a lock) to a journal **before** running it, unknown and failing calls included; a separate verifier gets the journal as an artifact copied from service `tools` and runs the vendored `is_correct` |
| `[system, user]` input, `tools`, `parallel_tool_calls`, `temperature` | `instruction.md` = system text + blank line + user text; tools only over MCP; temperature through `extra_body` on both arms |
| Ground truth | `tests/ground_truth.json` (verifier) and `solution/calls.json` (oracle replays them through the MCP server with a stdlib JSON-RPC client) |

Key `task.toml` lines (from the pilot template):

```toml
artifacts = [{ source = "/var/lib/wa-journal/calls.jsonl", service = "tools" }]

[verifier.environment]
docker_image = "<public image>@sha256:<digest>"
workdir = "/tmp"            # must exist in the image: Gym does not create it

[[environment.mcp_servers]]
name = "<Resources Server instance name>"   # so the model sees the same <server>_<tool> names as natively
transport = "streamable-http"
url = "http://tools:8000/mcp"
```

Deployment settings, on every `gym dataset validate` and `gym eval run` line of the Harbor arm (G2, G3, G5 and the
full run): `sandbox.opensandbox.networking.enabled: true` and
`harbor_environment_server.environment_servers.harbor.task_mcp_grants: required`, in `harbor/configs/harbor_arm.yaml`.

Rules this recipe adds:

- Tool code must be deterministic (no wall clock, randomness or network) and importable without `nemo_gym`.
- The tool server serves `/mcp` only: no route that reads, resets or grades the journal.
- Grade from the journal, not the harness transcript: grading then does not depend on how a harness names or
  reports calls. Journal before execution so a call that crashes the tool still counts, as natively.
- `test.sh` exits 75 only when the journal is missing or the pinned install fails (infrastructure).
- Negative controls: `--solution wrong` (one state-changing call no row asks for) must score 0. "No call" scores 1
  wherever the ground truth is empty (13% of workplace rows), as natively: it is not a valid negative control there.
- Not covered: simulated users, judges inside the server, cohort rewards, external or very large state (see
  SKILL.md Scope).
- Probes: P1, P4 (PyPI from the sidecar and the verifier), P5.

## Pinning

Pin what each sandbox imports to the Resources Server's `uv.lock`, transitive packages included, with hashes. The
lock exports nemo-gym's whole tree, so use it as constraints only. `uv pip compile` resolves from each package's own
metadata, so it does not apply the server's `[tool.uv]` `override-dependencies` or `exclude-dependencies` (most
servers have one): copy each list into a file, one requirement per line, and pass it.

```bash
uv export --frozen --no-hashes --no-emit-project --no-emit-package nemo-gym \
    --project resources_servers/<server> > constraints.txt
# requirements.in: one line per package the sandbox imports directly
# overrides.txt / excludes.txt: the server's [tool.uv] override-dependencies / exclude-dependencies, if any
uv pip compile requirements.in -c constraints.txt [--overrides overrides.txt] [--excludes excludes.txt] \
    --no-build --generate-hashes --python-version 3.13 --python-platform x86_64-manylinux_2_28 -o requirements.txt
```

`--no-build` in the compile finds packages with no wheel before an episode does: it fails naming one ("has no usable
wheels"). To list them all at once, read the lock (packages with an `sdist` and no `wheels`), then keep the ones
your compiled `requirements.txt` names:

```bash
python -c 'import sys, tomllib; [print(p["name"], p["version"]) for p in tomllib.load(open(sys.argv[1], "rb"))["package"] if p.get("sdist") and not p.get("wheels")]' resources_servers/<server>/uv.lock
```

Example: `reasoning_gym`'s `reasoning-gym` 0.1.25 needs `cellpylib`, `pycosat` (a C extension) and `zss`, none with a
wheel, so a `--no-build` install fails. The ways out, each a numbered difference (seed time, image):

- Build the named packages only: replace `--no-build` with `--only-binary :all: --no-binary <pkg> ...` in both the
  compile and the install, on an image with a compiler and headers (`ghcr.io/astral-sh/uv:python3.13-bookworm`, not
  `-slim`). The reasoning_gym dry run did this and its G2 installed in both sandboxes. It adds seed time and needs
  egress to the distribution's index.
- A prebuilt image with the packages installed, pinned by digest (needs a registry).

Install with `uv pip install --system --quiet --require-hashes --no-build -r requirements.txt` (an image with `uv`,
such as `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`, pinned by digest). A hashed file lists every wheel of each
pinned version (reasoning_gym: 50 KB), and overlay mode has no `COPY`, so the adapter embeds it gzipped:

```python
packed = base64.b64encode(gzip.compress(requirements.read_bytes(), mtime=0)).decode()
dockerfile = (
    f"FROM {BASE_IMAGE}\n"
    f"RUN mkdir -p /opt/nemo_gym && echo {packed} | base64 -d | gunzip > /opt/nemo_gym/requirements.txt\n"
    "RUN uv pip install --system --no-cache --quiet --require-hashes --no-build -r /opt/nemo_gym/requirements.txt\n"
)
```

Then compare names and versions with the server's own venv (`<server venv>/bin/python -m pip freeze`) and review
packages present there but missing from your file: `workplace_assistant` needed pyarrow, which the server got through
`datasets` and which changes pandas string storage. `resources_servers/workplace_assistant/harbor/pin_requirements.py`
is a worked example that also refuses any version not taken from the lock.

Pin every image by digest. The digest of a public tag can be read before any run; this is the form the math pilot
pins (the tag's index digest):

```bash
# ghcr.io; for Docker Hub use https://auth.docker.io/token?service=registry.docker.io&scope=repository:<repo>:pull
# and https://registry-1.docker.io/v2/<repo>/manifests/<tag> (official images are library/<name>)
token=$(curl -fsS "https://ghcr.io/token?scope=repository:<repo>:pull" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')
curl -fsSI -H "Authorization: Bearer $token" -H "Accept: application/vnd.oci.image.index.v1+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
    -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "https://ghcr.io/v2/<repo>/manifests/<tag>" | grep -i docker-content-digest
```

The last two types cover a single-platform tag. For one image per row, the
[swe_rebench README](../../../../resources_servers/swe_rebench/harbor/README.md#inputs) has a script that writes the
`--image-digests` file from Docker Hub.

## Known deviations of the pilot adapters

All five adapters (`math_with_judge`, `scale_swe`, `workplace_assistant`, `reasoning_gym`, `swe_rebench`) take
`--rows`, `--output-dir`, `--ids-file`, `--native-rows-out` and `--id-map-out`, and ship the three run configs,
`tests/test_harbor_adapter.py` and `tests/__init__.py`. Each ships `configs/parity_ids.txt` except `swe_rebench`,
which had no 50-task parity run. Still different:

- `workplace_assistant` takes no `--native-config` (it reproduces one fixed pairing), and its id map is keyed by the
  rows' `id` field (`harbor_parity --native-key id`). The native OpenCode arm reads `--opencode-rows-out`, not
  `--native-rows-out`, through `harbor/configs/native_pairing.yaml`. That config is not a supported pairing: it needs
  the session and MCP name fixes that are not on main ([native-baseline.md](native-baseline.md#known-gaps-on-main)).
- `scale_swe` and `workplace_assistant` also take `--limit` (the first N selected rows). Select gate datasets with
  `--ids-file` only.
- `swe_rebench` requires `--parser-repo` (the vendored log parser's upstream checkout) and takes `--image-digests`.
- `math_with_judge`'s overlay pins its grader packages by version only, without hashes, and
  `antlr4-python3-runtime==4.9.3` has no wheel, so the hashed `--no-build` install above would refuse it (see the ways
  out for packages with no wheel).
