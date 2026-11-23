# SWE-rebench-V2 as Harbor task folders

`adapter.py` converts SWE-rebench-V2 rows (the JSONL `../prepare_swe_rebench.py` writes, or `../data/example.jsonl`)
into one Harbor task folder per row, so the same tasks run through the `harbor` Environment Server instead of
`single_agent_turn` with the SWE-rebench Resources Server (`../configs/swe_rebench.yaml`, `opencode_agent`). It is
built on the Scale-SWE adapter (`resources_servers/scale_swe/harbor`).

Run every command from the repository root, with the Python environment Gym is installed in. The adapter needs
PyYAML (installed with Gym) and `git` on `PATH`. The tools have their own READMEs:
[conversion checks](../../../scripts/harbor_conversion_checks/README.md) and
[parity comparison](../../../scripts/harbor_parity/README.md). The task-folder format is documented in
[`fern/versions/latest/pages/data/task-folders/`](../../../fern/versions/latest/pages/data/task-folders/).
`<data>` below is a folder of your choice and `<provider>` your sandbox provider.

## Inputs

- **Rows** (`<rows.jsonl>`): `python resources_servers/swe_rebench/prepare_swe_rebench.py` writes every supported
  row to `resources_servers/swe_rebench/data/swe_rebench_training.jsonl`. It needs the `datasets` package and
  streams the whole Hub split. `../data/example.jsonl` holds 5 rows only.
- **Upstream log parsers** (`--parser-repo`): a checkout at the server's pinned commit
  (`log_parsers.UPSTREAM_COMMIT`). The server's own cache clone
  (`<log_parser_cache_dir>/SWE-rebench-V2-c71902a8cf8d`) works as well. To fetch it:

  ```bash
  git init -q <data>/SWE-rebench-V2 && git -C <data>/SWE-rebench-V2 fetch -q --depth 1 \
      https://github.com/SWE-rebench/SWE-rebench-V2.git c71902a8cf8d2b725f63d51f199f4d3e56f68d2d \
      && git -C <data>/SWE-rebench-V2 checkout -q FETCH_HEAD
  ```

- **Image digests** (`--image-digests`): read the digest of each selected row's image from Docker Hub, where every
  SWE-rebench-V2 image lives (`docker.io/<repo>:<tag>`). This writes them for the ids in `<data>/smoke_ids.txt`
  (see Gates):

  ```bash
  python - <rows.jsonl> <data>/smoke_ids.txt <data>/image_digests.json <<'EOF'
  import json, sys, urllib.request
  rows, ids, out = sys.argv[1:]
  want = {line.strip() for line in open(ids) if line.strip() and not line.strip().startswith("#")}
  images = sorted({row["image_name"] for row in map(json.loads, open(rows)) if row["instance_id"] in want})
  kinds = ["oci.image.index.v1", "docker.distribution.manifest.list.v2"]
  kinds += ["oci.image.manifest.v1", "docker.distribution.manifest.v2"]
  accept = ", ".join(f"application/vnd.{kind}+json" for kind in kinds)
  digests = {}
  for image in images:
      repo, tag = image.removeprefix("docker.io/").rsplit(":", 1)
      auth = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
      token = json.load(urllib.request.urlopen(auth))["token"]
      headers = {"Authorization": f"Bearer {token}", "Accept": accept}
      url = f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
      response = urllib.request.urlopen(urllib.request.Request(url, headers=headers, method="HEAD"))
      digests[image] = response.headers["Docker-Content-Digest"]
  json.dump(digests, open(out, "w"), indent=2, sort_keys=True)
  EOF
  ```

## Convert

| Flag | Meaning |
| --- | --- |
| `--rows` | Required. SWE-rebench-V2 rows, streamed. |
| `--output-dir` | Required. Must be absent or empty. |
| `--parser-repo` | Required. A SWE-rebench-V2 checkout at `log_parsers.UPSTREAM_COMMIT` (the command above, or the server's cache clone). |
| `--ids-file` | Instance ids to convert, one per line, `#` comments, order kept (for example the output of `python -m scripts.harbor_conversion_checks subset`). |
| `--native-rows-out` | The selected raw rows, in the same order: the native arm's input. |
| `--id-map-out` | JSON `{line index in --native-rows-out: task id}`. Task ids are the instance ids. |
| `--native-config` | The native server settings: `configs/swe_rebench_legacy_agent.yaml` (default), a pairing YAML such as `configs/swe_rebench.yaml` (its `config_paths` are merged in) or a `gym env resolve` dump. |
| `--image-digests` | JSON `{image_name: "sha256:<hex>"}` (see Inputs); each task names `image_name@digest`. Without it the tasks name the rows' tags. |
| `--agent-timeout-sec` | The agent budget (default 10800 s, opencode_agent's own). |

Exit codes: 0 done; 2 a native setting or input the dataset cannot reproduce. Exit 2, with nothing written, for:

- the output folder is not empty; `--parser-repo` is not at the server's pinned commit; `--native-config` is not a
  YAML mapping or has no `swe_rebench_resources_server.resources_servers.swe_rebench` block;
- a duplicate or missing id in `--ids-file`, or an `instance_id` that appears more than once among the selected rows;
- a malformed row: an `instance_id` that is not a folder name (letters, digits and `._-`, not starting with `.`), an
  `image_name` that is empty or holds whitespace or a control character, an `image_name` missing from
  `--image-digests` or whose digest there is not `sha256:<64 hex>`, text that is not valid UTF-8,
  `responses_create_params` that is not an object, `tools`, `parallel_tool_calls`, `temperature`, `top_p` or
  `instructions` set, an input that is not exactly one user message with string content, a prompt whose first line
  Gym would drop as a canary marker;
- every native setting the task folders cannot reproduce: golden-patch mode (`is_verifying_golden_patch: true`), the
  agent egress block (`enforce_agent_no_network: true`), `sandbox_config.provider_options.network_policy`,
  `sandbox_config.derive_cpu_env: false`, `sandbox_config.legacy_container_setup`,
  `sandbox_config.use_maven_mirror: false`, a resource other than `cpu`, `memory_mib` and `disk_gib`, a `cpu` that is
  not a whole number, `evaluation_timeout: null`, and an unknown `patch_capture_mode`.

Each dataset gets a `README.md` with the rows' sha256, the server settings, each vendored file's sha256 and the
regeneration command. Two conversions of the same inputs are byte-identical (G1 below). `gym dataset validate` and
`gym eval run` write `compose-images.json`, `compose-images.json.lock` and `manifest.toml` into the dataset folder:
run them on a copy if the folder must stay byte for byte as converted.

## How a row maps to a task

| Resources Server (`app.py`, episode session) | Task folder |
| --- | --- |
| Seed starts `image_name` with the config's resources and `GRADLE_USER_HOME` | `environment/Dockerfile` in overlay mode (`FROM image@digest` + 2 `RUN` lines, nothing built); `cpus`, `memory_mb`, `storage_mb`, `env` from the server config on both sandboxes; the harbor server adds the same CPU-count hints |
| `apply_anti_cheat_setup(required=anti_cheat_required)`, `prepare_git_for_commits`, pristine untracked files | `setup.sh` finds a Python 3.10+ (or installs a pinned python-build-standalone 3.12.12) and runs `seed.py`, which calls the vendored `swebench/anti_cheat.py` and `swebench/patch_capture.py` unchanged; a scrub that did not complete fails the seed, as at the server |
| Prompt: the row's single user message | `instruction.md`, byte for byte (Gym's reader turns CRLF into LF) |
| `capture_model_patch` against the row's `base_commit` | `[[verifier.collect]]` runs `capture.py` (vendored `swebench/patch_capture.py`); the base commit is on the hook's command line, not in the agent's sandbox |
| Grading in a fresh sandbox of the image (`verification.py`, `log_parsers.py`) | `environment_mode = "separate"` on the plain image; `tests/test.sh` runs `tests/nemo_gym/grade.py`, which calls the vendored `verification.py` and `log_parsers.py`; the upstream parser tree (MIT) sits at `tests/nemo_gym/parser_cache/SWE-rebench-V2-c71902a8cf8d/`, where `resolve_parser` finds it without cloning; the Maven and Gradle mirror files come from `responses_api_agents/swe_agents/maven_mirror/` |
| No verdict, unknown parser: masked | `test.sh` exits 75 without a reward for an unknown parser or a missing repository directory (both before any agent code runs); a run without a verdict after the tests started is a completed 0 (difference 14) |
| Golden-patch mode | `solution/solve.sh` applies `patch` with the eval script's `git apply` flags |

Vendored Python modules carry a one-line header with their source path, the commit and the sha256 of the original;
the rest of the file is unchanged. Data files the server writes into sandboxes as they are (`anti_cheat_setup.sh`,
`settings.xml`, `init.gradle`, upstream `LICENSE`) are copied byte for byte; their sha256 is in the dataset README.
Re-convert when any of them changes.

The upstream files (`LICENSE`, `lib/__init__.py`, `lib/agent/__init__.py`, `lib/agent/log_parsers.py`,
`lib/agent/swe_constants.py` of SWE-rebench/SWE-rebench-V2 at `c71902a8cf8d2b725f63d51f199f4d3e56f68d2d`, MIT) are
copied from `--parser-repo` into each generated dataset, with the upstream `LICENSE` beside them; none is committed to
this repository.

## Gates

Run them in this order. gym writes its results under `results/harbor/` in the directory it runs in, named after the
dataset folder.

**Smoke subset.** No parity ids file ships with this adapter. The smoke set is the 5 tasks of Evidence that have
every control (JavaScript, Java, Go, Python, TypeScript), converted as their own dataset:

```bash
printf '%s\n' vuejs__eslint-plugin-vue-2334 raml-org__raml-java-parser-155 mgechev__revive-477 \
    tox-dev__pipdeptree-279 syuilo__aiscript-257 > <data>/smoke_ids.txt
```

Then write `<data>/image_digests.json` for them (Inputs). Their rows come from the `prepare_swe_rebench.py` output;
`data/example.jsonl` holds only the last two. The adapter has no `--limit`: select tasks with `--ids-file`. Do not
pick smoke tasks with `gym eval run --limit` either: it applies to each arm's own task order, and the two arms order
their tasks differently.

**G1 determinism.** Convert twice into two parent folders, with the same file names in both:

```bash
for d in a b; do
  python resources_servers/swe_rebench/harbor/adapter.py --rows <rows.jsonl> --output-dir <data>/$d/swe_rebench_smoke \
      --ids-file <data>/smoke_ids.txt --native-rows-out <data>/$d/smoke.native.jsonl \
      --id-map-out <data>/$d/smoke.id_map.json --native-config resources_servers/swe_rebench/configs/swe_rebench.yaml \
      --image-digests <data>/image_digests.json --parser-repo <data>/SWE-rebench-V2
done
diff -r <data>/a <data>/b
```

Expected: `converted 5 task(s) into ...` twice, and `diff -r` prints nothing. The dataset README records the base
names of `--rows`, `--ids-file`, `--native-config` and `--image-digests`, so differently named inputs make `diff -r`
differ, and so do differently named side outputs (`diff -r` compares them too).

**G1 static check.** This adapter writes one folder per task, so the check runs on the converted folder before any
validate. Some hidden test names are common words that also occur in the seed and capture code, so list the ones
you reviewed in an allow file:

```bash
printf '%s\n' continue standard 'without error' > <data>/allow_values.txt
python -m scripts.harbor_conversion_checks check <data>/a/swe_rebench_smoke --native-rows <data>/a/smoke.native.jsonl \
    --native-id instance_id --hidden-from resources_servers/swe_rebench/task_data.py --normalize-newlines \
    --allow-values <data>/allow_values.txt
```

Expected on the smoke rows (exit 0; the 9 `allowed:` lines that follow `hidden-values` are left out here):

```
PASS instruction: 5 task(s) equal their native prompt; 5 only with CRLF/CR read as LF (a difference to record)
PASS hidden-values: none of 4948 hidden value(s) of 5 row(s) in instruction.md or environment/; 183 short value(s) not scanned: use --template-hidden; 9 match(es) of --allow-values values, listed under 'allowed'
SKIP hidden-template: no --template-hidden
PASS secrets: no secret pattern in 97 file(s)
PASS local-paths: no local path in 97 file(s)
check: 4 passed, 0 failed, 1 skipped
```

Each `allowed:` line names a match inside the base64 seed bundle of `environment/Dockerfile`, in the seed and
capture code, not row data. Without `--allow-values` these 9 matches fail the check. On other rows, read each match
before you add its value. `--normalize-newlines` is needed because these prompts hold CRLF (difference 1).
`hidden-template` is skipped, and the 183 short values cannot be scanned another way, because there is no template.

**G2 oracle.** Every reference solution must score 1.0:

```bash
C=resources_servers/swe_rebench/harbor/configs
gym dataset validate <data>/a/swe_rebench_smoke --sandbox <provider> --config $C/harbor_arm.yaml --concurrency 4
```

Read `results/harbor/validation/swe_rebench_smoke/rollouts_run_summary.json`: a pass is `counts.masked` 0,
`counts.failed` 0, every `tasks[].reward` 1.0 and every `tasks[].verifier.mode` `separate`. These 5 tasks took
205 s (34 to 94 s per task).

**G3 negative control.** Build it from the second, never-validated conversion, into a folder that must not exist:

```bash
python -m scripts.harbor_conversion_checks controls <data>/b/swe_rebench_smoke <data>/swe_rebench_smoke_empty --empty
gym dataset validate <data>/swe_rebench_smoke_empty --sandbox <provider> --config $C/harbor_arm.yaml --concurrency 4
```

`gym dataset validate` exits 1 on a control by design, and its table labels every row "oracle solved" even at
reward 0. So read `results/harbor/validation/swe_rebench_smoke_empty/rollouts_run_summary.json`: a pass is every
`reward` 0.0, every `mask_sample` false, `counts.masked` 0 and `counts.failed` 0 (208 s for these 5 tasks). Build
the reward-forgery probes (`--plant-reward`, `--plant-conftest`, `--plant-hooks`) the same way: each must also score
0 unmasked. A probe's row can show `failure_kind` `harbor:missing_reward`: the grader left no reward, which scores 0.

## Run both arms

Supply your own sandbox provider (`<provider>`), model flags (the same on both arms) and absolute paths. Before
choosing ids, run the native golden-patch mode on the candidate rows: a supported row can score 0 natively (see
Evidence), and such a row belongs in neither arm. Run the gates above on the Harbor dataset first.

```bash
C=resources_servers/swe_rebench/harbor/configs

# Harbor arm
gym eval run <dataset dir> --agent opencode_agent --sandbox <provider> \
    --config $C/harbor_arm.yaml --config $C/harbor_arm_harness.yaml <model flags> \
    --num-repeats 2 --concurrency 4 --output <harbor run>/rollouts.jsonl \
    ++observability_enabled=true ++model_call_capture_dir=<harbor run>/model_calls

# Native arm: the rows the adapter wrote with --native-rows-out, as the `train` split
gym eval run --config resources_servers/swe_rebench/configs/swe_rebench.yaml \
    --config nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml --config $C/native_arm.yaml \
    ++swe_rebench_native_rows=<native rows.jsonl> <model flags> --split train \
    --num-repeats 2 --concurrency 4 --output <native run>/rollouts.jsonl \
    ++observability_enabled=true ++model_call_capture_dir=<native run>/model_calls

# Compare
python -m scripts.harbor_parity --native <native run>/rollouts.jsonl --native-key instance_id \
    --harbor <harbor run>/rollouts.jsonl --json parity.json --markdown parity.md
```

The native server clones the upstream log parsers on first use into `~/.cache/nemo_gym/swe_rebench`; to keep it out
of the home directory, add `++swe_rebench_resources_server.resources_servers.swe_rebench.log_parser_cache_dir=<dir>`.

## Evidence

This conversion was produced as a dry run of the `convert-env-to-harbor` skill. Only smoke-size evidence exists; no
50-task parity run was made.

After the grader hardening in [Reward safety](#reward-safety) (mapped-file protection, the audit hook, the bounded
capture reader, differences 14 to 16), datasets regenerated from this revision were re-run on real sandboxes:

- Reference solutions: 7 of 7 scored 1.0 (JavaScript, Java, Go, Python, TypeScript, Rust, Dart), none masked,
  every task verified in a separate sandbox, with no "Permission denied" line that the earlier reference runs did
  not show.
- The empty control and `--plant-hooks` each scored 0, unmasked, on the 5 smoke tasks.
- A mutation control whose build and test hooks try to rewrite in place every file the grader has mapped scored 0,
  unmasked, on the 5 smoke tasks. On the 4 tasks where a hook ran (no hook covers the Java task's Maven build),
  every write to a mapped file was refused.

The model smoke and the `--plant-reward` and `--plant-conftest` probes below were not re-run with the hardened
grader. The rest of this section is the evidence from before the hardening:

- Reference solutions: 7 tasks, one per language, each scored 1.0 with `gym dataset validate`, none masked:
  `vuejs__eslint-plugin-vue-2334` (JavaScript), `raml-org__raml-java-parser-155` (Java), `mgechev__revive-477` (Go),
  `tox-dev__pipdeptree-279` (Python), `syuilo__aiscript-257` (TypeScript), `taiki-e__cargo-hack-70` (Rust) and
  `intel__rohd-458` (Dart). After the standalone-Python download was bounded (Requirements), the
  same 7 tasks were converted again and re-run: again 1.0 each, none masked, every task verified in a separate
  sandbox; the Dart task, whose image has no Python, ran its seed and grader on the downloaded Python.
- Negative controls (empty solution, planted reward file, planted `conftest.py`, and a probe planting build- and
  test-time hooks for several languages): each scored 0 without a mask on the JavaScript, Java, Go, Python and
  TypeScript tasks only. No control ran on the Rust or Dart task. The artifact probe does not cover Java/Maven pom
  plugins or Rust `build.rs`.
- Model smoke: 3 tasks (JavaScript, Java, Go) with `opencode_agent` and the same model on both arms gave identical
  per-task rewards (1, 0, 0), and the first model request matched on all 3 once CRLF was normalized (difference 1).
- Native golden pre-check: of 8 supported rows, `cta-observatory__ctapipe-2397` scores 0 under the server's own
  golden-patch mode, because its tests download data from an external host the sandbox could not reach. It was left
  out; run the native golden-patch mode before choosing ids.

## Differences from the native path

Each with its expected effect on reward:

1. **Prompt newlines.** Natively the row's text reaches the model with CRLF where the row has them; Gym's
   `instruction.md` reader gives LF. No reward effect expected; a first-request comparison needs newline
   normalization.
2. **Python in the agent's sandbox.** An image without Python 3.10+ (the Dart images) gets a standalone Python in
   `/var/lib/nemo_gym/python`, not on `PATH`; other images are unchanged. Negligible.
3. **Seed bundle in `/var/lib/nemo_gym`, visible to the agent** (seed and capture code, the pristine untracked list;
   no hidden data). The server keeps that list in memory. Tampering only makes the agent's own capture fail (0).
4. **A failed capture is a completed 0** (`git_error`, `infra_error`, `runner_failed`, missing), with no test run.
   The server grades the empty patch on a git error (0 on rows that need their golden patch) and masks an
   infrastructure error.
5. **No retry of a verification without a verdict.** The server retries once on a fresh sandbox before masking; the
   task masks at once (exit 75) when it masks at all (difference 14). Slightly more masks under flaky
   infrastructure.
6. **The eval script runs as root without the capabilities that reach the reward** (`CAP_DAC_OVERRIDE`,
   `CAP_FOWNER`, `CAP_CHOWN`, `CAP_SETUID`, `CAP_SETGID` and others; see Reward safety); files other uids own are
   handed to root first. A build that needs those capabilities behaves as for a non-root user; none of the 7
   reference runs showed "Permission denied". The files the grader has mapped (its interpreter, libpython, the
   extension modules it loaded, libc, ld.so, locale data) belong to another uid without write bits while the tests
   run: a build step that rewrites one in place fails with "Permission denied"; replacing one by rename (pip, npm,
   dpkg) still works.
7. **Grader Python**: the image's 3.10+ or the standalone 3.12 (natively the server's). The parsers are pure
   standard-library regular expressions; no effect expected.
8. **Verifier time limit**: `[verifier].timeout_sec` is `evaluation_timeout` (3600 s) + 900 s for the standalone
   Python download and the grader; `run_verification` itself keeps the server's 3600 s.
9. **A seed failure fails the episode** (a required scrub that did not complete, a failed untracked listing, no
   Python). The server fails the seed and the rollout is retried; on Harbor the harbor server's own retry and mask
   policy applies.
10. **Sandbox metadata labels** (`benchmark`, `instance_id`, `language`) are not set on Harbor sandboxes. No reward
    effect.
11. **Agent egress** is open on both paths (`enforce_agent_no_network: false` natively). Same.
12. **Metrics**: the server's verify keys (`resolved`, `patch_applied`, ...) become `reward.json` metrics; the
    server's per-language metrics are not reproduced. No reward effect.
13. **Capture interpreter**: the capture hook runs the Python the seed recorded in `/var/lib/nemo_gym/python_bin`;
    an agent that replaces that interpreter loses its capture (a measured `runner_failed`, 0). Natively the capture
    runs only git. No effect for agents that leave the system Python alone.
14. **A run without a verdict is a completed 0, not a mask.** The server masks (after one retry) an exit 97, the
    test-patch marker, a test run killed by a signal without a conclusive failure, and a parser that raised. The
    eval script's install block already runs the agent's code (its build files and package hooks) before
    `TEST_OUTPUT_BEGIN`, so every one of these can be the agent's doing: each scores 0 with `no_verdict_after_tests`
    1. Only an unknown parser and a missing repository directory, both checked before any agent code runs, mask. A
    row whose held-out test patch really does not apply scores 0 on the reference run (gate G2); leave it out.
15. **A grader failure after the tests started is a measured 0, not a mask.** An exception, a refused import or file
    access, or a changed mapped file after the eval script started exits 1 without a reward (see Reward safety). The
    server masks a verification that raised. A privilege drop that fails in the eval script's process is one of
    these: it scored a mask before, a 0 now.
16. **Capture files are bounded**: a `capture.json` over 1 MiB or a `model_patch.diff` over 64 MiB, or either one
    not a regular file (a symlink, a FIFO, a directory), is a completed 0 like any other failed capture. The server
    reads the patch with no such cap; the largest patch in the reference runs was 21 KB.

`resource_requests`, the sandbox ready timeout (1200 s) and the episode timeout (18000 s) are harbor server settings
(`configs/harbor_arm.yaml`), not task fields. The prompt (apart from difference 1), tools, sampling and agent budget
(10800 s) are the same on both paths.

## Reward safety

As in the Scale-SWE tasks (that README's "Reward safety" has the full argument; the protection code in `grade.py` is
the same, and a test keeps the two copies identical): every task verifies in a separate sandbox; the eval script
runs the agent's patched code as uid 0 without the capabilities that reach the reward folder, which `grade.py` first
hands to another uid (47311), creating the wrapper's exit record there with exit 1; the files `grade.py` has mapped
are handed to that uid without write bits, so the tests cannot rewrite in place code the grader runs; once the eval
script starts, an audit hook refuses every import, library load, `exec`, fork, further subprocess and every open but
of the outputs, and after the tests the mappings are checked again; anything that goes wrong after the tests started
exits 1 without a reward (a measured 0, never a mask); and `grade.py` writes the reward as bytes, as that uid. The
capture files are read only as regular files within their size bound, and every capture outcome but a readable
patch is a completed 0, never a mask. The dataset needs no server option for this; `configs/harbor_arm.yaml` carries
the native scheduling settings.

What the agent's code can still do:

- **Kill the grader.** Killing `grade.py` leaves no reward (scored 0); killing the wrapper around `test.sh` before it
  writes its exit record leaves the record `grade.py` created (exit 1), also a 0.
- **Mask through the wrapper's exit record (open).** A test process can lower the wrapper's file size limit, so the
  wrapper's own write empties the record and fails; the harbor server masks a run whose record is empty.
- **Rewrite the reward after `grade.py` exits (open).** The harbor server then runs `date`, `tar` and `gzip` in this
  sandbox as root with every capability, found through `PATH`, and the tests own those programs and folders: a test
  that replaces one with a script rewriting `reward.json` forges the score.
- **Hold the grader with a detached process.** A test process that starts its own session (`setsid`) and keeps the
  output pipe open holds `grade.py` until `[verifier].timeout_sec` (3600 + 900 s), which leaves no reward and scores
  a measured 0.
- **Forge the test output.** The verdict is read from the eval script's output, as at the server: a test hook that
  reports failing tests as passed scores 1 on both arms.
- **Mask through the sandbox provider (open).** Test code can end the verifier without an exit status, which the
  harbor server masks. The Scale-SWE README gives the two routes measured on real sandboxes: rewriting a shared
  library the server's later `bash` loads, and `kill -9 -1`.
- **Reach `grade.py` below Python.** Library loads from C (name service, `iconv`, unwinding) are not audited; the path
  after the tests makes none, and one would show as a new mapping in the check.

The Scale-SWE README says how the two open routes were shown. Closing them and the provider mask needs the reward
read and the exit status recorded outside the verifier sandbox (a harbor server change).

## Requirements and known limits

- Python 3.10+ in the image for the seed, the capture hook and the grader, or egress to GitHub release downloads for
  the pinned standalone Python (Dart images have no Python). The standalone Python lands in
  `/var/lib/nemo_gym/python` in the agent's sandbox (not on `PATH`) and `/opt/nemo_gym_verifier` in the verifier's.
  It is pinned for x86_64 only: on another architecture an image without Python 3.10+ fails its seed in the agent's
  sandbox and masks in the verifier. A failed download or sha256 check does the same: `setup.sh` exits 1 (the seed
  fails) and `test.sh` exits 75 (masked).
- The download is bounded to fit its 600 s share of the seed's `build_timeout_sec` and of the 900 s verifier margin:
  curl makes at most 3 attempts of 150 s each with 10 s between them (worst case 470 s), and the wget fallback is
  stopped after 470 s. An archive too slow to arrive in that time counts as a failed download.
- The standalone Python (python-build-standalone release 20251014, CPython 3.12.12) is downloaded at run time into
  the sandbox and is not vendored: CPython under the PSF license plus the third-party libraries the distribution
  bundles, built by astral-sh/python-build-standalone (MPL-2.0 tooling).
- Without `CAP_CHOWN` or `CAP_SETPCAP`, `grade.py` cannot protect the reward and masks the sample before any test
  runs; the same holds when it cannot hand a file it has mapped to the reward uid (a read-only filesystem counts as
  protected). A privilege drop that fails in the eval script's own process, after it started, scores 0. Without a
  root that can change its effective uid (`CAP_SETUID`, and uid 47311 usable), `grade.py` cannot write any output:
  every episode, the oracle included, ends with no reward and scores 0, so gate G2 shows it at once.
- `capture.py` writes the patch and its JSON file, and `grade.py` writes the eval script and the patch into the
  verifier sandbox, with Python's default encoding, which follows each image's locale (UTF-8 under a UTF-8 or C
  locale). The images span seven languages; under another locale a patch with non-ASCII text can be altered, which
  can change its reward. `grade.py` reads the capture as UTF-8 and writes its outputs as UTF-8 bytes. A locale whose
  codec would be loaded after the tests is loaded before them; an XML report in an encoding whose codec is not yet
  loaded makes the parser load it after the tests, which is refused and scores 0, never 1.
