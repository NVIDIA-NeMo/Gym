# Scale-SWE as Harbor task folders

`adapter.py` converts Scale-SWE rows (the JSONL `../prepare_scale_swe.py` writes) into one Harbor task folder per
row, so the same tasks run through the `harbor` Environment Server instead of `single_agent_turn` with the Scale-SWE
Resources Server. It was piloted with OpenCode (`opencode_agent`) on both paths and matched the native score (see
[Parity](#parity)). It is adapted from the SWE rows-to-Harbor converter PR (#5789), whose scale_swe conversion it
supersedes.

Run every command below from the repository root, with the Python environment Gym is installed in (the adapter
also needs PyYAML, which Gym installs). The tools have their own READMEs:
[conversion checks](../../../scripts/harbor_conversion_checks/README.md) and
[parity script](../../../scripts/harbor_parity/README.md). The task-folder format is documented in
[`fern/versions/latest/pages/data/task-folders/`](../../../fern/versions/latest/pages/data/task-folders/).

## What the conversion does

| Native (`app.py`, episode session) | Task folder (one per row) |
| --- | --- |
| The seed starts the row's `image_url` with the config's resources | `environment/Dockerfile` in overlay mode: `FROM image_url` and two `RUN` lines, nothing built. `cpus`, `memory_mb` and `storage_mb` come from `configs/scale_swe.yaml` on both sandboxes; the harbor server adds the same CPU-count hints. |
| `_seed_session`: `pre_commands`, `git init` if needed, anti-cheat scrub, git identity, base commit, untracked files | `setup.sh`, unpacked by the first `RUN` line from a deterministic bundle and run by the second, each step with the server's time limit and tolerance (a copy sits in `environment/setup.sh`). A HEAD that is not a commit fails the seed, as the server refuses it. |
| The prompt: the row's single user message | `instruction.md`, byte for byte (a CR in the prompt reaches the agent as LF; the dataset README lists affected tasks) |
| `capture_model_patch` in the agent's sandbox | A `[[verifier.collect]]` hook runs `capture.py`, which calls `resources_servers/swebench/patch_capture.py` (vendored) with the server's `patch_capture_mode` |
| Grading in a fresh sandbox of the image (`verification.py`) | `environment_mode = "separate"` on the plain image; `tests/test.sh` runs `tests/nemo_gym/grade.py`, which calls the vendored `verification.py` |
| No verdict: masked | `test.sh` exits 75 without a reward, which the harbor server masks, only where no agent code can have caused it (difference 11) |
| Golden-patch mode | `solution/solve.sh` applies the row's `patch` with the eval script's `git apply` flags |

**Hidden data.** The golden patch, `f2p_patch`, `f2p_script`, `FAIL_TO_PASS` and `PASS_TO_PASS` live only in
`solution/` and `tests/`, which the agent's sandbox never receives.

**Vendored code.** `resources_servers/scale_swe/verification.py` and `resources_servers/swebench/patch_capture.py`
are copied unchanged below a one-line header:
`Vendored unchanged from <path> at <Gym commit>; source sha256 <hex>; license Apache-2.0`.
`resources_servers/swebench/anti_cheat_setup.sh` is copied as it is. Convert again whenever one of the three changes.

## Generate the dataset

The adapter reads the server's config and modules from the source tree; outside a git checkout it records the Gym
commit as `unknown`.

```bash
python resources_servers/scale_swe/harbor/adapter.py --rows <rows.jsonl> --output-dir <dataset dir> \
    --ids-file <ids.txt> --native-rows-out <native rows.jsonl> --id-map-out <id map.json>
```

| Flag | Meaning |
| --- | --- |
| `--rows` | Required. Scale-SWE rows, streamed. |
| `--output-dir` | Required. Must be absent or empty. |
| `--ids-file` | Instance ids to convert, one per line, `#` comments, order kept (for example the output of `python -m scripts.harbor_conversion_checks subset`). |
| `--native-rows-out` | The selected raw rows, in the same order: the native arm's input. |
| `--id-map-out` | JSON `{line index in --native-rows-out: task id}`. Task ids are the instance ids. |
| `--native-config` | The native server settings: `configs/scale_swe.yaml` (default), a pairing YAML such as `configs/scale_swe_opencode.yaml` (its `config_paths` are merged in) or a `gym env resolve` dump. |
| `--limit`, `--dataset-name`, `--agent-timeout-sec` | Optional: the first N selected rows (applied to the task folders and `--native-rows-out` alike); Harbor's `org/name`; the agent budget (default 10800 s, opencode_agent's own). |

Exit 2, with nothing written, for a non-empty output folder, an `--ids-file` that is not valid UTF-8, a duplicate or
missing id, a negative `--limit`, a missing folder for `--native-rows-out` or `--id-map-out`, a malformed row (bad
JSON, an `instance_id` that is not a folder name or starts with a dot, an empty or missing `workdir` or `image_url`, an
`image_url` with whitespace or a control character, a `workdir` that is not an absolute path of letters, digits and
`._@+-/`, text that is not valid UTF-8, a prompt that is not one user message, a prompt whose first line Gym would drop
as a canary marker), and every native setting the task folders cannot reproduce: `is_verifying_golden_patch: true`,
`enforce_agent_no_network: true`, `sandbox_config.provider_options.network_policy`,
`sandbox_config.derive_cpu_env: false`, `evaluation_timeout: null`, a resource other than `cpu`, `memory_mib` and
`disk_gib`, a `cpu` that is not a whole number, and an unknown `patch_capture_mode`.

Output: one folder per task, `dataset.toml`, and a `README.md` recording the rows file (name, sha256, line count),
the selected count and ids-file sha256, the Gym commit, the server settings, each vendored file's sha256 and the
command that regenerates the folder. Two conversions of the same inputs are byte-identical.

Preparing a dataset for a run (`gym dataset validate`, `gym eval run`) writes `compose-images.json`,
`compose-images.json.lock` and `manifest.toml` into it. Run on a copy if the folder must stay as converted.

### Regenerate the parity dataset

`configs/parity_ids.txt` lists the 50 instance ids of the [parity](#parity) run, in run order. Their rows come from
the public Hugging Face dataset `AweAI-Team/Scale-SWE` in the shape `../prepare_scale_swe.py` writes (the parity run
read the dataset at revision `d8db2039`; `prepare_scale_swe.py` reads the current one):

```bash
# All 17,696 supported rows, streamed from the Hub into resources_servers/scale_swe/data/scale_swe_training.jsonl
python resources_servers/scale_swe/prepare_scale_swe.py
python resources_servers/scale_swe/harbor/adapter.py \
    --rows resources_servers/scale_swe/data/scale_swe_training.jsonl --output-dir <dataset dir> \
    --ids-file resources_servers/scale_swe/harbor/configs/parity_ids.txt \
    --native-rows-out <native rows.jsonl> --id-map-out <id map.json>
```

From the same rows, a dataset regenerated from this revision differs from the one the parity run used in
`tests/nemo_gym/grade.py`, hardened after the parity run (see [Reward safety](#reward-safety) and differences 11 to
13), and otherwise only in comments, docstrings, vendoring headers and the dataset README.

## Run both arms

Supply your own sandbox provider (`<provider>`), model flags (`--model-type <type> --model <model> --model-url <url>`,
the same on both arms) and absolute paths. The run configs are in `configs/`:

- `harbor_arm.yaml`: harbor server settings for every Harbor `gym dataset validate` and `gym eval run`.
- `harbor_arm_harness.yaml`: the OpenCode settings, for the Harbor `gym eval run` only.
- `native_arm.yaml`: what the native arm needs on top of `configs/scale_swe_opencode.yaml`: a routable model
  address for the remote sandbox, `token_id_capture` off as on the Harbor arm, and the rows file you supply.

```bash
C=resources_servers/scale_swe/harbor/configs

# Harbor arm
gym eval run <dataset dir> --agent opencode_agent --sandbox <provider> \
    --config $C/harbor_arm.yaml --config $C/harbor_arm_harness.yaml <model flags> \
    --num-repeats 2 --concurrency 8 --output <harbor run>/rollouts.jsonl \
    ++observability_enabled=true ++model_call_capture_dir=<harbor run>/model_calls

# Native arm
gym eval run --config resources_servers/scale_swe/configs/scale_swe_opencode.yaml \
    --config nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml --config $C/native_arm.yaml \
    ++scale_swe_native_rows=<native rows.jsonl> <model flags> --split train \
    --num-repeats 2 --concurrency 8 --output <native run>/rollouts.jsonl \
    ++observability_enabled=true ++model_call_capture_dir=<native run>/model_calls

# Compare
python -m scripts.harbor_parity --native <native run>/rollouts.jsonl --native-key instance_id \
    --harbor <harbor run>/rollouts.jsonl --json parity.json --markdown parity.md
```

## Gates

Run them in this order. `C` is as above, and `<data>` is a folder of your choice. gym writes its results under
`results/harbor/` in the directory it runs in, named after the dataset folder.

**Smoke subset.** The first 5 parity ids, converted as their own dataset (G1 below):

```bash
grep -v '^#' $C/parity_ids.txt | head -5 > <data>/smoke_ids.txt
```

The adapter's own `--limit 5` with `--ids-file $C/parity_ids.txt` picks the same 5, as it applies to the task
folders and `--native-rows-out` alike. Do not pick smoke tasks with `gym eval run --limit`: the two arms order their
tasks differently.

**G1 determinism.** Convert twice into two parent folders, with the same file names in both:

```bash
for d in a b; do
  mkdir -p <data>/$d   # the folder for --native-rows-out and --id-map-out must exist
  python resources_servers/scale_swe/harbor/adapter.py --rows <rows.jsonl> --output-dir <data>/$d/smoke \
      --ids-file <data>/smoke_ids.txt --native-rows-out <data>/$d/native_rows.jsonl --id-map-out <data>/$d/id_map.json
done
diff -r <data>/a <data>/b
```

Expected: `converted 5 task(s) into ...` twice, and `diff -r` prints nothing. The dataset README records the base
names of `--rows`, `--ids-file` and `--native-config`, so differently named inputs make `diff -r` differ.

**G1 static check.** This adapter writes one folder per task, so the check runs on the converted folder before any
validate:

```bash
python -m scripts.harbor_conversion_checks check <data>/a/smoke --native-rows <data>/a/native_rows.jsonl \
    --native-id _index --id-map <data>/a/id_map.json --hidden patch --hidden f2p_patch --hidden f2p_script \
    --hidden FAIL_TO_PASS --hidden PASS_TO_PASS --hidden-from resources_servers/scale_swe/task_data.py
```

Expected on the parity rows (exit 0):

```
PASS instruction: 5 task(s) equal their native prompt
PASS hidden-values: none of 1089 hidden value(s) of 5 row(s) in instruction.md or environment/
SKIP hidden-template: no --template-hidden
PASS secrets: no secret pattern in 57 file(s)
PASS local-paths: no local path in 57 file(s)
check: 4 passed, 0 failed, 1 skipped
```

`hidden-template` is skipped because there is no template. Each distinct hidden string counts as one value (every
test name in `FAIL_TO_PASS` and `PASS_TO_PASS` is one), so the count depends on the rows. On a folder that was
already validated, the file count is 60: the three files preparation writes.

**G2 oracle.** Every reference solution must score 1.0:

```bash
gym dataset validate <data>/a/smoke --sandbox <provider> --config $C/harbor_arm.yaml --concurrency 4
```

Read `results/harbor/validation/smoke/rollouts_run_summary.json`: a pass is `counts.masked` 0, `counts.failed` 0,
every `tasks[].reward` 1.0 and every `tasks[].verifier.mode` `separate`. 5 tasks took about 4 minutes (252 s).

The native golden check runs the server's golden-patch mode: start a stack from a config that includes
`configs/scale_swe.yaml` and your provider's config and sets `scale_swe_resources_server: null` (so only
`scale_swe_golden_patch_resources_server` runs), then run `python resources_servers/scale_swe/apply_golden_patch.py
+training_jsonl=<native rows.jsonl> +output_jsonl=<out.jsonl> +concurrency=4 ++head_server.port=<head port>`. It
must resolve every row.

**G3 negative control.** Build it from the second, never-validated conversion, into a folder that must not exist:

```bash
python -m scripts.harbor_conversion_checks controls <data>/b/smoke <data>/smoke_empty --empty
gym dataset validate <data>/smoke_empty --sandbox <provider> --config $C/harbor_arm.yaml --concurrency 4
```

`gym dataset validate` exits 1 on a control by design, and its table labels every row "oracle solved" even at
reward 0. So read `results/harbor/validation/smoke_empty/rollouts_run_summary.json`: a pass is every `reward` 0.0,
every `mask_sample` false, `counts.masked` 0 and `counts.failed` 0. Build and read the reward-forgery probe
(`--plant-conftest`, into `<data>/smoke_conftest`) the same way: it must also score 0. For the native negative
control, set `patch` to `""` in a copy of the rows and run the golden check above.

**G4/G5 smoke and G6 comparison**: 3 to 5 tasks on each arm with the commands in [Run both arms](#run-both-arms),
then `python -m scripts.harbor_conversion_checks first-request <native run>/model_calls <harbor run>/model_calls`.

## Parity

50 tasks (one per repository, all Python) x 2 repeats per arm, `opencode_agent` on both paths, the same
model behind an OpenAI-compatible model endpoint, the same tasks and model flags, concurrency 8.

| Path | Rollouts scored | Masked | Failed | Mean reward | 95% bootstrap interval |
| --- | --- | --- | --- | --- | --- |
| native (`single_agent_turn` + Scale-SWE Resources Server) | 100 / 100 | 0 | 0 | 0.550 | [0.420, 0.680] |
| Harbor (these task folders) | 100 / 100 | 0 | 0 | 0.570 | [0.440, 0.700] |

- Difference (Harbor - native), paired over the 50 tasks: +0.020, 95% paired bootstrap interval [0.00, +0.05],
  SE 0.014. `python -m scripts.harbor_parity`: PASS. Repeat-to-repeat SE of each arm's mean: 0.022.
- Reference solutions: 50/50 score 1.0 on both paths (strict parity PASS), with no "Permission denied" in any of the
  50 Harbor verifier logs.
- The first model request (system prompt, user message, tool definitions and every other field) was byte-identical
  on all 100 rollout pairs, and the grader reported the same FAIL_TO_PASS and PASS_TO_PASS test sets on both paths
  for every task.
- 44 of 50 tasks gave the same reward on all four rollouts; only 2 tasks had different per-task means, and reading
  their patches showed the model choosing a different fix, not a conversion effect.

Caveats from the parity reports:

- The interval's lower bound is exactly 0 and both discordant tasks favour Harbor (sign test p = 0.5). The verdict
  at the parity script's default tolerance 0 is brittle when the model is this close to deterministic: one more
  discordant task in the same direction would turn it to FAIL.
- The parity run's empty-patch control covered 5 of the 50 tasks (0 on both paths); the re-validation below ran it
  on all 50 on the Harbor arm.
- The two arms ran one after the other, so drift of the model endpoint between them cannot be ruled out.

### After the grader hardening

The 50-task model parity run above was made with the grader before the hardening described in
[Reward safety](#reward-safety) (mapped-file protection, the audit hook, the bounded capture reader, differences 11
to 13), and it was not repeated. The hardened grader was re-validated on datasets regenerated from this revision, on
real sandboxes, every task verified in a separate sandbox:

- Reference solutions: 5 of 5 on the smoke set and 49 of 50 on the parity set scored 1.0, none masked, with no new
  "Permission denied" line in any verifier log. The one 0, `ethereum_py-trie_pr110`, is a flaky PASS_TO_PASS
  property test (`test_trie_walk_root_change_with_traverse`); that task scored 1.0 on 3 of 3 re-runs, and the test
  passed in every earlier run of both arms.
- Empty patch on all 50 tasks: 49 scored 0, unmasked. `omni-us_jsonargparse_pr215` scores 1.0 with an empty
  patch, under the previous grader too, because its FAIL_TO_PASS test passes on the base commit. That is a row
  defect: on that task the reference run cannot tell a solution from none, on either arm.
- The plant controls (`--plant-reward`, `--plant-conftest`, `--plant-hooks`) and a mutation control whose tests try
  to rewrite in place every file the grader has mapped each scored 0, unmasked, on the 5 smoke tasks. Every write
  to a mapped file was refused. Against the previous grader the same mutation control's writes landed.
- Model smoke: 5 tasks with `opencode_agent` gave the same per-task rewards (1, 1, 1, 0, 1) as both parity repeats
  on both arms, none masked.

The parity result still stands. The tests run with the same identity and capabilities as before, no reference run
log shows a new permission error, and the new decision rules only change outcomes that used to be masks (a run
without a verdict, a grader failure after the tests, a capture of the wrong shape). Every rollout of the parity run
was scored on both arms: none was masked and none ended without a verdict.

## Differences from the native path

The ones that can move a reward or a count:

1. **No retry of a verification without a verdict.** Native retries once on a fresh sandbox, then masks; a task
   masks at once (exit 75) when it masks at all (difference 11). Expect a few more masked rows on the Harbor arm
   when infrastructure flakes.
2. **A bad base commit** fails the seed (the rollout fails, unscored) instead of the native retryable refusal.
3. **Seed records and capture code live in `/var/lib/nemo_gym` in the agent's sandbox** (the server keeps them in
   memory and outside the sandbox). They hold no hidden data; a missing or changed record scores 0, never a mask.
4. **`resource_requests`** (the scheduling floor) are not task fields. Add them to `harbor_arm.yaml` as
   `sandbox_provider_options` to match native scheduling; the parity run set none.
5. **`sandbox_ready_timeout_s`** is 900 s on the Harbor arm (`harbor_arm.yaml`) against the server's 1200 s.
6. **The grader runs on the image's Python** (3.9+; 3.10+ for the capture hook). An image without one masks.
7. **Capture failures score 0, not masked**: the capture hook runs where the agent controls everything, so any
   outcome but a readable patch is a completed 0. The server masks a capture that timed out or raised.
8. **Seed retries**: the server retries the anti-cheat scrub and the git identity step when the provider fails the
   exec; `setup.sh` runs each once.
9. **The eval script runs as root without the capabilities that override file permissions** (see Reward safety): a
   test that needs root to bypass permissions on a file it does not own, to change owners or to switch uid behaves
   as for a non-root user. The files the grader has mapped (its interpreter, libpython, the extension modules it
   loaded, libc, ld.so, locale data) belong to another uid without write bits while the tests run: a test that
   rewrites one of them in place (a `cp` over a system library, `localedef` updating `locale-archive`) fails with
   "Permission denied"; replacing one by rename (pip, conda, dpkg) still works.
10. **CR line endings in a prompt reach the agent as LF** through Gym's `instruction.md` reader; none in the 1,928
    sampled rows; the generated dataset README names affected tasks.
11. **A run the tests ended without a verdict is a completed 0, not a mask.** The server masks every run without a
    verdict (after one retry). Here only a run whose eval script stopped before `TEST_OUTPUT_BEGIN` (no agent code
    had run; the workdir is checked before the tests) masks. Exit 97 once the tests ran, and the test-patch marker
    anywhere in the output, score 0 with `no_verdict_after_tests` 1: the agent's code can print the marker, and so
    can its patch, whose file names `git apply` prints before `TEST_OUTPUT_BEGIN`. A row whose held-out test patch
    really does not apply scores 0 on the reference run (gate G2), not a mask; leave such rows out.
12. **A grader failure after the tests started is a measured 0, not a mask.** An exception, a refused import or
    file access, or a changed mapped file after the eval script started exits 1 without a reward (see Reward
    safety). The server masks a verification that raised. A privilege drop that fails in the eval script's process
    is one of these: it scored a mask before, a 0 now.
13. **Capture files are bounded**: a `capture.json` over 1 MiB or a `model_patch.diff` over 64 MiB, or either one
    not a regular file (a symlink, a FIFO, a directory), is a completed 0 like any other failed capture. The server
    reads the patch with no such cap; the largest patch in the parity runs was 0.67 MB.

The prompt (apart from difference 10), tools, sampling and agent budget (10800 s) are the same on both paths.

## Reward safety

At the server the verdict is computed outside every sandbox. In a task folder the verifier sandbox runs the agent's
patched code (its tests, a `conftest.py` it adds) as root, and Harbor reads `/logs/verifier/reward.json` whatever
`test.sh` exits with. So `grade.py` protects the reward itself:

- before the tests, it moves ownership of every root-filesystem entry another uid owns to root (group and mode kept;
  counted in `grading.json` as `ownership_moved_to_root`), then hands `/`, `/logs` and `/logs/verifier` to an
  otherwise unused uid (47311), and creates the harbor wrapper's exit record there saying exit 1;
- it imports everything the verdict needs, runs `asyncio.to_thread` inline (no thread is ever started), and hands
  every file it has mapped (its interpreter, libpython, the extension modules it loaded, libc, ld.so, locale data) to
  that uid without write bits, keeping read and execute access as they were (counted in `grading.json` as
  `protected_mappings`, with the time it took);
- the eval script runs as uid 0, as at the server, but drops privileges: only `CAP_FSETID`, `CAP_KILL`,
  `CAP_NET_BIND_SERVICE`, `CAP_NET_RAW` and `CAP_AUDIT_WRITE` stay in its bounding set, with `no_new_privs`. It
  cannot override file permissions, change owners or switch uid, so it cannot create, rename or rewrite anything on
  the reward's path, nor rewrite in place any file `grade.py` has mapped;
- from the moment the eval script starts, an audit hook in `grade.py` refuses every import, library load, `exec`,
  fork and further subprocess, and every file open but of its outputs and `/proc/self/maps`, so nothing the tests
  could have changed is read afterwards. After the tests it checks that no file mapping is new and that every
  protected file kept its owner, mode, size and mtime. A refusal, a failed check or any exception after the tests
  started exits 1 without a reward: a measured 0, never a 1 and never a mask;
- `grade.py` writes `reward.json` and `grading.json` as bytes, as that uid, so a test process left running cannot
  change them.

The capture files (`capture.json`, `model_patch.diff`) come from the agent's sandbox. They are read only if they are
regular files within their size bound (1 MiB, 64 MiB), without following a symlink or blocking on a FIFO, and parsed
so that nothing nested reaches `grading.json`; any other shape is a completed 0 (difference 13), never a mask.

The reference solutions, the negative and plant controls, a mapped-file mutation control and a model smoke were
re-run with this grader on real sandboxes (see [After the grader hardening](#after-the-grader-hardening)). For other
languages, run `gym dataset validate` on every task first, since the dropped capabilities and the protected files
may break builds.

What the agent's code can still do:

- **Kill the grader.** Killing `grade.py` leaves no reward (scored 0). Killing the wrapper around `test.sh` before
  it writes its exit record leaves the record `grade.py` created (exit 1), so that scores 0 too; the wrapper
  overwrites the record with the real exit code when it finishes, and the tests cannot rewrite it.
- **Mask through the wrapper's exit record (open).** A test process can lower the wrapper's file size limit
  (`prlimit` on a process of the same uid needs no capability). The wrapper's own write then empties the record and
  fails, and the harbor server masks a run whose record is empty. Shown with the real `verifier_command` and
  `parse_exit_record` on a local filesystem, not in a sandbox.
- **Rewrite the reward after `grade.py` exits (open).** After `test.sh` returns, the harbor server still runs
  programs in this sandbox as root with every capability: the wrapper's `date`, and `tar` and `gzip` collecting
  `/logs/verifier`, all found through `PATH`. The tests own those programs and the folders on `PATH` (uid 0 owns
  root's files), so they can replace one, for example `/usr/local/bin/tar`, with a script that rewrites
  `reward.json` to 1 before running the real program. This forges the score. Shown with the real collection code
  (`verifier_command`, `download_dir`) against a local stand-in for the sandbox, not on a live run. `grade.py`
  cannot close it: it has exited by then, and taking those files and folders away from uid 0 would break the eval
  script's own installs.
- **Hold the grader with a detached process.** `local_sandbox.py` waits until the eval script's output pipes close.
  A test process that starts its own session (`setsid`) and keeps the pipe open holds `grade.py` until the
  verifier time limit (`[verifier].timeout_sec`, the evaluation timeout plus a margin), which leaves no reward and
  scores a measured 0. A background process that stays in the eval script's process group only stretches the run to
  the evaluation timeout, when the group is killed and the output is graded.
- **Forge the test output.** The verdict is read from the eval script's output, as at the server: a `conftest.py`
  that reports failing tests as passed scores 1 on both arms. This is a property of the server's grading, not of
  the conversion.
- **Mask through the sandbox provider (open).** Test code can end the verifier without an exit status, which the
  harbor server masks. Two routes were measured on real sandboxes with this grader: tests that rewrote a shared
  library the grader does not map but the server's later `bash` loads (5 of 5 masked, `provider_unavailable`), and
  a test that ran `kill -9 -1` (masked). Killing the provider's exec daemon or stopping the sandbox through the
  network was not tried.
- **Reach `grade.py` below Python.** The audit hook sees Python-level imports, loads and opens; a library load from
  C (name service, `iconv`, unwinding) is not audited. The path after the tests makes none, and one would show as a
  new mapping in the check. A kernel or provider escape is out of reach of any in-sandbox grader.

Closing the two open routes and the provider mask needs the reward read and the exit status recorded outside the
verifier sandbox (a harbor server change).

The capture hook runs in the agent's sandbox, where the agent is root and owns the hook's files and outputs, so no
capture outcome is ever a mask (differences 7 and 13). Tamper safety comes from the tasks themselves
(`environment_mode = "separate"` and `grade.py`), not from a server option: `harbor_arm.yaml` carries the run
settings both arms were compared with.

## Requirements and known limits

- The image needs Python 3.10+ for the capture hook and 3.9+ for the grader (Scale-SWE images ship 3.11), `timeout`
  for the seed, and a sandbox that grants `CAP_CHOWN`, `CAP_SETUID` and `CAP_SETPCAP` to root (Docker's default set
  does).
- Without `CAP_CHOWN` or `CAP_SETPCAP`, `grade.py` cannot protect the reward and masks the sample before any test
  runs; the same holds when it cannot hand a file it has mapped to the reward uid (a read-only filesystem counts as
  protected). A privilege drop that fails in the eval script's own process, after it started, scores 0. Without a
  root that can change its effective uid (`CAP_SETUID`, and uid 47311 usable), `grade.py` cannot write any output:
  every episode, the oracle included, ends with no reward and scores 0, so gate G2 shows it at once.
- `capture.py` writes the patch and its JSON file, and `grade.py` writes the eval script and the patch into the
  verifier sandbox, with Python's default encoding, which follows the image's locale (UTF-8 under a UTF-8 or C
  locale). Under another locale a patch with non-ASCII text can be altered, which can change its reward. `grade.py`
  reads the capture as UTF-8 and writes its outputs as UTF-8 bytes. A locale whose codec the audit hook would have to
  load after the tests is loaded before them; one it cannot load then scores 0, never 1.
- Piloted on Python repositories only. The parity subset is 50 tasks; the full dataset was not converted or run.
- Scale-SWE images are pulled from Docker Hub; a large run can hit its anonymous pull limit. A mirror in a registry
  your sandbox provider can pull from removes the risk.
- The grader still runs next to the agent's test code. What that leaves open is listed under
  [Reward safety](#reward-safety).
