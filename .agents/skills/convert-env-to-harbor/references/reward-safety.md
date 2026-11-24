# Reward safety

The converted datasets are for RL post-training, where a policy learns any route to a reward. On the native path
the verdict is computed in the Resources Server process, out of the agent's reach. In a task folder the verifier is
a script in a sandbox, and Harbor reads `/logs/verifier/reward.json` (or `reward.txt`) after `test.sh` returns,
whatever its exit code. Every route below was found in a pilot, and every fix was proven with a real run.

## Where the agent can reach the reward

| Route | Seen in | Fix that held |
| --- | --- | --- |
| **Shared verification.** The verifier runs in the agent's sandbox, where the agent is often root: it replaced `python3` with a script writing `{"reward": 1.0}` | math: 5/5 scored 1.0 with no answer | Separate verification. For an overlay image without a registry: `artifacts = ["/logs/artifacts"]` in `task.toml` + `separate_verifier_for_artifacts: true` on the harbor server. Tamper control: 0/5 |
| **Agent code executed by the verifier.** A separate verifier that runs the agent's patch (pytest imports a `conftest.py`) runs it as root next to the grader; the code wrote `reward.json` and killed the grader | scale_swe: 1.0 with no fix | Ownership, mapped-file protection, capability drop and an audit hook in `grade.py` (rule 2). Probes `plant_kill`, `plant_daemon`, `plant_report` and a mapped-file mutation control: 0.0 unmasked |
| **Collect-hook output.** The hook runs in the agent's sandbox; the agent rewrote `capture.py` to report an infrastructure error | scale_swe: a mask instead of a 0 | Every hook outcome but a readable patch is a completed 0, never a mask. Probes `force_mask`, `capture_delete`: 0.0 unmasked |
| **Records the seed leaves in the agent's sandbox** (base commit, untracked list) | scale_swe | A missing or changed record scores 0, never a mask; no hidden data there |
| **Forged state files.** A forged journal and reward written in `main` | workplace_assistant probe: 0.0 on both of its 2 tasks | The journal lives in a sidecar the agent has no shell in, reaches the verifier as an artifact from that service, and only real tool calls append to it |
| **Collection after grading.** The harbor server ran `date`, `tar` and `gzip` from the verifier sandbox after `test.sh`, to time the run and collect the reward; agent code in the tests replaced them with scripts that rewrote the reward | scale_swe on OpenSandbox: 1.0 on 5 of 5 tasks whose grader had scored 0 | A harbor server change, not an adapter change: Gym reads the reward through the provider's file download before it runs anything else in that sandbox ([Reward files](../../../../fern/versions/latest/pages/data/task-folders/task-folder-verifiers.mdx#reward-files)). Same control: 0.0 on 5 of 5, unmasked. Probe: `--plant-collection` |

## Rules

1. **Separate mode by default.** Use shared mode only when the verifier must inspect live processes the agent
   started, and then accept that a root agent can forge the reward.
2. **Protect the reward path, and the grader's own code, when the verifier executes agent-written code.** The SWE
   graders (`scale_swe/harbor/.../grade.py`; swe_rebench copies the same code) do this, in order:
   - Ownership. Make uid 0 the owner of every root-filesystem entry another uid owns, then hand `/`, `/logs` and
     `/logs/verifier` to an unused uid (all three: with only the last protected, root renames a parent away), and
     remove any reward file already there. Write no other file there: the harbor server reads only the reward.
   - Mapped files. Import everything the verdict needs and make `asyncio.to_thread` inline, then hand every file in
     `/proc/self/maps` to that uid without write bits. Otherwise the tests, as root, rewrite an already-mapped
     library such as `_json` in place and the grader writes their reward. Do the same for the maps of every process
     above the grader (follow the parent links in `/proc/<pid>/stat`, assume no shape): `timeout` and the server's
     shell relay the exit status, and a library of theirs rewritten in place could turn a 0 into a mask. Skip a
     parent whose maps the grader is not permitted to read (`PermissionError`: the provider's own service or the
     sandbox's init, such as pid 1 on OpenSandbox) and list it in `grading.json`; it falls under the open route
     through the provider's in-sandbox service. Mask before the tests on any other error reading a parent.
   - Privilege drop. Run the tests as uid 0 with the bounding set cut to `CAP_FSETID`, `CAP_KILL`,
     `CAP_NET_BIND_SERVICE`, `CAP_NET_RAW`, `CAP_AUDIT_WRITE`, inheritable and ambient cleared and `no_new_privs`.
   - Fail closed. From the tests' start an audit hook refuses every import, library load, `exec`, fork, further
     subprocess and file open but of the outputs; after them, no mapping may be new and no protected file changed.
     Anything that goes wrong after the tests started exits 1 without a reward (a 0), never 75.
   - Bounded capture. Read the collect hook's files only as regular files within a size bound; any other shape is a
     completed 0 (rule 3). Write `reward.json` as bytes with `seteuid(<that uid>)` and `O_EXCL|O_NOFOLLOW`.

   If a protection step fails before the tests (a provider without `CAP_SETPCAP` or `CAP_CHOWN`), mask; never fall
   back to the unsafe mode. Without `CAP_SETUID` the grader cannot write any output, so every episode (the oracle
   included) scores 0, which G2 shows at once. Test it without root (the scale_swe and swe_rebench adapter tests),
   and on real sandboxes with a `--solution-dir` control whose tests try to rewrite every mapped file: it must score
   0 unmasked. Details and measurements: the scale_swe README's
   [Reward safety](../../../../resources_servers/scale_swe/harbor/README.md#reward-safety).
3. **Collect-hook output is agent-controlled.** Delete stale outputs first, make "could not run" distinguishable from
   "cut short", and score any outcome you cannot trust as a completed 0. A sha256 check inside the hook does not
   help: the agent can replace `sha256sum` or `python3`, or rewrite the output after the hook ends.
4. **Exit 75 only where native masks, on facts the agent cannot write**: no Python in the verifier image, a missing
   journal from a sidecar the agent cannot reach, a failed pinned install, a grader that produced no verdict before
   any agent code ran (once the agent's code ran, a missing verdict or a marker in its output is a 0). Where
   native masks, do exit 75: a verifier that never masks turns those cases into measured 0s. A missing reward
   without exit 75 is a measured 0 with a `failure_kind`; in shared mode exit 75 is ignored.
5. **Journal before execution.** A stateful tool server appends each call before running it, so a call that
   crashes the tool is still graded, as native `verify()` keeps every reported call.
6. **Sidecar journals need separate mode.** A sidecar file shared into `/logs` (the "Sidecar inputs" pattern) is
   writable by the agent; copy the journal out as an artifact from its service instead.
7. **The verifier's `workdir` must exist in its image.** Gym does not create it; a missing one masks every episode
   as `provider_unavailable` (workplace_assistant's first oracle run).
8. **Run the grader with `python3 -I`** so files in the working directory (the agent's repository) cannot shadow
   the standard library, and serve the grader's one `sandbox.exec` through a local subprocess shim (process group,
   kill on timeout, `error_type="timeout"`), as `scale_swe`'s `local_sandbox.py` does.
9. **Stale outputs.** Remove any reward file and hook output before writing yours.
10. **Hidden data stays out of the agent's view** (SKILL.md Step 6). `tests/` is uploaded to a separate verifier by
    Gym only; upstream Harbor needs a `tests/Dockerfile`. `gym dataset publish --harbor --output <dir> <dataset>`
    shows what a plain Harbor runner would get.
11. **Seed failures are loud.** Never let a seed or setup step fail quietly (`|| true`): a task that could not be
    set up (a base commit that does not check out) must fail its seed, not run on and score a silent 0.

## Probes (G3)

Every probe must score 0 unmasked on the Harbor arm:

```bash
python -m scripts.harbor_conversion_checks controls <dataset> <out> --plant-reward
gym dataset validate <out> --sandbox <provider> --config harbor_arm.yaml
```

`--plant-reward` writes forged reward files and `python3`/`python` stubs from the agent's sandbox and keeps rewriting
them for `PROBE_SECONDS` (default 120). In separate mode it proves separate mode is on. When the verifier executes
agent output, add an artifact-borne probe: `--plant-conftest` for pytest images (a `conftest.py` that writes the reward
files, leaves a daemon rewriting them and kills every process with an argument starting with `PROBE_KILL_PREFIX`,
default `/tests/`), `--plant-hooks` for npm, Go and pytest
repositories (the same payload from npm lifecycle scripts, a Go test `init()` and a `conftest.py`; the swe_rebench dry
run's probe it is taken from scored 0 on JS, TS, Go and Python tasks), and for other languages your own
`--solution-dir` probe (Maven plugin, Rust `build.rs`: not written yet). A plant probe may carry `failure_kind`
`harbor:missing_reward` (the grader was killed): still a 0, never a 1 and never a mask. To try the probes outside a
sandbox, set `PROBE_KILL_PREFIX` to a folder of your own (or `PROBE_KILL=0`); the default kills every process of your
user with an argument under `/tests/`.

`--plant-collection` (pytest images: it plants through a `conftest.py`) replaces `tar`, `gzip` and `date` during the
tests, so that they rewrite the reward to 1 if anything runs them after the grader finished. It tests the Gym that
collects the reward, not the dataset: every reward 0 and nothing masked means that Gym was not fooled; a reward of 1
means the reward was rewritten after the grader finished, a property of the Gym version. Outside a sandbox, point
`PROBE_COLLECT_TOOLS`, `PROBE_COLLECT_BIN` and `PROBE_COLLECT_KEEP` at scratch paths
([checks README](../../../../scripts/harbor_conversion_checks/README.md)); the defaults replace real programs.

## What stays open

The full list, with what was measured and what was not, is in the scale_swe README's
[Reward safety](../../../../resources_servers/scale_swe/harbor/README.md#reward-safety); copy it into an adapter
README that executes agent code. In short: test code can kill the grader (a 0, or a mask on a provider that reports
no exit status), forge its own test output (1 on both arms, as natively), and kill or impersonate a sandbox
provider's in-sandbox file service (OpenSandbox, E2B, Daytona, ECS Fargate), which masks or could forge. Providers
whose file download runs a sandbox program get no collection-stage guarantee, nor does shared verification. A
dataset run on a Gym from before the reward was read through the provider's file download keeps the collection
route and the exit-record masks: `--plant-collection` tells which Gym you have.
