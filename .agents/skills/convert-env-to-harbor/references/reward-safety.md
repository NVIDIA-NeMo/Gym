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

## Rules

1. **Separate mode by default.** Use shared mode only when the verifier must inspect live processes the agent
   started, and then accept that a root agent can forge the reward.
2. **Protect the reward path, and the grader's own code, when the verifier executes agent-written code.** The SWE
   graders (`scale_swe/harbor/.../grade.py`; swe_rebench copies the same code) do this, in order:
   - Ownership. Make uid 0 the owner of every root-filesystem entry another uid owns, then hand `/`, `/logs` and
     `/logs/verifier` to an unused uid (all three: with only the last protected, root renames a parent away), and
     create the wrapper's exit record `.ng-exit` there saying 1, so a wrapper killed before its write reads as a 0.
   - Mapped files. Import everything the verdict needs and make `asyncio.to_thread` inline, then hand every file in
     `/proc/self/maps` to that uid without write bits. Otherwise the tests, as root, rewrite an already-mapped
     library such as `_json` in place and the grader writes their reward.
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
   0 unmasked. What this cannot close (see "What stays open") needs the reward read and the exit status recorded
   outside the verifier sandbox, a harbor server change. Details and measurements: the scale_swe README's
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

## What stays open

- An agent's test code can kill the grader or the wrapper around `test.sh` (both a 0 with the pre-created exit
  record), and can forge its own test output, which scores 1 on both arms as natively.
- **Forgery after the grader exits.** After the grader has written the reward, the harbor server still runs `date`,
  `tar` and `gzip` in the verifier sandbox as root with every capability, to record the exit status and collect the
  reward, found through `PATH`. Test code running as uid 0 owns those files and folders, so it can replace one with
  a script that rewrites `reward.json` to 1 during collection. Shown with the real collection code against a local
  stand-in for the sandbox, not on a live run. No in-sandbox grader can prevent this.
- **Mask through the exit record.** Test code can lower the wrapper's file size limit, so the wrapper's own write
  empties `.ng-exit`; the server masks a run with an empty record.
- **Mask through the provider.** Test code can end the verifier without an exit status, which the server masks.
  Measured on real sandboxes with the hardened grader: tests that rewrote a shared library the grader does not map
  but the server's later `bash` loads (5 of 5 masked, `provider_unavailable`), and `kill -9 -1` (masked).
- Library loads from C are not audited (none happens after the tests in the SWE graders).

Closing the first three needs the reward read and the exit status recorded outside the verifier sandbox, a harbor
server change; the adapters cannot do it.
