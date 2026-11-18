# Sandbox, runtime, cleanup and robustness (items 8, 9, 10, 15)

Line numbers are at main `099832fa7e`.

- `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`
- `B` = `nemo_gym/agents/base_responses_api_agent.py`

## Ownership and cleanup (item 8)

**Ownership rule.** Only the owner destroys a sandbox, and the agent never falls back to running on the host (G:28, G:217-226, G:341-346, G:805-815; `AGENTS.md` Sandbox section).

- **Borrowed sandbox:** disconnect, and remove only the adapter's own files.
- **Owned sandbox:** stop it on close and on failed setup, and set `ttl_s`.

**Reviewer findings:**

- An unconfirmed cleanup receipt meant a borrowed session could never close (#3902). Close must reach a final state.
- When the installer exited early, the setup directory and 55-112 MB archives leaked (#5152).
- An owned sandbox without `ttl_s` leaked when the agent server crashed (#3961).

**Stopping the whole tree.** At a deadline or close, stop every process the harness started.

- Killing only the wrapper orphans the CLI and its tools (#3902).
- Harnesses `setsid` their tool commands, so a process-group kill misses them (#3902).
- A receipt said "confirmed" while descendants still ran (#5094).
- Guard against PID or group reuse.

**Proposed (plan step 6):** rollout stop on by default for every adapter that holds a `SandboxSession`, and `cleanup_confirmed` in every close reply.

## The task image (item 9)

**Needs:** nothing beyond bash. No python3, uv or network to install. Examples flagged:

- Codex still needed python3, installed with apt/apk (#5094).
- Hermes' uv install read the task's `uv.toml` and wrote into the task's home (#5094).

**Built on the host, unpacked in the sandbox: fine.**

- `SandboxPythonRuntime`, and Claude Code's and Droid's downloads, build or download on the Agent Server.
- A reviewer praised this design in #5152.
- `AGENTS.md` forbids an *unused host copy*, not host-side building.

**Isolation:** the harness's private `HOME` and XDG directories must not leak into the tool shells the verifier depends on (#3902).

**Probing the image (#5152, #5379):**

- check glibc before a musl loader;
- use `tar --no-same-owner` in user namespaces;
- create directories as the sandbox user;
- don't assume the newest bash.

**Network:**

- Owned sandboxes offer `network_access` (`inherit`, `model_only`, `model_and_tools`) through `restrict_sandbox_network()`.
- A borrowed sandbox refuses anything but `inherit` (G:352-383).
- If the runtime would download at run time, offer a pinned preinstalled binary.
- Web fetch is off by default (#5082, #5538, #5546).
- Proposed (plan step 3): download CLI runtimes on the Agent Server with pinned digests, so `model_only` works everywhere.

## Refuse, don't drop (item 10)

**The rule:** apply, refuse (422) or document every config key and request field (G:656-662).

**On main today:**

- Agent configs are plain `BaseModel`s, so unknown keys are silently ignored (`nemo_gym/config/config_types.py:703`). #5718 adds strict configs for six adapters.
- `SandboxAccess.user` and `env` are ignored or refused differently by each adapter (plan step 4).

**`calls_model_from_sandbox`:**

- Declare it as a literal on the entrypoint class. It is not inherited; Harbor reads it from source (B:342-350; `nemo_gym/data/tasks/harbor/prepare.py:108-114,860`).
- Missing on Claude Code, Kilo, Stirrup and NOOA (fixed in draft #5863), and on #5003.
- The guide does not mention it.

**Time budget:** `resolve_agent_timeout(metadata, ceiling=config)` (G:412, G:534-538). Kilo refuses Harbor rows' `agent_timeout_sec` (fixed in #5863).

## Robustness (item 15)

**Fail closed:**

- A plugin load failure was ignored; add a sentinel the runner checks (#3902).
- The cache-hit check ignored the `--version` exit status (#5094).
- A failed redirect fell through and the CLI launched anyway (#5094).

**Bound sizes:**

- A quadratic CR strip took seconds per 1 MB line (#5094).
- A 64 KiB cap ignored artifacts (#5703).
- An unbounded log was appended to stdout (#3902).

**Paths and text:**

- `realpath(strict=False)` let a symlink escape the root (#5703).
- Project safe, redacted fields from errors (#5543).

**Secrets:**

- Keep `direct_http` cookies out of files the model can read (G:259).
- Keep per-session credentials out of process-wide state (G:756).
- A CLI runs through `harness_runner.sh`, which a task's `pkill python` cannot match (G:480).
