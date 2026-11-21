# Evidence and tests (items 13, 14)

Line numbers are at main `099832fa7e`.

- `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`
- `S` = `.agents/skills/add-harness/SKILL.md`

## Evidence (item 13)

**Behaviour-changing agent work needs real smoke rollouts with a model.** You must inspect the agent's and the verifier's behaviour. Green unit tests are not enough (repo `AGENTS.md`, "Quality bar"; G:856; S:115-142).

**What the PR body lists:**

- the exact validation performed;
- the rollout evidence, or `N/A` with a reason;
- any impact on benchmark results.

**What reviewers asked for:**

- Hold `allowed_agents` until full TB2.1 or SWE-Pro qualification runs exist (#5082). Adding to `allowed_agents` also needs the benchmark owner's review.
- A scripted-model smoke is not enough; attach a real policy and judge run (#5820).
- A version bump that touches other presets needs to be split out, or come with a before/after comparison (#5538).

The CI review bot flagged missing real-model smokes on about 10 PRs (#5151, #5153, #5543, #5659, #5696, #5698, #5762, #5820, #5823, #5877).

**A scripted-model smoke** (a stub Model Server, A/B against main's adapter) is good evidence for outcome and retry behaviour. It does not replace one real-model run.

## Tests (item 14)

**Each claimed behaviour has a test that fails when the fix is reverted.** Examples flagged:

- A test still passed with the timeout removed (#3902).
- No test failed when the start-time check was removed (#5094).

**Dependencies:** a dependency installed in `conftest` fails loudly. A warning that silently skips the tests in CI is a finding (#5153). For a truly optional external tool, use `pytest.mark.skipif(shutil.which(...) is None, ...)` (repo `AGENTS.md`).

**Waits:**

- Every wait is bounded: use `wait_for`, not a bare await. An unbounded await makes a failing test hang instead of fail (#3902).
- Leave enough setup time before a short deadline. A 2 s deadline fired before setup finished (#5152).

**Seed refusals** are tested with `check_agent_seed_rejected`.

**What the PR ran:**

- `gym env test +entrypoint=responses_api_agents/<agent>` for each changed agent and each subclass (S:71).
- `tests/unit_tests/test_session_contract_repo.py`.
- Every test that names a changed config.

**Linux-only tests:**

- Session probes and lifecycle tests run only on Linux (G:830).
- On a Mac they skip. Ask for a Linux run, for example on a Linux machine.
