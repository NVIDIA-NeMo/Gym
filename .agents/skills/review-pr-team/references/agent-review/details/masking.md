# When to mask (item 4)

Line numbers are at main `099832fa7e`. `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`.

## Rule

From G:678-693:

- **A limit is reached** (time, turns, output, context): `incomplete`, still graded, and partial work is kept.
- **Harness termination is not confirmed:** fail the activation and skip verification.
- **Verification fails because of infrastructure:** raise, or set `mask_sample: true`. Never report a reward of 0, and never cache it as the session's verification result.

From the repo's `AGENTS.md`, Sandbox section: when verification is inconclusive (a sandbox start, exec or provider error), retry on a fresh sandbox up to `inconclusive_verification_retries` times. If it is still inconclusive, report `evaluation_completed=False`. Example: `resources_servers/deepswe/app.py:117,745`.

**A lost session:** `lost_session_verification()` returns reward 0.0, `mask_sample: true` and `failure_kind=session_lost` (`nemo_gym/server_runtime/base_resources_server.py:409-423`). Use it whenever the verifier no longer holds the session, including after close.

**The #5675/#5679 pattern** (merged 2026-11-09):

- An infrastructure failure becomes a typed `NgFailureEvent` through `observe_failure`, with its cause kept.
- The row is marked `evaluation_completed: false`.
- A boundary outcome policy picks either mask (`retry`, the default) or an explicit `count_as_zero`. The row records `failure_outcome_action` and its attribution.
- An ambiguous probe is inconclusive, so it is masked.
- A real task failure, such as an invalid `reward.txt`, still scores 0.
- Pointers: `nemo_gym/core/outcome_policy.py:31-54,157-174`; `resources_servers/terminal_bench_2_1/app.py:704-764`; recovery reruns masked rows (`nemo_gym/rollouts/rollout_recovery.py:467`).

**Training:** a model call that was not captured, or cannot be chained, sets `mask_sample` (G:648).

## Never the model's 0

- a lost sandbox or session;
- a verifier infrastructure error, such as an apt lock or a download 504;
- an unconfirmed termination;
- a seed or provider failure;
- an unreachable model server: retried, not graded (plan step 1, decision 2);
- an exit with no result or completion event, which is typed as `agent_run_error`.

## What reviewers flagged

- Exit 0 without a completion event was scored as the model's zero (#5537).
- A hung model server at the deadline was scored as incomplete (#5546).
- Verify infrastructure failures returned 0 without `mask_sample` (#3902).


## Field names

`AGENTS.md` says `evaluation_completed=False`; G says `mask_sample: true`; the #5675 pattern sets both. Accept either one, as long as the row is not a scored 0. See `known-conflicts.md`.
