# Where the sources disagree

Checked on 2026-11-10 against main `099832fa7e`.

- `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`
- `S` = `.agents/skills/add-harness/SKILL.md`

Don't flag a PR for following either side of a conflict listed here. Mention the conflict instead.

1. **`num_workers`.**
   - G:181 and G:314 say agent sessions need `num_workers: 1`.
   - `fern/versions/latest/pages/agent-server/index.mdx:86` and `nemo_gym/agents/base_responses_api_agent.py:512-519` (`routes_sessions_to_owner`) allow more workers when the entrypoint exports a module-level `app`.
   - Accept either form.
2. **Runner launch.** S:53 names `RUNNER_FILE` and `stdlib_runner=True`, which #5327 removed. The current rule is G:480: run through `harness_runner.sh`.
3. **Client patches.**
   - S:138 says to normalize a client's replayed fields in `nemo_gym/core/responses_api_types.py`.
   - Plan step 7 (proposed) moves them to a client replay registry.
   - Until the registry exists, a core patch is acceptable if it is commented with the client it serves.
4. **TB2.1 masking.** G:689 says TB2.1 scores 0 and caches it. That is stale since #5675: TB2.1 now masks verifier infrastructure failures by default. The SWE-bench Pro half of that line was not re-checked.
5. **Mask field names.**
   - `AGENTS.md` says `evaluation_completed=False`; G:689 says `mask_sample: true`.
   - The #5675 pattern sets both.
   - `lost_session_verification()` sets `mask_sample` and a reward of 0.0.
6. **Outcome policy table.** G:718-722 lacks the RETRY rows for the transient `provider_*` kinds and the VERIFY row for `provider_oom_killed` that `nemo_gym/agents/agent_outcome.py:85-107` has. G:702's list of adapters that override `classify_outcome` is also stale.
7. **Legacy relay script.**
   - S:179 gives the test path `tests/unit_tests/test_add_legacy_agent_environment_servers.py`; on main it is under `tests/unit_tests/tooling/`.
   - S:179 also says to add the adapter by hand. On main the script already skips session adapters (`scripts/add_legacy_agent_environment_servers.py:163-170`).
8. **Harness retries.** G:727-728 says the outcome decisions "do not change native harness retries". That conflicts with accepted decision 6 until #5870 lands.
9. **Retry probes.**
   - The `retry_429` and `retry_500` conformance scenarios record a harness that never retries as incomplete (`scripts/harness_conformance/README.md:138-139,152-153`).
   - Decision 6 allows a harness not to retry; #5911 makes Claude Code send each request once.
   - Claude Code is not registered for the probes, so nothing fails today.
10. **Plain 422.** A plain `HTTPException(422)` passes `check_agent_seed_rejected`, but the base class does not log it (`nemo_gym/agents/base_responses_api_agent.py:903-905`).
