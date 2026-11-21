# Where the sources disagree

Checked on 2026-11-11 against main `eeccee711d`. The code wins. Don't flag a PR for following the code side of a
conflict listed here; mention the conflict instead. A PR that copies the doc side gets the finding, with the doc cited
as the likely cause.

- `F` = `fern/versions/latest/pages/`
- `S` = `.agents/skills/add-benchmark/SKILL.md`, `P` = `.agents/skills/add-benchmark/references/patterns.md`
- `R` = `nemo_gym/server_runtime/base_resources_server.py`, `T` = `environment_servers/single_agent_turn/app.py`

## Session contract

1. **The stateful tutorial shows the old seed.**
   - `F/environment-tutorials/stateful-environment.mdx:97-104` overrides `seed_session` with the flat body and has no close; `:132-137` scores a missing session 0; `:155-156` re-assigns state on a re-seed. `F/environment-tutorials/real-world-environment/resources-server-implementation.mdx:95-125` does the same.
   - Code: typed envelope plus close (R:284-300, R:1163-1190); a lost session is `lost_session_verification` (R:1192-1199); a repeated seed returns the existing session and another episode gets 409 (R:285-288, R:1141).
2. **Who seeds.** `stateful-environment.mdx:29, 169` have the agent seed. Under an Environment Server, the Environment Server seeds (T:420-451).
3. **The hooks example.** `F/environment-tutorials/resources-server-apis.mdx:408` points to the MCP page as the hooks example; `F/environment-tutorials/mcp-resources-server.mdx:117-121` is a cookie seed returning `BaseSeedSessionResponse`, which `single_agent_turn` rejects because it needs `resources_session_id` (T:442-444).
4. **The seed model.** `resources-server-apis.mdx:72-101` describes a `/run`-driven `seed_session(body: BaseSeedSessionRequest)`. Code: the subclass goes in `seed_request_model` (R:587, R:1147).
5. **Who holds the cookie.** `resources-server-apis.mdx:345` says the Agent Server. Code: the Environment Server keeps the seed reply's cookie and hands it out (T:436, T:532, T:668).
6. **The in-repo session example breaks the contract.** `resources_servers/example_session_state_mgmt/`: bad `task_data` is a 500 (`nooa_app.py:47`), a tool call on a closed session is 409 not 404 (`nooa_app.py:75-79`), verify on a lost session returns 0 (`app.py:109-117`).
7. **TB2.1** answers a verify for an unknown or closed session with a terminal 409 (`resources_servers/terminal_bench_2_1/app.py:585-611`), not the masked `lost_session_verification` the base contract asks for (R:1192-1199). Open question whether that is intended.

## New work and the old path

8. **Onboarding teaches `/run` as the default.**
   - `S:21-31` (native benchmarks on `simple_agent`, external ones wrapped in `/run`), `S:200-201` (smoke with `--agent ..._simple_agent`), `S:292` ("`/run` endpoint must be async"). Only `S:159` points to session harnesses.
   - `P:199-222, 302-376, 618-646` (`simple_agent` pairing with no Environment Server, a custom `run()`, an external library wrapped in `run()`); `P:386` points to `harbor_agent` as the pattern to copy.
   - `F/environment-tutorials/single-step-environment.mdx:47, 59, 144-169, 304-333`; `multi-step-environment.mdx:36, 61`; `resources-server-apis.mdx:17-34`; `integrate-external-environments.mdx:6-29`; `mcp-resources-server.mdx:300-326` (no Environment Server, so Gym adds a relay; the shipped config uses `legacy_agent`).
   - `AGENTS.md:111` (config wires resources server, agent and model; no Environment Server), `AGENTS.md:122` ("the `/run` endpoint must be async").
   - The typed path (`single_agent_turn` plus `taskset`) appears only as opt-in material: `resources-server-apis.mdx:111-142, 522-575` (":534 Omit `taskset` to keep the current flat-row format").
9. **The scaffold still writes `legacy_agent`.** `F/agent-interactions/index.mdx:69-73` says new work need not use it. `nemo_gym/environment/scaffold.py:349-360` writes it ("for now this builds the legacy config structure ... once we have any migrated agents"; `simple_agent` already supports sessions, `nemo_gym/config/global_config.py:227-247`). `F/contribute/environments/new-environment.mdx:172, 241, 346` and `adding-a-benchmark.mdx:28, 71` scaffold it without saying so.
10. **The migration guide targets `single_agent_turn_legacy`.** `F/troubleshooting/configuration.mdx:261-296` moves configs to `single_agent_turn_legacy`. That is accepted (item 11); the guide just never shows the preferred typed taskset on `single_agent_turn`.
11. **Settled 2026-11-11: `single_agent_turn_legacy` is allowed for new work; `legacy_agent` is not.** Both review guides now agree (`../../agent-review/review.md` item 1). A typed taskset on `single_agent_turn` is preferred, not required.
    **Exception, decided 2026-11-13:** a benchmark that does not need more than one harness may stay on `legacy_agent`
    (for example an `external-agent-loop` benchmark run by its own upstream harness, #5743). The relay is a finding
    only when the benchmark should pair with several harnesses.
12. **`gym env init`** is deprecated for new datasets in favour of `gym dataset init` (`nemo_gym/cli/main.py:953-956`); `S:37-41` and `single-step-environment.mdx:55-77` still present it as the path. The team treats the Gym and Harbor formats as equal for new datasets (decided 2026-11-11), so the CLI's preference for Harbor is not a review rule. Whether the CLI message should change is open.
13. **`legacy_agent/README.md:8`** says every agent "needs" an Environment Server; Gym creates one automatically unless `error_on_agent_without_environment_server` is set (`global_config.py:1238-1290`).
14. **Harbor.** `F/infrastructure/sandbox/daytona-access.mdx:30` sends users to `harbor_agent` with no deprecation note (`responses_api_agents/harbor_agent/app.py:236-242`). `environment_servers/harbor/README.md:4` names only `tests/test.sh`; a Python verifier is also accepted (`nemo_gym/data/tasks/harbor/python_verifier.py:65-90`). `harbor/README.md:94, 97` name test and requirements paths that do not exist.
15. **Harbor transcripts must be text-only** (`nemo_gym/trajectory/atif_export.py:1155-1162`), but no `F/data/task-folders` or `F/agent-interactions/harbor.mdx` page says so; `F/troubleshooting/task-folders.mdx:184` describes `harbor:trajectory_export_failed` only in general terms (issue #5871).

## Pairing

16. `F/reference/configuration.mdx:294-296` says only `--agent-type` refuses unlisted agents; rollout collection also refuses static pairings (`nemo_gym/rollouts/rollout_collection.py:5270-5323`).
17. `F/reference/configuration.mdx:523` says `NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING` waives only dataset `allowed_agents`; it waives every pairing guard, including `allowed_model_types` (`global_config.py:852, 1133, 2216-2220`).

## Serving (repo `AGENTS.md`)

18. **`ServerClient` retries.** `AGENTS.md:127`: "3 tries, exponential backoff". Code: a fixed 0.5 s backoff (`nemo_gym/client/http_client.py:842-844`); refused or dropped connections retry with no count limit unless the caller bounds them (`http_client.py:749-799`); the 3-try cap covers other exceptions on external calls only (`http_client.py:800-834`); an HTTP error status is never retried (`http_client.py:735-748`).
19. **httpx.** `AGENTS.md:129` bans it; nothing enforces it (`pyproject.toml:832-840` has no banned-API rule) and at least seven runtime files construct an `httpx.AsyncClient` (see `serving.md`).
20. **Cookies.** `AGENTS.md:130` says `cookies=request.cookies`; on the typed session path the cookies come from the seed reply (T:441, T:668), and `request.cookies` is the cookie-seed fallback.
21. **Sandbox helpers.** `AGENTS.md:158-160` keeps helpers out of `nemo_gym.sandbox`; `nemo_gym/sandbox/__init__.py:152-192` exports `upload_dir`, `download_path`, `exec_as_root`, `upload_text`, `read_text`, `stage_media` and others.
22. **`ttl_s`.** `AGENTS.md:170` says set it; it defaults to `None` and nothing checks it (`nemo_gym/sandbox/config.py:226`).
23. **Inconclusive verification.** `AGENTS.md:173-175` names a `inconclusive_verification_retries` knob and says report `evaluation_completed=False` instead of a silent 0. Code: no shared knob (per-server copies, defaults 1 or 2); `evaluation_completed=False` alone is a scored 0 because only `mask_sample` removes a row (`nemo_gym/metrics/reward_profile.py:892-902`); ten SWE servers end that way (`resources_servers/swe_rebench/app.py:376-384`). Accept a PR that sets `mask_sample: true` (with or without `evaluation_completed: false`). Flag one that sets `evaluation_completed: false` alone as a `RISK`, not a `BLOCKER` (decided 2026-11-11), since it follows `AGENTS.md`. Whether to fix `AGENTS.md:173-175` to say `mask_sample` is open.
24. **Status codes.** `AGENTS.md` has none. The rule lives in R:1084-1092 and `nemo_gym/server_runtime/base_environment_server.py:185-210`.

## Data

25. `F/data/jsonl/index.mdx:26, 52` and `F/data/jsonl/prepare-validate.mdx:225, 320` say collate adds `agent_ref`; it strips it and stamps `task_source` (`nemo_gym/data/dataset_preparation.py:1266-1298, 1328-1334`).
26. `F/data/jsonl/index.mdx:176` says a missing `responses_create_params` raises; collate records the row in `offending_example_idxs`, which nothing reads, and still writes it (`dataset_preparation.py:806-808, 1281-1320`).
27. `F/data/jsonl/prepare-validate.mdx:274` says invalid JSON is skipped; `json.loads` raises (`dataset_preparation.py:1286`).
28. `F/data/jsonl/index.mdx:86` declares datasets on the agent; the Resources Server is the normal home (`dataset_preparation.py:613-616`; `adding-a-benchmark.mdx:103`).
29. The JSONL pages omit the `benchmark` type, `taskset`, `allowed_agents`, `prepare_script`, `task_data.py` and the `input_image` rules (`nemo_gym/config/config_types.py:555-581, 658-690`; `nemo_gym/core/media.py`).

## Onboarding process

30. `S:43-53` and `single-step-environment.mdx:62-77` say the scaffold creates `uv.lock`; it does not (`nemo_gym/environment/scaffold.py:505-515`). The scaffold writes `example.jsonl` at the server root (`scaffold.py:514`) while its config reads `data/example.jsonl` (`scaffold.py:578`) — a code bug, not only a doc one.
31. `S:189` requires 95% server test coverage; servers run plain `pytest` (`nemo_gym/cli/env.py:342`); the 95% bar is core-only (`pyproject.toml:787`).
32. `S:161-177` and `P:258` make `gitlab_identifier` required; it is optional and deprecated in favour of `source:` (`config_types.py:570-573`).
33. `P:256` lists 3 `domain` values; the enum has 12 (`config_types.py:519-560`).
34. `new-environment.mdx:174, 231` call `example_rollouts.jsonl` optional for manifest-backed servers; CI requires it for any changed Resources Server (`.github/workflows/unit-tests.yml:275`; `env.py:301-307`).
35. `new-environment.mdx:241` says validate does not apply to standalone servers; `env.py:971-1023` validates plain configs.
36. `S:266` implies a new server gets a README row; `scripts/update_env_list.py:151-182, 296` skips configs whose datasets sit on the Resources Server, as the scaffold writes them.
37. `CONTRIBUTING.md:302, 313` say `benchmarks/**` changes are docs-only for CI; `.github/actions/classify-changes/action.yml:57-75` does not list that path. Not confirmed on a real PR.
38. `S:294` requires `-S` signing; `AGENTS.md:17` and `CONTRIBUTING.md:146-147` make it optional.
39. **Baselining bar.** `P:686-713` (a model suite, at least 30% for an open model, variance under 1%) vs `new-environment.mdx:176-186` (about 500 samples on Qwen3-4B/30B, 16 responses each; optional for training environments). `CONTRIBUTING.md:340-342` points fixed benchmarks to a guide that points to `adding-a-benchmark.mdx`, which has no profiling guidance. **Settled for new benchmarks (2026-11-11): reviewers apply `P:686-706`. A training environment needs only a real-model smoke (`CONTRIBUTING.md:337-341`).**
40. **`verified`.** Meaning: "baselined and reviewed" (`AGENTS.md:295`) vs "passed reward profiling and training checks" (`F/reference/configuration.mdx:288`). Who flips it: the author after the exit checks (`S:268`) vs maintainers after baselining (`CONTRIBUTING.md:356-357`). Nothing in code reads it. **Settled (2026-11-11): the author flips it after the exit checks, with the run folder, `quality_summary.json` and baselining numbers in the PR (`S:268`).**

## Code inconsistencies (not docs)

41. `resources_servers/swebench_pro/app.py:523-531` serializes the lent descriptor without `scope="operate"`, so the harness may be able to destroy it (compare `nemo_gym/server_runtime/sandbox_lending.py:198-207`).
42. `resources_servers/vision2code/app.py:141` writes `_ng_failure_class` into its verify response (as `nemo_gym/verification/judge.py:363` does); `resources_servers/swebench_pro/app.py:818-820` strips it as a collector-owned marker. One of them is wrong.
43. **Config file names.** `resources_servers/example_single_tool_call/configs/example_single_tool_call_single_agent_turn.yaml`
    names a pairing by its Environment Server path. **Decided 2026-11-13:** the main pairing config is `<name>.yaml`; a
    `legacy_agent` relay kept beside it is `<name>_legacy_agent.yaml` (as `swe_rebench`, `swemer_*`, `vibench` and
    `deepswe_external1` already do on main). Flag a new `<name>_single_agent_turn.yaml`; the example server is not a
    precedent.
