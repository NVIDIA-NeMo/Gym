# SWE Atlas Codebase Q&A

Evaluates the 124 Codebase Q&A tasks from [Scale AI's SWE Atlas](https://github.com/scaleapi/SWE-Atlas)
through `harness_agent`. This subset asks an agent to investigate an existing repository and answer a question.
It does not include the Test Writing or Refactoring subsets.

## Preparation

```bash
python -m resources_servers.swe_atlas_qa.prepare
```

`prepare.py` fetches upstream revision `c6673fe259c3493a2f834266abf63247a5d1507c`, checks the revision,
rejects modifications to the task assets, and generates the gitignored `data/benchmark.jsonl`.
The five checked-in smoke examples are separate and are never overwritten by preparation. Rows contain the upstream
instruction, task ID, image, and working directory. Rubrics, evaluator prompts, and reference answers remain in the server's private source cache.
The digest-pinned v1.1 task images come from the upstream task definitions on GHCR.

These tasks, answers, rubrics, and resulting trajectories are **evaluation-only**. Follow the upstream canary's
prohibition on including benchmark data in training corpora.

## Harness and endpoints

The resource server creates a task sandbox in `/app` and returns its descriptor from `/seed_session`.
`harness_agent` attaches to that same sandbox. It runs the chosen harness and then calls `/verify`.
The resource server collects `/logs/agent/answer.txt`, closes the agent sandbox, and runs the original
upstream evaluator in a separate Python sandbox. No rubrics, reference answers, or judge credentials are staged
in the agent sandbox. Agent network access is limited to the Gym policy-model host.

Set `SWE_ATLAS_HARNESS_BUNDLE` to a local tarball prepared for `harness_agent`. It must contain
`nemo_gym/`, the selected `responses_api_agents/<agent>/` package, and the harness's runtime and dependencies.
Set `SWE_ATLAS_HARNESS_PYTHON` to the Python command inside the sandbox if it differs from `python3`.
The bundle must run in the upstream task image, including Alpine/musl images. Keep task assets and credentials
out of this archive. Runtime installation belongs to the harness, not this resource server.

Configure an OpenAI-compatible policy endpoint with the usual Gym values
`policy_base_url`, `policy_model_name`, and `policy_api_key` (an unused placeholder for an unauthenticated server).
Set `SWE_ATLAS_JUDGE_BASE_URL` and `SWE_ATLAS_JUDGE_MODEL` for the rubric judge. Set
`SWE_ATLAS_JUDGE_API_KEY` if it requires authentication. The default is an unused placeholder.
It is passed only to the evaluator process after dependency installation.

```bash
POLICY_MAX_OUTPUT_TOKENS=16384  # Example only. Match the endpoint and available context.
gym eval run --benchmark swe_atlas_qa --split benchmark --output runs/swe-atlas/rollouts.jsonl \
  +use_absolute_ip=true \
  "+policy_model.responses_api_models.vllm_model.extra_body={max_tokens:${POLICY_MAX_OUTPUT_TOKENS}}"
```

Set an explicit output budget for each model call. The existing model-server `extra_body.max_tokens` setting
supplies it when the harness omits a cap, including Cline and Fabric Mini-SWE/DeepAgents. Choose a value supported
by the endpoint and remaining context. A small endpoint default can truncate tool arguments before required fields
are emitted. This setting requires a backend that accepts `max_tokens`.

Native request values take precedence over `extra_body`. For harnesses that send their own cap, also configure
the supported `agent_kwargs` field: Hermes uses `max_tokens`; SimpleStrands, OpenCode, Pi, Prime, OpenClaw,
and KiloCode use `max_output_tokens`. Native limits may clamp these values. Check the upstream request's effective
cap and `finish_reason` when validating a harness. A request-level Responses `max_output_tokens` budget is not
supported by every harness and is separate from a per-call limit.

Also configure the OpenSandbox provider using its documented connection settings. Set `SWE_ATLAS_GYM_MODEL_HOST`
to this machine's IP reachable from the task sandbox. The model server defaults to port 8000. For another port,
set `+policy_model.responses_api_models.vllm_model.port=9000`; the sandbox URL follows that setting.
The default configuration routes the harness through the Gym model server,
which proxies the configured policy endpoint. The judge endpoint must be reachable from the verifier sandbox.

The default harness is OpenCode. Select another built-in harness and its corresponding `agent_kwargs`, for example:

```yaml
swe_atlas_qa_agent:
  responses_api_agents:
    harness_agent:
      agent: pi
      agent_kwargs:
        model: ${policy_model_name}
        command: pi
        timeout: 10800
```

The default OpenCode configuration allows the required answer directory and temporary scratch directory inside the task sandbox:

```yaml
agent_kwargs:
  opencode_config:
    permission:
      external_directory:
        "/logs/*": allow
        "/tmp/*": allow
```

Keep these permissions when replacing `agent_kwargs`. Native external-directory permission prompts can otherwise
auto-reject writes to `/logs/agent/answer.txt`. This does not change OpenCode's global permission defaults.

For DeepAgents, configure its [filesystem root](../../responses_api_agents/nemo_fabric_agent/README.md)
to honor absolute task paths inside the sandbox. For Mini-SWE, set `agent_kwargs.fabric_config.runtime.max_turns`
to the study's turn budget so its native limit does not end the rollout before submission.

Use the same tasks and grader when comparing harnesses. Their CLI commands, model configuration, dependencies,
and libc compatibility remain harness-specific.

## Scoring and validation

`reward` is the upstream binary strict-pass result. `agg_score` is the fraction of scored rubrics passed.
The upstream evaluator excludes unscored rubrics when computing strict pass. Gym preserves that behavior and reports
`num_scored`, `num_unscored`, and `all_rubrics_scored` so incomplete judging is visible.
Missing or empty agent answers receive zero. Infrastructure and malformed-verifier failures set
`mask_sample=true` and `evaluation_completed=false`.
The collected answer is retained in the rollout so the grader's exact input can be inspected.

```bash
pytest resources_servers/swe_atlas_qa/tests
```

The checked-in `data/example.jsonl`, `data/example_metrics.json`, and `data/example_rollouts.jsonl` satisfy Gym's
example-data validation contract. The five distinct task IDs end in `998`, `999`, `99a`, `99b`, and `9ce`.
The recorded runs used Claude Code with `azure/anthropic/claude-opus-4-6` through `harness_agent` on OpenSandbox,
with NVIDIA inference and the upstream evaluator. These examples come from the saved baseline run and do not
provide an unbiased benchmark score.
They are the first five task IDs with a nonempty answer and complete judging in that saved run, using the first
complete attempt for each task (attempt 2 for `9ce`, attempt 1 for the others).

Exports retain the exact answer, score and rubric counts, request, usage, and complete recorded response,
including intermediate assistant messages, tool calls, and tool results.
The metrics describe the five input rows and are generated with:

```bash
gym dataset collate "+config_paths=[resources_servers/swe_atlas_qa/configs/swe_atlas_qa.yaml]" \
  +mode=example_validation +output_dirpath=/tmp/swe-atlas-examples
gym env test +entrypoint=resources_servers/swe_atlas_qa +should_validate_data=true
```

Supply the endpoint and harness configuration described above when collating. The full dataset, private grading assets,
and runtime bundles remain uncommitted. The server starts with `verified: false` until benchmark review and baselining.

## Attribution

The dataset and evaluator are from Scale AI's SWE Atlas, licensed under Apache-2.0. Preparation retains the upstream
`LICENSE` with the downloaded source. Cite *SWE Atlas: Benchmarking Coding Agents Beyond Issue Resolution*
([paper](https://arxiv.org/abs/2605.08366)). Repository task images and bundled harness runtimes retain their respective
third-party licenses.

## Validation scope and tested configuration

The `cmunley1/swe-atlas-harness-validation` handoff has 13 × 3 real Opus 5.5 smoke runs with accepted answer
files and complete rubric grading. The harnesses are Claude Code, Cline, Codex, DeepAgents, Hermes, KiloCode,
Mini-SWE, OpenClaw, OpenCode, Pi, Prime, Simple Strands, and Terminus 2. These runs used local fixes and
prepared runtimes. They do **not** establish clean installation with public defaults or training validation.
The full-cohort Opus 4.6 leaderboard score has not yet been reproduced.

Tested CLI pins were Claude Code `2.1.293`, Codex `0.142.1`, Cline `3.0.55`, KiloCode `7.4.15`,
OpenClaw `2026.6.11`, OpenCode `1.17.11`, Pi `0.80.2`, and Prime `0.7.0`.
Select these explicitly when matching the tested setup. The Claude Code preset leaves its version unset,
and the Codex preset selects `0.144.4`.

The accepted configurations also used:

- Hermes `max_turns: 90`, Mini-SWE `fabric_config.runtime.max_turns: 250`, and Simple Strands
  `max_turns: 500` with `repo_dir: .`.
- Terminus `temperature: null` for the tested endpoint, and OpenClaw `thinking: off` with its agent workspace `/app`.
- A proxy output cap of `128000` for Cline, Mini-SWE, and DeepAgents, subject to the endpoint/context constraints above.
- The [documented Fabric adapters and runtime setup](../../responses_api_agents/nemo_fabric_agent/README.md)
  for Mini-SWE and DeepAgents, plus the filesystem and OpenCode permission settings described above.

For training, use independently authored tasks and keep SWE Atlas held out.
