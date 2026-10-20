# SWE Atlas Codebase Q&A

Evaluates the 124 Codebase Q&A tasks from [Scale AI's SWE Atlas](https://github.com/scaleapi/SWE-Atlas)
through `harness_agent`. This subset asks an agent to investigate an existing repository and answer a question.
It does not include the Test Writing or Refactoring subsets.

## Preparation

```bash
python -m resources_servers.swe_atlas_qa.prepare
```

`prepare.py` fetches upstream revision `c6673fe259c3493a2f834266abf63247a5d1507c`, checks the revision,
rejects modifications to the task assets, and generates `data/benchmark.jsonl` and `data/example.jsonl`.
All downloaded and generated data is gitignored. Rows contain the upstream instruction, task ID, image, and
working directory. Rubrics, evaluator prompts, and reference answers remain in the server's private source cache.
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
gym eval run --benchmark swe_atlas_qa --split benchmark --output runs/swe-atlas/rollouts.jsonl \
  +use_absolute_ip=true
```

Also configure the OpenSandbox provider using its documented connection settings. Set `SWE_ATLAS_GYM_MODEL_HOST`
to this machine's IP reachable from the task sandbox and optionally `SWE_ATLAS_GYM_MODEL_PORT` (default 8000).
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

No downloaded task rows or rollouts are committed. Run preparation before local data validation and retain live
rollout evidence separately. The server starts with `verified: false` until benchmark review and baselining.

## Attribution

The dataset and evaluator are from Scale AI's SWE Atlas, licensed under Apache-2.0. Preparation retains the upstream
`LICENSE` with the downloaded source. Cite *SWE Atlas: Benchmarking Coding Agents Beyond Issue Resolution*
([paper](https://arxiv.org/abs/2605.08366)). Repository task images and bundled harness runtimes retain their respective
third-party licenses.
