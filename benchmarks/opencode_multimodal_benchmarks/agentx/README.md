# Agent-X reference-answer evaluation

605 non-generative tasks: 494 image tasks and 111 video tasks. [AgentX.json](../manifests/AgentX.json) pins the dataset revision, file hashes, selected tasks and exclusions. Video preparation samples approximately one frame per second, up to 128 frames.

The manifest also pins each video's ordered JPEG frame hashes and sampling
metadata, generated from the checksum-verified source MP4. Both source conversion
and portable import enforce these identities across the full split before applying
a smoke limit. Replacing, reordering or duplicating frames, or changing their
timestamps, fails validation even if the bundle checksum is updated.
The decoding policy uses OpenCV 4.11.0.86 with FFmpeg and JPEG quality 95; its
decoder/encoder build is recorded in the manifest. A different build must produce
the same bytes or preparation fails. A validated portable bundle needs neither
the source videos nor a video decoder at import time.

## Task and score

Prepared rows contain the question and image inputs in
`responses_create_params.input`. Reference answers and category labels stay in
`verifier_metadata`; they are not model inputs. No external task tools or sandbox
are required by the verifier. The catalog configuration uses `simple_agent` for
image-question answering; a tool-using harness is a separate configuration choice.

The independent `agentx-reference-judge-v4` scorer compares only the candidate and
reference answers. It sends neither the question nor reference justification to
the judge. Objective verdicts are `equivalent`, `different`, or `unsure` (1, 0,
and 0 respectively). Subjective tasks retain continuous similarity scores in
[0, 1]. Malformed JSON and invalid scores are judge errors. This is not the
official GoalAcc evaluator; scores must retain the scorer version and judge identity.

Reasoning cleanup uses the shared tool-reward helper before grading. The saved
response remains unchanged. Empty final answers score zero.
Unresolved judge errors remain errors, not incorrect model answers. The reward
range is [0, 1], higher is better. Reports include benchmark-specific breakdowns;
record both the metric key and its scale when comparing results.

## Prepare and check

Run from the Gym repository root after obtaining the dataset:

```bash
gym eval prepare --benchmark opencode_multimodal_benchmarks/agentx \
  +prepare_script_args.source_root=/data/agentx
gym env validate opencode_multimodal_benchmarks/agentx
gym env test opencode_multimodal_benchmarks/agentx
gym env publish opencode_multimodal_benchmarks/agentx
```

Preparation writes `benchmarks/opencode_multimodal_benchmarks/data/agentx.jsonl`.
Prepared rows embed image bytes and carry the canonical source index as `task_id`.
Collation wraps them in typed `task_id`/`task_input` tasks.
The `visual_tasks` route selects `single_agent_turn`, which owns the agent session
and verification lifecycle; the source JSONL is unchanged.
The offline verifier fixtures use synthetic answers and scripted judge replies,
not a hosted judge or benchmark images. Publication checks are local validation,
not certification or an upload.

## Evaluate

Configure `policy_model`, a `judge_server` or `judge_endpoint`, and the sandbox provider.
The catalog defaults to OpenCode with WebSearch, WebFetch and the Agent-X web filter.
Evaluation defaults to `judge_model: gpt-5.5`, with a 4096-token judge budget.
Use a routed deployment name such as `azure/openai/gpt-5.5` when required by your provider. Keep credentials in the runtime environment.
See [run configuration](../README.md#run-configuration) before running:

```bash
gym eval run --benchmark opencode_multimodal_benchmarks/agentx \
  --model-type openai_model --split benchmark --limit 5 --num-repeats 1 \
  --output results/agentx_smoke.jsonl
```

Canonical rows are ordered image-first: a prefix limit of 494 or fewer covers
images only and does not exercise video handling. A smoke run marked complete
is complete only for its selected subset; check the Image/Video counts for coverage.
Remove the smoke limit to evaluate all 494 image and 111 video tasks.
Save the resolved configuration,
judge identity, sampling, budgets, timeouts, repeats and supported random seeds.
The prepared requests use temperature 0.6 and top-p 0.95; the catalog sets one
repeat. Agent/model limits are supplied by the run configuration.

The integration code is Apache-2.0; source dataset terms remain those of its
publisher. See [implementation provenance](../../../resources_servers/opencode_multimodal_benchmarks/third-party-notices.md)
for adapted evaluation code. No benchmark data is included in this entry.

The integration does not copy the upstream evaluator or its judge prompt.
Dataset access and redistribution remain subject to the publisher's terms.

## Harness pairings

| Harness | Configuration | Status |
| --- | --- | --- |
| `opencode_agent` | Default [`config.yaml`](config.yaml), plus model and sandbox-provider settings | Experimental; OpenCode 1.17.11, WebSearch and WebFetch enabled, one activation |

[`opencode.yaml`](opencode.yaml) remains a compatibility alias for the default config.

Configure `webfetch` and `websearch` independently under the pairing's
`visual_agent.responses_api_agents.opencode_agent.opencode_config.permission`
using `allow` or `deny`.
Both tools are allowed in this pairing, with `network_access: inherit`.
The preset also sets `OPENCODE_ENABLE_EXA=true` in the sandbox to expose native
WebSearch; allowing its permission alone does not activate it in OpenCode 1.17.11.
Bash and Python retain the sandbox's existing network access as well. Denying a
web tool prevents its use; it does not change sandbox egress. Record
both the network policy and tool settings with the score.

The pairing also loads the benchmark-local
[`web_filter.yaml`](web_filter.yaml)
overlay. Its JavaScript and JSON policy are staged with the existing
`sandbox_config.files` mechanism and loaded through the stock agent's `plugins`
setting. The same overlay is inherited by [`training.yaml`](training.yaml).
There are no changes to the shared agent, tool transports, judge or reward logic.

This is **best-effort answer-source filtering**, not network isolation. The plugin
blocks known Agent-X source URLs and narrow benchmark identifiers in native
WebSearch/WebFetch requests and Bash commands. It checks returned text, titles
and metadata; matching results become `No content available.`. For truncated
output it also scans the full saved file and overwrites rejected content before
returning the tool result. Clean results and native execution behavior are retained.
The policy never contains the task's reference answer. `[gym-web-filter]` records
in the OpenCode stderr log identify load events and blocked calls by tool, phase
and rule index without copying queries, URLs or page content into the log.

Unlisted mirrors, dynamically constructed URLs, encoded content, shell-created
files and concurrent reads can bypass this filter. It does not inspect arbitrary
files or all network traffic, and OpenCode may persist internal execution records
before the result hook. The filter is tested against the pinned OpenCode 1.17.11;
recheck hook behavior when upgrading. Record the policy and enabled state with
results; earlier unfiltered evaluations are a different setup. To disable this
overlay's plugin explicitly, override
`visual_agent.responses_api_agents.opencode_agent.plugins=[]`.

The OpenCode pairing uses the stock agent, shared media staging and
`single_agent_turn`. Its evaluation overlay disables token capture and auxiliary
tool rewards. The Environment Server owns activation, close and verification. See the
[agent README](../../../responses_api_agents/opencode_agent/README.md)
for runtime requirements and configurable limits.

See the shared [grading, deadlines and recovery contract](../README.md#grading-and-recovery)
for observed judging, failure outcomes and resumable-run requirements.


## Training

Compose [`training.yaml`](training.yaml) with the deployment's policy model and
sandbox configuration. It selects the separate `agentx_tool_rewards.py` training
wrapper over the independent evaluator, defaults to `gpt-4o-mini`, enables token capture and tool rewards, and
sets `forward_agent_evidence: true` on `single_agent_turn`. Routed NVIDIA
deployments use `azure/openai/gpt-4o-mini` for `judge_model`. The evaluation recipe
keeps token capture and tool rewards disabled and defaults to `gpt-5.5`.

Reward components retain the independent verifier score and the tool adjustment
separately. This training preset enables `tool_rewards.penalty_mode: proportional`:
failure, duplicate-call and shell-usage costs are multiplied by positive correctness,
with their combined penalty capped at that score. A nominal cost of 0.10 gives total
reward 0.891 at correctness 0.99 and 0.90 at correctness 1.0. Zero scores incur no
penalty. Existing positive-bonus gates and amounts are preserved; the preset offers
no positive bonuses. Shared tool-reward defaults remain unchanged.

A failed native WebSearch HTTP 429 remains a failed call but incurs neither failure
nor duplicate penalties. Its immediate retry is also exempt from duplicate penalties;
repeating a successful search remains penalized.
Judge failures remain masked; tool rewards do not replace a missing grade.
Reverification of tool rewards separately requires `forward_agent_evidence: true`.

The `agentx.py` evaluation entrypoint rejects tool-reward settings and returns
only correctness. The training wrapper records correctness, auxiliary rewards,
and total reward separately. In the shared `penalty_mode: default`, `correct_only: true`
suppresses adjustments below the correctness threshold. The proportional mode selected
here applies scaled penalties to partial scores too. Setting `correct_only: false`
continues to allow configured exploration bonuses.

When collecting with the Agent-X training preset, benchmark accuracy and modality
reports use the verifier's correctness component. Generic `mean/reward` statistics
and saved training rewards retain tool adjustments. This reporting behavior is
local to the Agent-X training wrapper; other tool-reward wrappers are unchanged.
