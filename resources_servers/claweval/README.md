# ClawEval with interchangeable harnesses

Resources owns ClawEval's native services, Pyxis task sandbox, tools, fixtures
and grading. `environment_servers/claweval` owns assistant/simulated-user turn
orchestration. An existing Agent Server owns the assistant loop. The new
configurations never call agent `/run` or compose `legacy_agent`.

| Run config under `benchmarks/claweval/` | Harness | Tools | User-turn history |
| --- | --- | --- | --- |
| `sessions_opencode.yaml` | OpenCode 1.17.11 | Required authenticated MCP | Resume one harness session |
| `sessions_simple.yaml` | Simple Agent | Direct HTTP grant | New session with complete history |

Both use the same native task references, services and grader. Swap agent
configuration, grant transport and continuation mode, without pair-specific
adapter code. A harness must reject unsupported required grants/continuation.
This branch includes the original ClawEval integration and the pending
multi-harness core/image and capture dependencies #5719 and #5833, including
#5719's prerequisite stack.

## Prepare and run

Use the existing native interpreter and staged assets. Set `CLAW_EVAL_ROOT`,
its full clean `CLAW_EVAL_REVISION`, `CLAW_EVAL_PYTHON`, `CLAW_EVAL_CONFIG`,
`CLAW_EVAL_FIXTURE_ROOT`, `CLAW_EVAL_SANDBOX_IMAGE` and `CLAW_EVAL_SANDBOX_DEPS`.
The native Pyxis runtime still needs a suitable Slurm allocation. Changed HEAD,
tracked/staged edits and untracked files are refused before import.
For an image-capable policy, set `model.input_modalities: [text, image]` in the
native profile (or the equivalent `model_overrides`). Native ClawEval otherwise
uses its text-only attachment fallback; that does not validate image transport.

```bash
python -m benchmarks.claweval.prepare --split general \
  --task-id T002_email_triage --output outputs/claweval/native.jsonl
python -m benchmarks.claweval.prepare_sessions \
  --input outputs/claweval/native.jsonl --output outputs/claweval/tasks.jsonl
```

Set the Resources dataset's `jsonl_fpath` to that file and keep
`taskset: claweval`. Typed rows contain public prompt, task ID, split and digest;
no agent reference, answers or grader fixtures. Seed checks the digest and
prompt against the original task.

Compose `sessions_simple.yaml` and a policy Model Server config in `run.yaml`;
keep credentials in `env.yaml`:

```bash
gym env resolve --config run.yaml
gym env validate --config run.yaml
gym eval run --config run.yaml --split example --output outputs/claweval/rollouts.jsonl
```

For OpenCode choose `sessions_opencode.yaml`, compose a sandbox provider, and set
`CLAW_EVAL_AGENT_IMAGE`. The harness runs in a separate agent-owned sandbox;
the original Pyxis task sandbox stays Resources-owned and accessible through
native tools. Built-in shell/file/web/subagent tools are denied; only
`claweval_*` MCP tools are allowed. Each token exposes only that task's exact
native tool names and schemas, and is revoked at Resources close.

## Lifecycle and evidence

The Environment Server seeds Resources, opens an agent session, invokes the
rollout-prefixed Responses endpoint, asks the native simulated user for the
next turn, and repeats within the native round/deadline limits. Agent close
must succeed **before** verification. Final cleanup also runs after failed
seed, inference, simulation and close. Infrastructure/content failures never
receive an ordinary zero score.

Verification reconciles the harness transcript with actual native tool
dispatches, then writes the native trace and calls the original grader.
Missing/ambiguous dispatches, unsupported content, unconfirmed termination and
exceeding the native model-turn budget prohibit grading. Successful retries
replay retained scores. `verification.json` retains the score on its native
**0–1 scale**, components, trace and source/configuration/task fingerprints.
Agent observations accompany the episode result. Native support-model calls
are not claimed as captured policy calls.
The episode response combines assistant outputs and usage across activations;
`activation_responses` retains the original receipts. Continuing sessions retain
their harness invocation identities and close trajectory for Gym's later capture
join. The native turn budget uses explicit call references or uniquely identified
persisted assistant turns when capture IDs have not yet been joined.

## Current limits

- Image prompt/tool content uses the shared image support from the runbook's
  dependency stack. Images remain image parts/MCP image content. Native audio
  and video are explicitly refused; full 101-task multimodal coverage is not
  claimed.
- The reference [OpenCode integration at `9316a63305e37ce4b4db176a21655dfe6a63ada3`](https://gitlab-master.nvidia.com/ehosseiniasl/opencode/-/tree/9316a63305e37ce4b4db176a21655dfe6a63ada3/integrations/claw-eval)
  uses 1.18.34 and a separate native-media bridge. This independently written
  integration uses Gym's reusable adapter; scores are not interchangeable.
- Identical repeated calls are accepted only when their results/media are also
  indistinguishable, with the unproven ordering recorded in verification.
  Repeated calls with distinguishable results are refused until their identities
  can be joined safely. Tool dispatches and the grading transcript are retained
  even if verification fails.
  Configure each harness's step cap for the native tasks being compared;
  exceeding a native budget does not produce a benchmark-valid score.
- OpenCode compaction is off. Policy calls preserve the rollout and optional
  training-token-capture prefix. RL training, full-suite parity and harness
  conformance require separate qualification; an evaluation smoke is not
  certification.
- Existing native-runner configs remain available for comparison. This PR
  adds no Terminal-Bench support or new sandbox-isolation claim.
