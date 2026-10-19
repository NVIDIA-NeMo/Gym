# SWE-Together

Run all 109 interactive coding tasks with OpenCode, a simulated user, and a fixed Claude Code judge. The reusable composition is `opencode.yaml`; choose model routes and a sandbox deployment independently.

The recipe pins the [upstream tasks and prompts](https://github.com/Togetherbench/SWE-Together/tree/891d19eb4b3a64a47c3d49bbd066a311e0133254), task image digests, OpenCode 1.15.13, and Claude Code 2.1.108. The default model roles are GPT-5.5 with high reasoning for the candidate, Gemini 3.1 Pro Preview for simulation and interaction metrics, and Opus 4.6 for correctness. Set each role's `*_BASE_URL`, `*_API_KEY`, and `*_MODEL` variables in [models.yaml](models.yaml) to use your endpoints. Real credentials stay on the model-server host.

## Prepare and run

Run these commands from the Gym root, in its installed Python environment. Preparation downloads the pinned public task source and verifies every task asset. Runtime preparation downloads and verifies four artifacts on the trusted environment host, including a complete Python installation for images that lack Python.

```bash
export SWE_TOGETHER_WORK_DIR="$PWD/outputs/swe_together"
python -m benchmarks.swe_together.prepare
python -m benchmarks.swe_together.prepare_runtime \
  --output "$SWE_TOGETHER_WORK_DIR/runtime"
```

For an existing upstream checkout, set `SWE_TOGETHER_SOURCE`. For an offline runtime cache, pass `--seed-manifest /path/to/manifest.json`; the pinned digests are still verified. Keep the prepared source and runtime manifest accessible to the environment server.

Configure a provider that supports private shared-volume mounts and externally enforced network policies. [opensandbox.yaml](opensandbox.yaml) composes the existing OpenSandbox provider. Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and `OPENSANDBOX_SHARED_STORAGE_PATH` for that deployment. Set `SWE_TOGETHER_POLICY_HOST` and `SWE_TOGETHER_JUDGE_HOST` to independently reachable model-server interfaces and set their respective `SWE_TOGETHER_*_EGRESS_TARGET` values to the addresses the provider should allow. Each sandbox role must reach only its own fixed model route. Keep simulator and interaction listeners private. Use role-scoped proxies when deploying the model servers behind a gateway, and reject arbitrary model selection and server-side tools there.

The model-server host variables select the listener interfaces as well as the addresses used by Gym. Use distinct private interfaces that are local to the server host and reachable from their respective sandboxes. If your deployment exposes role-scoped proxies instead, keep the model listeners private and advertise the proxy routes through your deployment configuration.

If the proxy validates model names, map OpenCode's fixed `dummy_model` alias to `POLICY_MODEL` and Claude Code's fixed `claude-opus-4-6` alias to `JUDGE_MODEL`. Each route must reject every other model selection; the aliases do not enable model switching.

Preserve `X-Session-Id`, `X-Session-Affinity`, and
`X-OpenCode-Assistant-Message-Id` through those proxies so model-call capture can
correlate requests with native conversation turns. These are evidence headers,
not credentials.

```bash
gym env start \
  --config benchmarks/swe_together/opencode.yaml \
  --config benchmarks/swe_together/opensandbox.yaml

# In a second shell with the same environment, after the servers are ready:
gym eval run --no-serve \
  --config benchmarks/swe_together/opencode.yaml \
  --config benchmarks/swe_together/opensandbox.yaml \
  --input benchmarks/swe_together/data/full109.jsonl \
  --output "$SWE_TOGETHER_WORK_DIR/results.jsonl" \
  --num-repeats 2 --concurrency 12
```

On hosts with strict thread limits, set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before starting the servers. Replace the provider config with your deployment's equivalent if needed. Keep the same composition for startup and collection. `CANDIDATE_CONCURRENCY` defaults to 32 and should cover total concurrent episodes sent to that agent instance.

## Runtime and interaction

The environment provisions the verified Python runtime for task inspection and
archive transfer, using private read-only mounts at `/opt/gym-runtime`. It does not
select or install an agent harness. Each agent stages its own pinned executable at
seed; this recipe supplies local binaries from the prepared cache because task
sandboxes have restricted egress. With another deployment, omit the local binary
paths to use each adapter's normal install-at-seed path. Runtime preparation is a
recipe convenience: keep its binary paths accessible to the agent servers and its
Python manifest accessible to the environment server.

The OpenSandbox composition renews owned sandbox leases hourly so the shared runtime helper remains available during runs longer than one day. Renewal does not extend episode or command deadlines; borrowed agent handles do not renew leases.

The `swe_together` environment owns task sandboxes, user simulation, and verification. It sends zero-based increasing activation IDs to the candidate adapter and consults the user simulator directly after each turn, using a deterministic summary of task progress. OpenCode resumes one native conversation with only the new user input. Setup, each activation, grading, and cleanup have separate limits; the environment’s 4,800-second interaction budget starts after setup and includes simulator calls and waits between turns. OpenCode allows 300 seconds per finalization phase, including native artifact collection; the environment allows 1,200 seconds for cleanup and the activation response after interaction expires. Each request carries relative `timeout_seconds`; agents enforce local monotonic deadlines, including queueing and staging. HTTP server processes are shared across episodes; each episode owns separate candidate and judge sessions. The candidate is closed before its task files are copied into a fresh sandbox for the judge; raw verdicts, frozen task state, patches, interaction metrics, and cleanup receipts are retained under the artifact directory.

The environment translates upstream task tool restrictions into `tools.deny.v1` denied tools. The adapter applies those restrictions using OpenCode’s native permission keys; upstream’s nested `permission.tools` declarations do not enforce the intended denials in the pinned release. Network policy provides external enforcement. The recipe uses Gym's existing OpenAI-compatible model transport, an explicit 262,144-token context limit, a 32,000-token output cap, and the same fixed model for auxiliary calls. These settings and restricted egress require separate baseline validation before leaderboard comparisons.

## Smoke qualification

Add `--config benchmarks/swe_together/smoke.yaml` after `opencode.yaml` in both commands and use `--num-repeats 1`. This gives the candidate 600 seconds and grades even empty submissions with the real judge. Smoke checks lifecycle and grading health; a correctly measured zero score is a healthy result.

```bash
gym eval health-check "$SWE_TOGETHER_WORK_DIR" \
  --rollouts-file "$SWE_TOGETHER_WORK_DIR/results.jsonl"
```

Use Gym's standard health report. The collector combines the episode envelope's
cumulative native trajectory with captured model calls. The result retains
ordered activations, simulator decisions, grading results, and cleanup receipts.
Episode usage sums activation usage; any missing activation count leaves the total unknown.
Missing evidence remains unobserved, and a failed check remains failed. Check
collection completeness against the selected input and retain every failed attempt;
smoke qualification does not establish benchmark accuracy or permission to retry
graded failures.

## Agent swapping

The recipe creates `swe_together_candidate` and `swe_together_judge` through
`_inherit_from`. Change those agent instances and their model routes in a final
config overlay. The candidate must advertise native conversation continuation,
ordered observations, and the requested runtime policy. The judge must support
seeded single-turn Responses calls and confirmed cleanup. See the
[environment contract and ownership notes](../../environment_servers/swe_together/README.md).
Single-turn support alone is insufficient for a candidate swap; seed rejects that
mismatch explicitly. Use the repository's `scripts/run_harness_conformance.py` and
`inspect_harness_conformance.py` runbook for supported adapters before a live smoke.
When changing `_inherit_from`, set the unused base adapter instance to `null`
in the overlay so it is not launched as an extra server.
