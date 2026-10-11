# SWE-Together

Evaluate interactive coding tasks with a simulated user and a fixed Claude Code judge. Resources owns task sandboxes and simulator state; `single_agent_multi_turn_user_sim` owns episode execution. Candidates must support native conversation continuation and ordered visible events.

Use the public recipe in [benchmarks/swe_together](../../benchmarks/swe_together). Its trusted preparation command obtains the pinned upstream task assets and runtime manifest. Import `configs/swe_together.yaml`, set `asset_root` to the canonical source's `tasks/`, `artifact_root` to the evidence directory, and `runtime_manifest_path` to the prepared manifest. The provider must support private shared-storage mounts. Candidate and judge `sandbox_config.provider_options.network_policy` require default-deny egress with their respective Gym model route allowed.

The manifest names local, SHA-256-pinned Python, OpenCode, ripgrep, and Claude artifacts. A trusted helper uploads each verified hash once through the filesystem API, unpacks the complete Python installation, and prepares a distinct read-only `/opt/gym-runtime` directory for each trial and role. Candidate mounts contain Python/OpenCode/ripgrep; judge mounts contain Python/Claude. Runtime preparation precedes repository snapshots and candidate seeding. Hidden task assets stay on the Resources host.

`/next-user-turn` replays identical zero-based activation requests and returns `kind: continue` with the next user message, or `kind: stop`. The reference simulator retains upstream persona, tools, prompts, history, cursor, 15-resume limit, and four-consecutive-no-op stop rule. An `activation_timeout` produces the upstream interruption message. Permission rejection follows the source completion-consultation path.

The Environment Server supplies independent setup, interaction, and grading deadlines. Resources enforces the immutable interaction window and discards late simulator replies before committing their history. Candidate budget exhaustion preserves the patch for grading. Auxiliary transport failures remain failures.

Canonical `agent.kwargs` are forwarded as `harbor.agent-kwargs.v1` runtime policy. Snapshots use independent indexes, preserve prepared edits and untracked files, and retain all repositories and binary artifacts. The repository containing the task workdir is first; the judge uses that repository's patch section. Git display configuration, external diff programs, and text converters cannot control captured patch text.

The fresh judge sandbox runs [claude_code_sandboxed](../../responses_api_agents/claude_code_sandboxed) with Claude Code 2.1.108 and the source turn/time limits. Frozen goal weights and rounding determine `judge_score`; the pass threshold is 0.85. The judge-reported score/verdict remain separate evidence, including the upstream override of `gameable` when goal results are present. `scoring_profile: reference` retains the source skip for empty/tiny patches; `judge_all` is an explicit diagnostic deviation. Failed judge measurements carry the standard `judge_failed` classification.

Production aggregation reports planned/returned/graded/masked/missing counts, measured judge mean, pass@1, complete-repeat stability, pass2, and User Correction. Missing auxiliary measurements stay absent; zero simulator messages have measured User Correction zero. The separately named zero-filled mean preserves upstream compatibility. Default planning is 109 tasks × 2 repeats.

Artifacts include source/image/runtime pins, repository baselines and every incremental/cumulative patch, exact simulator input and decisions, acceptance/discard times, auxiliary calls, raw/derived judge verdict, judge transcript and close receipt, optional canonical test reward, and sandbox close receipts. Task creation failures still close all owned sandboxes; failed stops retain ownership for retry. Runtime staging is removed after both roles stop.

Source-adapted protocol modules and prompts retain upstream Apache-2.0 attribution and [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
