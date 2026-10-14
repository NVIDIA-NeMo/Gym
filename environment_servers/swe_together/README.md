# SWE-Together

This environment runs the 109 pinned SWE-Together tasks with an OpenCode candidate,
a single model call for each user-simulator consultation, and a separate agentic
judge. Use the [benchmark recipe](../../benchmarks/swe_together/README.md) to prepare
the task assets and runtime manifest, configure the model routes, and collect rollouts.

The server owns each episode's task and grading sandboxes. Agent server processes
are shared; each episode seeds and closes its own agent sessions. After setup,
one interaction deadline covers candidate activations, repository inspection, and
simulator calls. Setup, grading, and cleanup have separate budgets. Native OpenCode
continuation receives only each new user message. The simulator cannot end an
episode: the environment stops on the interaction budget, 15 resumes, or four
consecutive no-ops.

Grading begins after confirmed candidate shutdown. The environment restores the
complete initial task working directory and every repository discovered during
setup into a fresh sandbox running the same pinned image. This includes ignored
and untracked files, binary data, nested repositories, file modes, symlinks, and
Git state. Initial roots are deduplicated and their contents replaced, reproducing
deletions too. Transfer uses the image's existing user and retains existing root
directories, so an owned `/workspace` does not require write access to `/`.
Unreadable or unwritable task paths fail explicitly with the path and user identity.
An unwritable baseline subtree is retained only after its complete membership,
bytes, modes, ownership, modification times and symlink targets match the archive.
Differences and hardlinks involving such a subtree fail before replacement.
The archive moves in 4 MiB chunks with SHA-256 checks; its manifest, chunks, and
restore receipt remain in the episode artifacts. The filesystem boundary is the
initial task roots; system directories and live processes are supplied by the
pinned image, and harness runtime and credential directories are excluded.
Repository diffs are retained for simulator context and judge evidence; they are
not used to reconstruct the submission.

Only after replication does the environment add the frozen rubric and test assets
to the grading sandbox. The judge runs once, closes, and then its verdict is read.
A failed or refused invocation remains an unmeasured, masked result with its
original outcome and separate cleanup receipt. Infrastructure failures are never
converted into candidate zeroes. The environment parses task memory declarations
explicitly and retains the declared resource limit.

Correctness uses the frozen rubric weights and a 0.85 pass threshold. User
Correction and Intent Coverage are reported separately using accepted simulator
messages, including simulator actions and original-intent progress. `judge_all`
scores every smoke submission; the `reference` profile preserves the source
short-patch eligibility rule. These profiles are recorded in every result.
Malformed interaction-model output remains an unmeasured metric; raw responses
and normalized coverage tables are retained as diagnostics. An explicit valid
no-match result still scores zero.

Artifacts under `artifact_root/<episode>/` include task/runtime provenance, ordered
activations, exact simulator inputs and decisions, incremental diffs, frozen state,
judge outcome/verdict, interaction metrics, and owner/borrower cleanup receipts.
Task assets remain host-only until each role needs them. Runtime mounts are private
and read-only, and sandbox network policy must default to deny.

The task manifest, simulator, prompts, and scoring mechanics are adapted from
[Togetherbench/SWE-Together](https://github.com/Togetherbench/SWE-Together) at
`891d19eb4b3a64a47c3d49bbd066a311e0133254`; see `UPSTREAM_LICENSE`. The simulator's
old second model-call fallback is deliberately omitted: one consultation produces
one non-agentic model response, with local parsing of that response.
