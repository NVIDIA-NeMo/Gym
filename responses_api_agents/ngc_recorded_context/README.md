# Recorded-context API agent

Bounded Chat Completions continuation against a frozen tool-data contract. Configuration, dataset preparation and run instructions are in
`benchmarks/partner_if/recorded-context.md`. The canonical task and pinned actor/tool
fixtures are supplied by the authenticated data package.

Tool requests use frozen data and a trusted file-search process; no
model-selected shell or source code executes. Unsupported operations exclude
the attempt. Complete
captures use partner_if's shared registered judge. Existing raw API panels stay
outside the benchmark package. Core Gym dependencies supply aiohttp, pydantic
and jsonschema. Component dependencies install the pinned `ripgrep-bin` executable
for readonly searches; no system ripgrep or private runtime installation is required.
Both entrypoints verify that executable at startup and fail before inference if
the component dependency is missing or unusable. A directory search is excluded
when its inventory includes uncaptured file contents within the requested glob
scope; an inventory listing alone does not require those contents.

The `expanded-v3` profile also supplies replay-generated session metadata and
an actual-message transcript for skills that inspect their own session. These
files identify the API replay, not a native Claude process or a completed
review. Older `recorded-v1` and `expanded-v2` semantics remain available.

`expanded-v4` additionally implements a bounded joined readonly Explore worker
with the exact parent policy model. Actual nested calls and findings are
captured; all calls count against the shared limit. No native review runtime
or approval is simulated.


## Ordering-prefix boundary

The existing `single_agent_turn` Environment Server uses its opt-in verifier continuation loop.
The dispatcher materializes the flat registry row into the server's typed episode
request and unwraps the typed result for collection.
The `partner_if` Resources Server selects the ordering-prefix boundary declared by the task metadata.
Shared verification runs after every new
response/tool batch. Premature advice stops with reward 0; a verified same-version
neighboring-namespace check with no premature advice stops with reward 1. No
decisive event at the cutoff, unsupported calls and judge errors are excluded.
No final answer is invented; its artifact is explicitly absent before one exists.

`first_answer` remains ReplayConfig's default for reconstructing historical runs.
Changing the evaluation boundary during a live resumption is refused. Explicit
`regrade_saved_prefix` plus its SHA256 digest creates new receipts from saved actor
responses and forbids all fresh actor requests. Original captures are untouched.


## Agent sessions and Model Server

The default benchmark config runs `native_agent.py`. This is a private,
in-process, frozen-tool replay pairing with `partner_if`, not a general shell
harness. It supports evaluation and measured observations; retained-evidence
conformance remains partial for providers emitting encrypted reasoning. It does
not claim token-ID capture, training, checkpoint restoration, Terminal-Bench,
MCP or task-sandbox support. Its version is the checkout plus the recorded source
hashes; the benchmark composition is version 0.4.0.

Seed `/v1/agent_sessions`, activate the rollout-prefixed `/v1/responses` once,
then close. Unknown/closed sessions and changed duplicate activations are refused;
identical retries return the retained response without another generation. The
adapter rejects sandboxes, required external grants and harness-continuation
requests; optional grants are unused. Sessions require one server worker.

A step uses a checksum-verified saved prefix only inside its configured artifact
root, with exact same-model request reconstruction. Parent and joined Explore
calls share the configured call ceiling and always go through Gym's Model Server,
with captured response IDs and session ownership. Close cancels unfinished work
and returns its actual observations. There is no fallback to a host shell.
Tool-call IDs are scoped to the parent or worker conversation, so reused IDs
cannot attach a worker's result to a parent call.

The generic Environment Server closes each fresh agent session before invoking the shared verifier.
An undecided prefix returns a standard Responses request through `next_agent_request`;
its `metadata` carries `ngc_replay_parent_turn_limit`, `ngc_replay_saved_prefix` and
`ngc_replay_saved_prefix_sha256`. The replay adapter removes these controls before
checking the frozen actor input or making a policy request. The first activation
defaults to one parent turn. No separate replay Environment Server is installed.
The verifier checks receipt checksums inside `ordering_artifacts_dir` before grading.
`ReplayConfig.first_answer` bounds each engine step; it does not change the
verifier's prefix scoring boundary. The final episode response sums
only fresh known usage; missing counts stay unknown. Sampling and call limits
come from `replay` config, not generic single-response CLI overrides.
The pairing uses a separate Model Server configuration to preserve those settings
for both parent and Explore calls when other cohort routes override sampling.

The `app.py` entrypoint is retained for historical saved-capture grading using
`benchmarks/partner_if/config_ngc_replay_audit.yaml`; it refuses fresh policy
requests. It is not the live path. Shared synthetic tests live under
`benchmarks/partner_if/tests` and are selected by this component's pytest config.


The current profile enables readonly shell conditionals and the literal stderr
null sink (`shell_control_flow`), Gym leading-think transport decoding
(`gym_reasoning_decode`), and grounded reads of the optional absent pre-review
file (`missing_runtime_reads`). Historical defaults remain false. These are tool
coverage/transport repairs; the prompt and ordering rubric do not change. New
requests to uncaptured files, native reviewer execution, unsupported shell
expansion and conditional probe execution remain explicit exclusions.
