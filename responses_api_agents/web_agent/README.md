# Web agent

`web_agent` is a multimodal rollout loop over Gym's normalized web task,
observation, action, verifier, and artifact contracts. It does not launch or
control Chromium directly.

For WebVoyager, the agent supports two model protocols on one environment:

- `nano_omni_toolcall` reads structured Responses tool calls;
- `qwen_xml_computer_use` builds the Qwen screenshot history and parses XML
  `computer_use` calls.

Both adapters normalize output to `WebActionProfile.COMPUTER_USE` and send it
to the `visual_browser` resource server. Browser launch, proxy/CAPTCHA,
PyAutoGUI execution, screenshots, and recording remain outside the agent.

Invalid model syntax receives bounded retries and never executes arbitrary
Python. Policy failure remains a valid zero-reward sample. Browser-provider,
proxy/CAPTCHA, model transport, and judge failures set `mask_sample` and are
routed to recovery instead of training.

Exhausting the model context budget is a truncated policy outcome, not a
transport failure. The agent stops requesting actions and runs normal
evaluation before closing a stateful browser session. For WebVoyager, the
external judge still runs after close using retained evidence. Completed
actions, usage and recordings are preserved; the evaluator determines the
reward rather than the agent forcing a zero. A first-turn context rejection
has no generated output or usage. Evaluator failures remain masked. Responses
include `truncation_reason` for context or output-budget stops.

For Nano Omni, the model server's reasoning and tool-call parsers are the
protocol boundary. The agent validates parser-produced structured calls but
does not decode nested action strings, repair delimiters, infer aliases, or
silently change out-of-range action arguments.

Response call-count admission is configurable through
`nano_omni_max_tool_calls`, with the existing default of 8. A reference profile
may set it to `null` to accept the complete valid call list without changing
its arguments. Individual action validation and operation timeouts still apply.
The visual-browser resource server independently revalidates each call batch.
Its `max_tool_calls` must match the selected agent limit; for an uncapped
reference recipe, explicitly set both limits to `null`. The resource default
remains eight, and argument validation remains active in either mode.

With `nano_omni_retry_invalid_tool_calls: false`, invalid structured calls end
interaction and proceed to live-state evaluation. Complete validated calls
preceding the first invalid call execute in order before evaluation; the invalid
call and everything after it are not executed. The full model response remains
unchanged in the rollout. This does not split or repair an invalid `computer`
call's nested action list, and call-count and terminal-order limits still apply.
Missing calls still use the
bounded parse-retry budget. Reference retries should also disable
`nano_omni_parse_retry_feedback` and leave `nano_omni_parse_retry_temperature`
unset, so the request is repeated without correction text or a sampling change.
These are profile choices; other profiles retain their existing defaults.

Nano observations add the current screenshot, step and tab context without
echoing the preceding assistant action as new user text. The original parsed
assistant turn stays in history unchanged, and the resource observation retains
its action/error fields for diagnostics. Profiles that explicitly continue
after execution errors still receive the actual error as feedback.

Verification is episode-scoped. The agent retains immutable screenshot
evidence, closes the browser, and then calls the external WebVoyager judge.
Transient judge failures can therefore be retried without replaying live-site
actions. Every seeded rollout returns its artifact session ID; finalized video
references are returned when recording is enabled.
