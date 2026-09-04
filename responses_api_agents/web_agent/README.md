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

Verification is episode-scoped. The agent retains immutable screenshot
evidence, closes the browser, and then calls the external WebVoyager judge.
Transient judge failures can therefore be retried without replaying live-site
actions. Every seeded rollout returns its artifact session ID; finalized video
references are returned when recording is enabled.
