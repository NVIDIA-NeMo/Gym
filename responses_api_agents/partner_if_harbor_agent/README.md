# Partner IF Harbor agent

Runs a native Harbor task (OpenHands 0.62 on OpenSandbox) and sends the captured
visible evidence to the shared `partner_if` grader. OpenHands calls the configured
`model_server` from inside the sandbox, so set `use_absolute_ip: true`.
