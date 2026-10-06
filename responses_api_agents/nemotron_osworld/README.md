# Nemotron OSWorld agent

This agent preserves the validated Nemotron-Omni OSWorld interaction loop over
`resources_servers/osworld`: observe a screenshot, run the pinned Nemotron prompt
and parser, execute the parsed pyautogui action, retain the faithful action history,
and pass that history to the official OSWorld evaluator.

`app.py` keeps the non-context-managed evaluation path. The pinned reference agent
still owns prompt/history construction and parse recovery, while its synchronous
network seam is bridged back to Gym's shared aiohttp `ServerClient`.

`cc_app.py` is the training path. It uses
`nemo_gym.context_management.ContextManagedResponsesClient` with ordinary
`responses_api_models/vllm_model`:

- `_ng_rollout_id` is mandatory and is preserved in full, including an optional
  `_a<32 lowercase hex>` attempt suffix.
- Every selected action is captured externally. Continuations declare the selected
  parent response ID; an intentional rewrite starts the next parentless
  `<owner>_sN` capture segment.
- Provider history, provider truncation, streaming, inline token arrays, and
  automatic transport replay are disabled.
- `/run` returns the official verifier reward, the final selected model
  `response.id`, and a top-level token-free `context_compaction_result`. Media
  occurrences stay ordered while each raw media asset is exported once.
- Model, action, and whole-rollout operations are bounded. Definite rollout-local
  failures do not affect sibling rollouts, and the sandbox is released on every
  post-seed exit.

## Configuration

Use the dormant CC profile:

```text
responses_api_agents/nemotron_osworld/configs/nemotron_osworld_cc.yaml
```

It loads `responses_api_models/vllm_model/configs/vllm_model.yaml`, explicitly
keeps `return_token_id_information: false`, and uses the nested
`context_history.policy.config.images` schema. NeMo RL owns external token capture
and TransferQueue publication; there is no dedicated CC model server.

Compose it with:

```text
resources_servers/osworld/configs/osworld.yaml
resources_servers/osworld/configs/opensandbox_osworld.yaml
```

The regular non-CC profile remains:

```text
responses_api_agents/nemotron_osworld/configs/nemotron_osworld.yaml
```

## Licensing

Code and data: Apache-2.0.
