#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
export PYTHONUNBUFFERED=1

: "${OPENAI_BASE_URL:?Supply the Inference Hub API base URL at runtime}"
: "${GYM_API_KEY:?Supply an Inference Hub API key at runtime}"
: "${GYM_AGENT:?Supply codex_agent or claude_code_agent at runtime}"
: "${GYM_MODEL:?Supply the Gym model at runtime}"

cd /workspace
if [[ -e /workspace/env.yaml || -e /workspace/results ]]; then
    echo "Use a fresh workspace; existing configuration or results will not be overwritten." >&2
    exit 1
fi

# Read the Hub endpoint and key from the runtime environment.
# Both harnesses use Gym's local model proxy and the same Workplace tools.
cat > /workspace/env.yaml <<'YAML'
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${oc.env:OPENAI_BASE_URL}
      api_key: ${oc.env:GYM_API_KEY}
      model: ${oc.env:GYM_MODEL}
      return_token_id_information: false
      uses_reasoning_parser: false
      is_responses_native: true
      sampling_overrides:
        max_output_tokens: 8192

workplace_assistant:
  resources_servers:
    workplace_assistant:
      entrypoint: app.py
      domain: agent
      expose_tools_over_mcp: true

workplace_assistant_environment_server:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: workplace_assistant_agent
YAML

python - <<'PY'
import json
import math
import os
import subprocess
from pathlib import Path

import yaml

from nemo_gym.cli.env import RunHelper
from nemo_gym.global_config import GlobalConfigDictParserConfig


def _has_model_output(output):
    # Return whether saved output has a tool call or text from the model.
    # Replies, refusals and reasoning summaries count; blank messages don't.
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'function_call':
            return True
        if item.get('type') == 'message' and item.get('role') == 'assistant':
            parts = item.get('content')
        elif item.get('type') == 'reasoning':
            parts = item.get('summary')
        else:
            continue
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get('type') in {'output_text', 'summary_text'}:
                text = part.get('text')
            elif part.get('type') == 'refusal':
                text = part.get('refusal')
            else:
                continue
            if isinstance(text, str) and text.strip():
                return True
    return False


# Choose only the harness-specific settings. Keep the backend, resources and
# evaluation flow shared so both selections perform the same Carlos task.
harness = os.environ['GYM_AGENT']
model = os.environ['GYM_MODEL']
if not model.strip():
    raise ValueError('GYM_MODEL must name a model')
agent = {
    'entrypoint': 'app.py',
    'resources_server': {'type': 'resources_servers', 'name': 'workplace_assistant'},
    'model_server': {'type': 'responses_api_models', 'name': 'policy_model'},
    'model': '${oc.env:GYM_MODEL}',
    'concurrency': 1,
    'timeout': 600,
}
if harness == 'codex_agent':
    agent.update(openai_api_key='local-gym', codex_version='0.162.1')
elif harness == 'claude_code_agent':
    # Use the Claude Code version Anthropic recommends for Haiku 5.5.
    # local-gym connects Claude to Gym's Messages adapter, which calls Hub.
    agent.update(anthropic_api_key='local-gym', claude_code_version='2.1.293', max_turns=30)
else:
    raise ValueError('GYM_AGENT must be codex_agent or claude_code_agent')
config_path = Path('/workspace/env.yaml')
config = yaml.safe_load(config_path.read_text())
config['workplace_assistant_agent'] = {'responses_api_agents': {harness: agent}}
config_path.write_text(yaml.safe_dump(config, sort_keys=False))

output = Path("/workspace/results/carlos_rollouts.jsonl")

# Gym's native helper waits for services to be ready and handles shutdown.
# This reference starts them separately and collects the example with --no-serve.
servers = RunHelper()
try:
    servers.start(GlobalConfigDictParserConfig())
    subprocess.run(
        [
            "gym", "eval", "run", "--no-serve",
            "--agent", "workplace_assistant_agent",
            "--input", "/opt/gym/resources_servers/workplace_assistant/data/example.jsonl",
            "--output", str(output),
            "--limit", "1", "--num-repeats", "1", "--concurrency", "1",
        ],
        check=True,
    )
finally:
    servers.shutdown()

# A zero email reward is a valid scored run. Missing or failed collection is not.
rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
if len(rows) != 1:
    raise RuntimeError(f"Expected one scored rollout, found {len(rows)}")
reward = rows[0].get("reward")
if (isinstance(reward, bool) or not isinstance(reward, (int, float))
        or not math.isfinite(reward) or not 0 <= reward <= 1):
    raise RuntimeError("The saved rollout has no valid Gym reward")
response = rows[0].get("response", {})
# A failed harness run can leave an empty message even when Gym saved a score.
if not (response.get("id") and response.get("model")
        and _has_model_output(response.get("output"))):
    raise RuntimeError("The saved rollout has no model output")
print(f"Saved rollout: {output}")
print(f"Gym email-task reward: {reward}")
PY
