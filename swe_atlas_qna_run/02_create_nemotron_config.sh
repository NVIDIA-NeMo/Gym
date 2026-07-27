#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
CONFIG="${POLICY_CONFIG_PATH}"
mkdir -p "$(dirname "${CONFIG}")"

python - <<'PY2'
import os
from pathlib import Path

config = Path(os.environ["POLICY_CONFIG_PATH"])
config.write_text("""policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${oc.env:POLICY_BASE_URL}
      api_key: ${oc.env:POLICY_API_KEY}
      model: ${oc.env:POLICY_MODEL_NAME}
      return_token_id_information: false
      uses_reasoning_parser: true
      uses_interleaved_reasoning: true
      replace_developer_role_with_system: false
      chat_template_kwargs: null
      extra_body:
        temperature: 1.0
        top_p: 1.0
        max_output_tokens: null
      default_headers: {}
""")
print(f"Wrote {config}")
PY2
