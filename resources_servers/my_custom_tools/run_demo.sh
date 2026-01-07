#!/bin/bash
# Demo script for my_custom_tools with NeMo Skills Python Tool integration
# 
# Prerequisites:
#   - vLLM server running on localhost:8000 with tool calling enabled:
#     vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
#   - nemo_skills sandbox server running on localhost:6000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
POLICY_BASE_URL="${POLICY_BASE_URL:-http://localhost:8000/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
OUTPUT_FILE="${OUTPUT_FILE:-$GYM_DIR/data/my_custom_tools_python_rollouts.jsonl}"

echo "=============================================="
echo "  My Custom Tools Demo with Python Execution"
echo "=============================================="
echo "GYM_DIR:         $GYM_DIR"
echo "POLICY_BASE_URL: $POLICY_BASE_URL"
echo "POLICY_MODEL:    $POLICY_MODEL_NAME"
echo "NUM_SAMPLES:     $NUM_SAMPLES"
echo "OUTPUT_FILE:     $OUTPUT_FILE"
echo ""

# Activate virtual environment
cd "$GYM_DIR"
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found in $GYM_DIR"
    echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]"
    exit 1
fi

# --- Step 1: Cleanup any existing servers ---
echo "[1/5] Cleaning up existing servers..."
pkill -f "ng_run" 2>/dev/null || true
lsof -ti:11000 2>/dev/null | xargs kill -9 2>/dev/null || true
ray stop --force 2>/dev/null || true

# Remove old venv to force reinstall (comment out if you want to keep it)
# rm -rf "$SCRIPT_DIR/.venv"

sleep 2
echo "      Done."

# --- Step 2: Start servers ---
echo "[2/5] Starting servers..."
CONFIG_PATHS="resources_servers/my_custom_tools/configs/my_custom_tools.yaml"
CONFIG_PATHS="$CONFIG_PATHS,resources_servers/xlam_fc/configs/xlam_fc.yaml"
CONFIG_PATHS="$CONFIG_PATHS,resources_servers/mcqa/configs/mcqa.yaml"
CONFIG_PATHS="$CONFIG_PATHS,resources_servers/my_custom_tools/configs/math_with_judge_no_judge.yaml"
CONFIG_PATHS="$CONFIG_PATHS,responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[$CONFIG_PATHS]" \
  +policy_base_url="$POLICY_BASE_URL" \
  +policy_api_key="$POLICY_API_KEY" \
  +policy_model_name="$POLICY_MODEL_NAME" &

SERVER_PID=$!
echo "      Server PID: $SERVER_PID"

# --- Step 3: Wait for servers to be ready ---
echo "[3/5] Waiting for servers to be ready..."
sleep 15
echo "      Done waiting."

# --- Step 4: Prepare data ---
echo "[4/5] Preparing data..."
PREPARED_DATA="$GYM_DIR/data/my_custom_tools_example.jsonl"

python3 << EOF
import json

input_file = "$SCRIPT_DIR/data/example.jsonl"
output_file = "$PREPARED_DATA"

with open(input_file, 'r') as f, open(output_file, 'w') as out:
    for line in f:
        if line.strip():
            sample = json.loads(line)
            sample['agent_ref'] = {
                'type': 'responses_api_agents',
                'name': 'my_custom_tools_simple_agent'
            }
            out.write(json.dumps(sample) + '\n')

print(f"      Prepared data: {output_file}")
EOF

# --- Step 5: Run rollouts ---
echo "[5/5] Running rollouts..."
rm -f "$OUTPUT_FILE"

ng_collect_rollouts \
  +agent_name=my_custom_tools_simple_agent \
  +input_jsonl_fpath="$PREPARED_DATA" \
  +output_jsonl_fpath="$OUTPUT_FILE" \
  +num_samples_in_parallel=3 \
  +limit="$NUM_SAMPLES"

# --- Results ---
echo ""
echo "=============================================="
echo "  Results"
echo "=============================================="
echo "Output file: $OUTPUT_FILE"
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    python3 << EOF
import json

with open("$OUTPUT_FILE", 'r') as f:
    total_reward = 0
    count = 0
    for i, line in enumerate(f):
        data = json.loads(line)
        reward = data.get('reward', 0)
        total_reward += reward
        count += 1
        
        # Get prompt
        prompt = data['responses_create_params']['input'][-1]['content'][:60]
        
        # Get tool output if available
        tool_output = ""
        for o in data.get('response', {}).get('output', []):
            if o.get('type') == 'function_call_output':
                tool_output = o.get('output', '')[:40]
                break
        
        status = "✓" if reward == 1.0 else "✗"
        print(f"  {status} Sample {i}: reward={reward:.1f} | {prompt}...")
        if tool_output:
            print(f"      Output: {tool_output}")

    avg = total_reward / count if count > 0 else 0
    print(f"\n  Average Reward: {avg:.2f} ({int(total_reward)}/{count})")
EOF
fi

echo ""
echo "=============================================="
echo "  Cleanup"
echo "=============================================="
echo "Servers are still running (PID: $SERVER_PID)"
echo "To stop: kill $SERVER_PID"
echo "Or run:  pkill -f ng_run"
echo ""

