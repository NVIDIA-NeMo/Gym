# Source utils and setup environment
# source "$(dirname "$0")/server_utils.sh"
cd /lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/3rdparty/Penguin-workspace/Penguin/
source /resources_servers/workbench/scripts/tau2bench/server_utils.sh
setup_vllm_env
MODEL_NAME=$1
MODEL_PATH=$2
TOOL_PARSER=$3
VLLM_PORT=$4
if [ -z "$TOOL_PARSER" ]; then
    TOOL_PARSER="HERMES"
fi
if [ -z "$VLLM_PORT" ]; then 
    VLLM_PORT=10241
fi
export VLLM_PORT=$VLLM_PORT

VLLM_CACHE_ROOT=$SCRATCH_PATH TORCH_COMPILE_CACHE_DIR=$SCRATCH_PATH vllm serve $MODEL_PATH --served-model-name $MODEL_NAME --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER -tp 4 --port $VLLM_PORT  &

VLLM_PID=$!
echo "vLLM server started at $(date +%Y-%m-%d_%H:%M:%S) with PID: $VLLM_PID"

# Wait for server to be ready and exit with appropriate code
wait_for_server $MODEL_NAME 
exit $?