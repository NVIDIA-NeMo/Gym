MODEL_NAME=$1
DOMAIN=$2
SAVE_FILE_NAME=$3
VLLM_PORT=$4
if [ -z "$DOMAIN" ]; then
    DOMAIN="airline"
fi
echo "vLLM server started at $(date +%Y-%m-%d_%H:%M:%S)"
if ! curl -s "http://0.0.0.0:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "curl -s http://0.0.0.0:$VLLM_PORT/v1/models"
    echo "vLLM server is not running"
    exit 1
fi
export $SCRATCH_PATH="/lustre/fsw/portfolios/llmservice/users/rgala"
source $SCRATCH_PATH/repos/tau2-bench/.venv/bin/activate

tau2 run \
--domain $DOMAIN \
--agent-llm "openai/$MODEL_NAME" \
--agent-llm-args "{\"base_url\": \"http://0.0.0.0:$VLLM_PORT/v1\", \"api_key\": \"\", \"temperature\":0.6}" \
--user-llm gpt-4.1 \
--num-trials 3 \
--max-concurrency 16 \
--save-to $SAVE_FILE_NAME