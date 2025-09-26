#!/bin/bash

# Common environment setup
setup_vllm_env() {
    export HF_HOME=/lustre/fsw/portfolios/llmservice/users/rgala/hf_cache
    source /lustre/fsw/portfolios/llmservice/users/rgala/venvs/debug/bin/activate

    # Set vLLM port
    if [ -z "$VLLM_PORT" ]; then
        VLLM_PORT=10241
    fi
    
    # Setup directories and cache
    LOGS_VLLM_DIR="/lustre/fsw/portfolios/llmservice/users/rgala/logs/logs_vllm"
    mkdir -p $LOGS_VLLM_DIR
    
    VLLM_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    export VLLM_CACHE_ROOT=$SCRATCH_PATH
    export TORCH_COMPILE_CACHE_DIR=$SCRATCH_PATH
    
    # Export variables for use in calling script
    export VLLM_PORT
    export LOGS_VLLM_DIR
    export VLLM_TIMESTAMP
}

# Wait for server to be ready and return exit code
wait_for_server() {
    local server_name=${1:-"vLLM"}
    local max_attempts=${2:-30}  # Default 30 attempts (5 minutes)
    local port=${3:-$VLLM_PORT}
    
    echo "Waiting for $server_name server on port $port..."
    
    for i in $(seq 1 $max_attempts); do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "$server_name server is ready at $(date +%Y-%m-%d_%H:%M:%S)!"
            return 0
        fi
        if [ $i -eq $max_attempts ]; then
            echo "ERROR: $server_name server failed to start after $((max_attempts * 10 / 60)) minutes"
            return 1
        fi
        sleep 10
    done
}