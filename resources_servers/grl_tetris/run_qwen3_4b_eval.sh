#!/bin/bash
# Automated inference pipeline for grl_tetris with Qwen3 4B
# Following CONTRIBUTING.md requirements:
# - 200 prompts
# - 16 rollouts per prompt
# - Total: 3200 rollouts
# - vLLM setup with Qwen3 4B

set -e  # Exit on error

# Configuration
MODEL_NAME="Qwen/Qwen3-4B"  # Adjust to actual Qwen3 4B model path
VLLM_PORT=10240
VLLM_HOST="0.0.0.0"
RAY_PORT=6379
TENSOR_PARALLEL_SIZE=1  # Adjust based on GPU availability (1 for single GPU, 2+ for multi-GPU)
GPU_MEMORY_UTILIZATION=0.85  # Adjust based on your GPU memory
MAX_MODEL_LEN=32768

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/qwen3_4b_eval"
INPUT_JSONL="${DATA_DIR}/test_examples_200.jsonl"
OUTPUT_ROLLOUTS="${DATA_DIR}/rollouts.jsonl"
ANALYSIS_REPORT="${DATA_DIR}/reward_analysis.md"
LOG_DIR="${DATA_DIR}/logs"

# Create log directory
mkdir -p "${LOG_DIR}"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for vLLM server to be ready
wait_for_vllm() {
    log_info "Waiting for vLLM server to be ready..."
    local max_attempts=120  # 10 minutes (120 * 5 seconds) - increased for first-time model download
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
            log_info "vLLM server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 5
    done
    
    log_error "vLLM server failed to start after ${max_attempts} attempts"
    return 1
}

# Function to cleanup background processes
cleanup() {
    log_info "Cleaning up processes..."
    
    # Kill vLLM server
    if [ ! -z "${VLLM_PID}" ] && kill -0 ${VLLM_PID} 2>/dev/null; then
        log_info "Stopping vLLM server (PID: ${VLLM_PID})"
        kill ${VLLM_PID} 2>/dev/null || true
        # Force kill if still alive after 2 seconds
        sleep 1
        kill -9 ${VLLM_PID} 2>/dev/null || true
    fi
    
    # Kill NeMo Gym servers
    if [ ! -z "${NEMO_GYM_PID}" ] && kill -0 ${NEMO_GYM_PID} 2>/dev/null; then
        log_info "Stopping NeMo Gym servers (PID: ${NEMO_GYM_PID})"
        kill ${NEMO_GYM_PID} 2>/dev/null || true
        sleep 1
        kill -9 ${NEMO_GYM_PID} 2>/dev/null || true
    fi
    
    # Stop Ray cluster (only if multi-GPU setup was used)
    if [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
        log_info "Stopping Ray cluster..."
        timeout 5 ray stop --force 2>/dev/null || true
    fi
    
    log_info "Cleanup completed"
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Main execution
main() {
    log_info "Starting Qwen3 4B evaluation pipeline for grl_tetris"
    log_info "Configuration:"
    log_info "  Model: ${MODEL_NAME}"
    log_info "  Input prompts: ${INPUT_JSONL}"
    log_info "  Output rollouts: ${OUTPUT_ROLLOUTS}"
    log_info "  Rollouts per prompt: 16"
    log_info "  Total expected rollouts: 3200"
    
    # Check if input file exists
    if [ ! -f "${INPUT_JSONL}" ]; then
        log_error "Input file not found: ${INPUT_JSONL}"
        log_info "Please ensure test_examples_200.jsonl exists in ${DATA_DIR}"
        log_info "You can create it by running:"
        log_info "  head -n 200 ${SCRIPT_DIR}/data/test_examples.jsonl > ${INPUT_JSONL}"
        exit 1
    fi
    
    # Step 1: Clean up any existing processes
    log_info "Step 1: Cleaning up any existing processes..."
    
    # Kill all Ray-related processes more aggressively
    pkill -9 -f "ray::IDLE" 2>/dev/null || true
    pkill -9 -f "ray::RayletMonitor" 2>/dev/null || true  
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "DefaultWorker" 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    
    # Kill vLLM
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    
    # Wait for cleanup
    sleep 5
    
    # Step 2: Start Ray cluster (for multi-GPU setups)
    if [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
        log_info "Step 2: Starting Ray cluster for multi-GPU setup..."
        ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0 --disable-usage-stats
        sleep 3
    else
        log_info "Step 2: Skipping Ray cluster (single GPU setup)"
    fi
    
    # Step 3: Start vLLM server
    log_info "Step 3: Starting vLLM server..."
    log_info "  This may take 2-5 minutes for model loading..."
    
    HF_HOME="${SCRIPT_DIR}/.cache" nohup /workspace/Gym/.venv/bin/vllm serve ${MODEL_NAME} \
        --dtype auto \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --host ${VLLM_HOST} \
        --port ${VLLM_PORT} \
        --max-model-len ${MAX_MODEL_LEN} \
        --trust-remote-code \
        > "${LOG_DIR}/vllm_server.log" 2>&1 &
    
    VLLM_PID=$!
    log_info "vLLM server started (PID: ${VLLM_PID})"
    
    # Wait for vLLM to be ready
    if ! wait_for_vllm; then
        log_error "Failed to start vLLM server. Check logs at ${LOG_DIR}/vllm_server.log"
        exit 1
    fi
    
    # Step 4: Start NeMo Gym servers
    log_info "Step 4: Starting NeMo Gym servers..."
    
    export policy_base_url="http://localhost:${VLLM_PORT}/v1"
    export policy_api_key="dummy"
    export policy_model_name="${MODEL_NAME}"
    
    log_info "Environment variables set:"
    log_info "  policy_model_name=${policy_model_name}"
    log_info "  policy_base_url=${policy_base_url}"
    
    cd "${SCRIPT_DIR}/../.." || exit 1  # Navigate to Gym root
    
    if [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
        # Multi-GPU: use shared Ray cluster
        env policy_base_url="${policy_base_url}" \
            policy_api_key="${policy_api_key}" \
            policy_model_name="${policy_model_name}" \
            nohup /workspace/Gym/.venv/bin/ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_tetris/configs/grl_tetris.yaml]" \
            "+ray_head_node_address=127.0.0.1:${RAY_PORT}" \
            > "${LOG_DIR}/nemo_gym_servers.log" 2>&1 &
    else
        # Single GPU: no Ray cluster needed
        env policy_base_url="${policy_base_url}" \
            policy_api_key="${policy_api_key}" \
            policy_model_name="${policy_model_name}" \
            nohup /workspace/Gym/.venv/bin/ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_tetris/configs/grl_tetris.yaml]" \
            > "${LOG_DIR}/nemo_gym_servers.log" 2>&1 &
    fi
    
    NEMO_GYM_PID=$!
    log_info "NeMo Gym servers started (PID: ${NEMO_GYM_PID})"
    
    # Wait for NeMo Gym servers to be ready
    log_info "Waiting for NeMo Gym servers to be ready..."
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if grep -q "All .* servers ready!" "${LOG_DIR}/nemo_gym_servers.log" 2>/dev/null; then
            log_info "NeMo Gym servers are ready!"
            break
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 5
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "NeMo Gym servers failed to start. Check logs at ${LOG_DIR}/nemo_gym_servers.log"
        exit 1
    fi
    
    sleep 5  # Extra buffer time
    
    # Step 5: Checkpoint check and collect rollouts
    log_info "Step 5: Checking for existing rollouts and preparing collection..."
    
    ACTUAL_INPUT="${INPUT_JSONL}"
    CHECKPOINT_FILE="${DATA_DIR}/remaining_prompts.jsonl"
    
    # Check if we have partial rollouts and need to resume
    if [ -f "${OUTPUT_ROLLOUTS}" ]; then
        existing_count=$(wc -l < "${OUTPUT_ROLLOUTS}" 2>/dev/null || echo "0")
        if [ ${existing_count} -gt 0 ] && [ ${existing_count} -lt 3200 ]; then
            log_warn "Found ${existing_count} existing rollouts (expected 3200)"
            log_info "Creating checkpoint to resume from remaining prompts..."
            
            # Create remaining prompts file
            python "${SCRIPT_DIR}/checkpoint_resume_rollouts.py" \
                --input "${INPUT_JSONL}" \
                --rollouts "${OUTPUT_ROLLOUTS}" \
                --output "${CHECKPOINT_FILE}" \
                --target-repeats 16 \
                2>&1 | tee "${LOG_DIR}/checkpoint.log"
            
            if [ -f "${CHECKPOINT_FILE}" ]; then
                remaining_prompts=$(wc -l < "${CHECKPOINT_FILE}")
                if [ ${remaining_prompts} -gt 0 ]; then
                    log_info "Resuming collection with ${remaining_prompts} remaining prompts"
                    ACTUAL_INPUT="${CHECKPOINT_FILE}"
                    
                    # Backup existing rollouts
                    cp "${OUTPUT_ROLLOUTS}" "${OUTPUT_ROLLOUTS}.backup.$(date +%Y%m%d_%H%M%S)"
                    log_info "Backed up existing rollouts"
                    
                    # Truncate rollouts to only complete prompts (discard partial)
                    # This ensures clean continuation without duplicates
                    completed_prompts=$((${existing_count} / 16))
                    complete_rollouts=$((${completed_prompts} * 16))
                    
                    if [ ${complete_rollouts} -lt ${existing_count} ]; then
                        log_info "Truncating partial rollouts: keeping first ${complete_rollouts} (discarding $((${existing_count} - ${complete_rollouts})) partial)"
                        head -n ${complete_rollouts} "${OUTPUT_ROLLOUTS}" > "${OUTPUT_ROLLOUTS}.tmp"
                        mv "${OUTPUT_ROLLOUTS}.tmp" "${OUTPUT_ROLLOUTS}"
                    fi
                else
                    log_info "All prompts completed! Skipping collection."
                    # Skip to analysis
                    ACTUAL_INPUT=""
                fi
            else
                log_error "Failed to create checkpoint file"
                exit 1
            fi
        elif [ ${existing_count} -ge 3200 ]; then
            log_info "Found ${existing_count} rollouts already collected"
            log_info "Skipping collection and proceeding to analysis"
            ACTUAL_INPUT=""
        fi
    fi
    
    # Collect rollouts if needed
    if [ ! -z "${ACTUAL_INPUT}" ]; then
        # Determine if this is a resume operation
        if [ "${ACTUAL_INPUT}" == "${CHECKPOINT_FILE}" ]; then
            # Resume mode: collect to temp file, then append
            TEMP_OUTPUT="${OUTPUT_ROLLOUTS}.new"
            log_info "Collecting NEW rollouts to append (resume mode)..."
            log_info "  Input file: ${ACTUAL_INPUT}"
            log_info "  Temp output: ${TEMP_OUTPUT}"
            log_info "  Will append to: ${OUTPUT_ROLLOUTS}"
            
            # Get current state
            current_rollouts=$(wc -l < "${OUTPUT_ROLLOUTS}" 2>/dev/null || echo "0")
            remaining_prompts=$(wc -l < "${ACTUAL_INPUT}")
            log_info "  Current progress: ${current_rollouts} rollouts"
            log_info "  Remaining: ${remaining_prompts} prompts × 16 = $((${remaining_prompts} * 16)) new rollouts"
            
            TARGET_FILE="${TEMP_OUTPUT}"
        else
            # Fresh start: write directly
            log_info "Collecting rollouts (fresh start)..."
            log_info "  Input file: ${ACTUAL_INPUT}"
            log_info "  Output file: ${OUTPUT_ROLLOUTS}"
            log_info "  Target: 200 prompts × 16 rollouts = 3200 total rollouts"
            TARGET_FILE="${OUTPUT_ROLLOUTS}"
        fi
        
        /workspace/Gym/.venv/bin/ng_collect_rollouts \
            +agent_name=grl_tetris_game_agent \
            +input_jsonl_fpath="${ACTUAL_INPUT}" \
            +output_jsonl_fpath="${TARGET_FILE}" \
            +limit=null \
            +num_repeats=16 \
            +num_samples_in_parallel=16 \
            +responses_create_params.temperature=0.6 \
            +responses_create_params.top_p=0.95 \
            +responses_create_params.max_output_tokens=4096 \
            2>&1 | tee "${LOG_DIR}/rollout_collection.log"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log_error "Rollout collection failed. Check logs at ${LOG_DIR}/rollout_collection.log"
            
            # In resume mode, still try to append partial results
            if [ "${ACTUAL_INPUT}" == "${CHECKPOINT_FILE}" ] && [ -f "${TEMP_OUTPUT}" ]; then
                new_rollouts=$(wc -l < "${TEMP_OUTPUT}" 2>/dev/null || echo "0")
                if [ ${new_rollouts} -gt 0 ]; then
                    log_info "Appending ${new_rollouts} partial rollouts before exit..."
                    cat "${TEMP_OUTPUT}" >> "${OUTPUT_ROLLOUTS}"
                    rm -f "${TEMP_OUTPUT}"
                fi
            fi
            log_info ""
            log_info "To resume from checkpoint, simply run this script again:"
            log_info "  ${SCRIPT_DIR}/run_qwen3_4b_eval.sh"
            log_info ""
            log_info "The script will automatically:"
            log_info "  1. Detect partial rollouts (${OUTPUT_ROLLOUTS})"
            log_info "  2. Create checkpoint with remaining prompts"
            log_info "  3. Resume collection from where it left off"
            exit 1
        fi
        
        # Success! If resume mode, append temp file to main file
        if [ "${ACTUAL_INPUT}" == "${CHECKPOINT_FILE}" ] && [ -f "${TEMP_OUTPUT}" ]; then
            new_rollouts=$(wc -l < "${TEMP_OUTPUT}")
            log_info "Successfully collected ${new_rollouts} new rollouts"
            log_info "Appending to existing rollouts..."
            cat "${TEMP_OUTPUT}" >> "${OUTPUT_ROLLOUTS}"
            rm -f "${TEMP_OUTPUT}"
            
            total_rollouts=$(wc -l < "${OUTPUT_ROLLOUTS}")
            log_info "Total rollouts now: ${total_rollouts}/3200"
        fi
    fi
    
    # Verify output file
    if [ ! -f "${OUTPUT_ROLLOUTS}" ]; then
        log_error "Output rollouts file not found: ${OUTPUT_ROLLOUTS}"
        exit 1
    fi
    
    local rollout_count=$(wc -l < "${OUTPUT_ROLLOUTS}")
    log_info "Collected ${rollout_count} rollouts"
    
    # Step 6: Analyze rewards
    log_info "Step 6: Analyzing reward distribution..."
    
    cd "${SCRIPT_DIR}" || exit 1
    
    python analyze_rewards.py \
        --rollouts-path "${OUTPUT_ROLLOUTS}" \
        --model-name "Qwen3-4B" \
        --output "${ANALYSIS_REPORT}" \
        2>&1 | tee "${LOG_DIR}/reward_analysis.log"
    
    if [ $? -ne 0 ]; then
        log_warn "Reward analysis failed. You can run it manually later."
    else
        log_info "Reward analysis completed: ${ANALYSIS_REPORT}"
    fi
    
    # Step 7: Summary
    log_info "=========================================="
    log_info "Evaluation pipeline completed successfully!"
    log_info "=========================================="
    log_info "Results:"
    log_info "  Rollouts: ${OUTPUT_ROLLOUTS}"
    log_info "  Analysis: ${ANALYSIS_REPORT}"
    log_info "  Logs: ${LOG_DIR}/"
    log_info ""
    log_info "To view the interactive rollout viewer:"
    log_info "  ng_viewer +jsonl_fpath=${OUTPUT_ROLLOUTS}"
    log_info ""
    log_info "To view the reward analysis report:"
    log_info "  cat ${ANALYSIS_REPORT}"
}

# Run main function
main "$@"

