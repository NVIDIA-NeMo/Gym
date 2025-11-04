#!/bin/bash
# Automated inference pipeline for grl_sokoban with Qwen3 30B-A3B
# Following CONTRIBUTING.md requirements:
# - 500 prompts (test_examples.jsonl)
# - 1-4 rollouts per prompt (configurable)
# - vLLM setup with Qwen3 30B-A3B
# Supports both single GPU and multi-GPU setups

set -e  # Exit on error

# Configuration
MODEL_NAME="Qwen/Qwen3-30B-A3B"
VLLM_PORT=10240
VLLM_HOST="0.0.0.0"
RAY_PORT=6379

# GPU Configuration - Optimized for 4x A100 80GB
TENSOR_PARALLEL_SIZE=4  # Use all 4 GPUs (set to 1 for single GPU)
GPU_MEMORY_UTILIZATION=0.85  
MAX_MODEL_LEN=32768  # Max sequence length 
USE_SHARED_RAY=false  

# Rollout configuration - Optimized for 4x A100 80GB
# Matching 4B model setup: 200 prompts × 16 repeats = 3,200 rollouts
NUM_REPEATS=16  # Number of rollouts per prompt
NUM_SAMPLES_IN_PARALLEL=16  
TEMPERATURE=0.8
MAX_OUTPUT_TOKENS=4096

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/qwen3_30b_eval"
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

# Function to check if vLLM server is ready
check_vllm_ready() {
    if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
        return 0  # Server is ready
    else
        return 1  # Server is not ready
    fi
}

# Function to wait for vLLM server to be ready
wait_for_vllm() {
    log_info "Waiting for vLLM server to be ready..."
    # 30B model on 4 GPUs can take 15-30+ minutes to load, especially on first run
    # Increased timeout to 40 minutes (480 * 5 seconds = 2400 seconds = 40 minutes)
    local max_attempts=480  # 40 minutes (480 * 5 seconds)
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
            log_info "vLLM server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        # Show dots for progress, and log every 60 attempts (5 minutes)
        if [ $((attempt % 60)) -eq 0 ]; then
            echo ""
            log_info "Still waiting... (~$((attempt * 5 / 60)) minutes elapsed)"
        else
            echo -n "."
        fi
        sleep 5
    done
    
    echo ""
    log_error "vLLM server failed to start after ${max_attempts} attempts (~$((max_attempts * 5 / 60)) minutes)"
    log_error "This may indicate:"
    log_error "  - Model loading is taking longer than expected"
    log_error "  - GPU memory issues"
    log_error "  - Network issues (if downloading model)"
    log_error "  - Check ${LOG_DIR}/vllm_server.log for detailed error messages"
    return 1
}

# Function to cleanup background processes
# KEEP_VLLM: if set to "true", will keep vLLM server running (only kill on force quit)
cleanup() {
    local exit_signal=$1
    local is_force_quit=false
    
    # Check if this is a force quit (SIGINT/SIGTERM from terminal)
    if [ "${exit_signal}" = "INT" ] || [ "${exit_signal}" = "TERM" ]; then
        is_force_quit=true
    fi
    
    log_info "Cleaning up processes..."
    
    # Only kill vLLM server if:
    # 1. It's a force quit (Ctrl+C/SIGTERM), OR
    # 2. KEEP_VLLM is explicitly set to false
    if [ "${is_force_quit}" = "true" ] || [ "${KEEP_VLLM:-true}" != "true" ]; then
        if [ "${REUSE_VLLM}" != "true" ] && [ ! -z "${VLLM_PID}" ] && kill -0 ${VLLM_PID} 2>/dev/null; then
            if [ "${is_force_quit}" = "true" ]; then
                log_info "Force quit detected - stopping vLLM server (PID: ${VLLM_PID})"
            else
                log_info "Stopping vLLM server (PID: ${VLLM_PID})"
            fi
            kill ${VLLM_PID} 2>/dev/null || true
            sleep 1
            kill -9 ${VLLM_PID} 2>/dev/null || true
        elif [ "${REUSE_VLLM}" = "true" ]; then
            if [ "${is_force_quit}" = "true" ]; then
                log_info "Force quit detected - stopping reused vLLM server (PID: ${VLLM_PID})"
                kill ${VLLM_PID} 2>/dev/null || true
                sleep 1
                kill -9 ${VLLM_PID} 2>/dev/null || true
            else
                log_info "Keeping vLLM server running (was reused from previous run)"
            fi
        fi
    else
        # Normal exit - keep vLLM server running
        if [ "${REUSE_VLLM}" != "true" ] && [ ! -z "${VLLM_PID}" ] && kill -0 ${VLLM_PID} 2>/dev/null; then
            log_info "Keeping vLLM server running (PID: ${VLLM_PID})"
            log_info "  To stop it manually: kill ${VLLM_PID}"
            log_info "  Or set KEEP_VLLM=false to stop it on exit"
        elif [ "${REUSE_VLLM}" = "true" ]; then
            log_info "Keeping vLLM server running (was reused from previous run)"
        fi
    fi
    
    # Kill NeMo Gym servers (always kill these, they're lightweight)
    if [ ! -z "${NEMO_GYM_PID}" ] && kill -0 ${NEMO_GYM_PID} 2>/dev/null; then
        log_info "Stopping NeMo Gym servers (PID: ${NEMO_GYM_PID})"
        kill ${NEMO_GYM_PID} 2>/dev/null || true
        sleep 1
        kill -9 ${NEMO_GYM_PID} 2>/dev/null || true
    fi
    
    # Stop Ray cluster (only if multi-GPU setup with shared Ray was used)
    if [ "${USE_SHARED_RAY}" = "true" ] && [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
        log_info "Stopping Ray cluster..."
        timeout 5 ray stop --force 2>/dev/null || true
    fi
    
    log_info "Cleanup completed"
}

# Initialize reuse flag (global scope)
export REUSE_VLLM=false

# Set up trap to cleanup on exit
# Use separate handlers for force quit vs normal exit
trap 'cleanup INT' INT
trap 'cleanup TERM' TERM
trap 'cleanup EXIT' EXIT

# Main execution
main() {
    log_info "Starting Qwen3 30B-A3B evaluation pipeline for grl_sokoban"
    log_info "Configuration:"
    log_info "  Model: ${MODEL_NAME}"
    log_info "  Input prompts: ${INPUT_JSONL}"
    log_info "  Output rollouts: ${OUTPUT_ROLLOUTS}"
    log_info "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
    log_info "  Use shared Ray: ${USE_SHARED_RAY}"
    log_info "  Rollouts per prompt: ${NUM_REPEATS}"
    log_info "  Parallel samples: ${NUM_SAMPLES_IN_PARALLEL}"
    
    # Check if input file exists
    if [ ! -f "${INPUT_JSONL}" ]; then
        log_error "Input file not found: ${INPUT_JSONL}"
        log_info "Please ensure test_examples_200.jsonl exists in ${DATA_DIR}"
        log_info "You can create it by taking the first 200 lines from test_examples.jsonl:"
        log_info "  head -n 200 ${SCRIPT_DIR}/data/test_examples.jsonl > ${INPUT_JSONL}"
        exit 1
    fi
    
    # Calculate target rollouts
    local num_prompts=$(wc -l < "${INPUT_JSONL}")
    local target_rollouts=$((num_prompts * NUM_REPEATS))
    log_info "  Target rollouts: ${target_rollouts} (${num_prompts} prompts × ${NUM_REPEATS} repeats)"
    
    # Step 1: Check for existing servers and clean up stale processes
    log_info "Step 1: Checking for existing servers and cleaning up stale processes..."
    
    # Check if vLLM server is already running and ready
    if check_port ${VLLM_PORT} && check_vllm_ready; then
        log_info "vLLM server is already running and ready on port ${VLLM_PORT}"
        log_info "  Reusing existing server (saves ~15-30 minutes of model loading time)"
        VLLM_PID=$(lsof -Pi :${VLLM_PORT} -sTCP:LISTEN -t | head -1)
        export REUSE_VLLM=true
    else
        log_info "No existing vLLM server found or not ready"
        export REUSE_VLLM=false
        
        # Clean up any stale vLLM processes
        pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    fi
    
    # Clean up Ray processes (but keep Ray cluster if shared)
    if [ "${USE_SHARED_RAY}" != "true" ] || [ ${TENSOR_PARALLEL_SIZE} -eq 1 ]; then
        # Kill all Ray-related processes more aggressively
        pkill -9 -f "ray::IDLE" 2>/dev/null || true
        pkill -9 -f "ray::RayletMonitor" 2>/dev/null || true  
        pkill -9 -f "raylet" 2>/dev/null || true
        pkill -9 -f "gcs_server" 2>/dev/null || true
        pkill -9 -f "DefaultWorker" 2>/dev/null || true
        pkill -9 -f "ray::" 2>/dev/null || true
        ray stop --force 2>/dev/null || true
    fi
    
    # Wait for cleanup
    sleep 2
    
    # Step 2: Start Ray cluster (for multi-GPU setups with shared Ray)
    if [ "${USE_SHARED_RAY}" = "true" ] && [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
        log_info "Step 2: Starting shared Ray cluster for multi-GPU setup..."
        log_info "  Ray cluster will be shared across all processes"
        ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0 --disable-usage-stats
        sleep 3
    else
        if [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
            log_info "Step 2: Skipping shared Ray cluster (will use separate Ray instances per process)"
            log_info "  Each process will start its own Ray workers"
        else
            log_info "Step 2: Skipping Ray cluster (single GPU setup)"
        fi
    fi
    
    # Step 3: Start vLLM server (if not already running)
    if [ "${REUSE_VLLM}" = "true" ]; then
        log_info "Step 3: Using existing vLLM server (PID: ${VLLM_PID})"
        log_info "  Skipping model loading (saves ~15-30 minutes)"
        # Note in log that we're reusing an existing server
        echo "[INFO] Reusing existing vLLM server (PID: ${VLLM_PID}) - continuing from previous run at $(date)" >> "${LOG_DIR}/vllm_server.log"
    else
        log_info "Step 3: Starting vLLM server..."
        log_info "  Model: ${MODEL_NAME}"
        log_info "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
        log_info "  Expected startup time: 15-30 minutes for 30B model on 4 GPUs"
        log_info "  This is normal for first run or after restart"
        
        # Note: Configuration is already set at the top of the script
        # The values are optimized for 4x A100 80GB by default
        
        # Append to log file instead of overwriting to preserve previous runs
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
            >> "${LOG_DIR}/vllm_server.log" 2>&1 &
        
        VLLM_PID=$!
        log_info "vLLM server started (PID: ${VLLM_PID})"
        
        # Wait for vLLM to be ready
        if ! wait_for_vllm; then
            log_error "Failed to start vLLM server. Check logs at ${LOG_DIR}/vllm_server.log"
            exit 1
        fi
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
    
    if [ "${USE_SHARED_RAY}" = "true" ] && [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
        # Multi-GPU: use shared Ray cluster
        log_info "Using shared Ray cluster at 127.0.0.1:${RAY_PORT}"
        log_info "  All processes will connect to the same Ray cluster"
        env policy_base_url="${policy_base_url}" \
            policy_api_key="${policy_api_key}" \
            policy_model_name="${policy_model_name}" \
            nohup /workspace/Gym/.venv/bin/ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]" \
            "+ray_head_node_address=127.0.0.1:${RAY_PORT}" \
            >> "${LOG_DIR}/nemo_gym_servers.log" 2>&1 &
    else
        # Single GPU or multi-GPU without shared Ray: separate Ray instances per process
        if [ ${TENSOR_PARALLEL_SIZE} -gt 1 ]; then
            log_info "Starting without shared Ray cluster (separate Ray instances per process)"
        else
            log_info "Starting without Ray cluster (single GPU setup)"
        fi
        env policy_base_url="${policy_base_url}" \
            policy_api_key="${policy_api_key}" \
            policy_model_name="${policy_model_name}" \
            nohup /workspace/Gym/.venv/bin/ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]" \
            >> "${LOG_DIR}/nemo_gym_servers.log" 2>&1 &
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
        echo ""
        log_error "NeMo Gym servers failed to start. Check logs at ${LOG_DIR}/nemo_gym_servers.log"
        exit 1
    fi
    echo ""
    
    # Verify head server is ready (ng_collect_rollouts needs this)
    # The head server starts in a background thread, so we need to wait for it
    log_info "Verifying head server is ready on port 11000..."
    local head_server_ready=false
    local max_head_attempts=30
    
    for i in $(seq 1 ${max_head_attempts}); do
        # Check if head server is responding
        if curl -s -f http://127.0.0.1:11000/global_config_dict_yaml > /dev/null 2>&1; then
            head_server_ready=true
            break
        fi
        # Also check log to see if it's starting
        if grep -q "Uvicorn running on http://127.0.0.1:11000" "${LOG_DIR}/nemo_gym_servers.log" 2>/dev/null; then
            # Head server is starting, wait a bit more
            sleep 2
        else
            sleep 1
        fi
    done
    
    if [ "${head_server_ready}" = "false" ]; then
        log_error "Head server is not ready on port 11000 after ${max_head_attempts} attempts"
        log_error "This is required for ng_collect_rollouts to work"
        log_error "Check ${LOG_DIR}/nemo_gym_servers.log for details"
        exit 1
    fi
    log_info "Head server is ready!"
    
    # Step 5: Checkpoint check and collect rollouts
    log_info "Step 5: Checking for existing rollouts and preparing collection..."
    
    ACTUAL_INPUT="${INPUT_JSONL}"
    CHECKPOINT_FILE="${DATA_DIR}/remaining_prompts.jsonl"
    
    # Check if we have partial rollouts and need to resume
    if [ -f "${OUTPUT_ROLLOUTS}" ]; then
        existing_count=$(wc -l < "${OUTPUT_ROLLOUTS}" 2>/dev/null || echo "0")
        if [ ${existing_count} -gt 0 ] && [ ${existing_count} -lt ${target_rollouts} ]; then
            log_warn "Found ${existing_count} existing rollouts (expected ${target_rollouts})"
            log_info "Creating checkpoint to resume from remaining prompts..."
            
            # Create remaining prompts file
            python "${SCRIPT_DIR}/checkpoint_resume_rollouts.py" \
                --input "${INPUT_JSONL}" \
                --rollouts "${OUTPUT_ROLLOUTS}" \
                --output "${CHECKPOINT_FILE}" \
                --target-repeats ${NUM_REPEATS} \
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
                    completed_prompts=$((${existing_count} / ${NUM_REPEATS}))
                    complete_rollouts=$((${completed_prompts} * ${NUM_REPEATS}))
                    
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
        elif [ ${existing_count} -ge ${target_rollouts} ]; then
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
            log_info "  Remaining: ${remaining_prompts} prompts × ${NUM_REPEATS} = $((${remaining_prompts} * ${NUM_REPEATS})) new rollouts"
            
            TARGET_FILE="${TEMP_OUTPUT}"
        else
            # Fresh start: write directly
            log_info "Collecting rollouts (fresh start)..."
            log_info "  Input file: ${ACTUAL_INPUT}"
            log_info "  Output file: ${OUTPUT_ROLLOUTS}"
            log_info "  Target: ${num_prompts} prompts × ${NUM_REPEATS} rollouts = ${target_rollouts} total rollouts"
            TARGET_FILE="${OUTPUT_ROLLOUTS}"
        fi
        
        cd "${SCRIPT_DIR}/../.." || exit 1
        
        /workspace/Gym/.venv/bin/ng_collect_rollouts \
            +agent_name=grl_sokoban_game_agent \
            +input_jsonl_fpath="${ACTUAL_INPUT}" \
            +output_jsonl_fpath="${TARGET_FILE}" \
            +limit=null \
            +num_repeats=${NUM_REPEATS} \
            +num_samples_in_parallel=${NUM_SAMPLES_IN_PARALLEL} \
            +responses_create_params.temperature=${TEMPERATURE} \
            +responses_create_params.max_output_tokens=${MAX_OUTPUT_TOKENS} \
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
            log_info "  ${SCRIPT_DIR}/run_qwen3_30b_eval.sh"
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
            log_info "Total rollouts now: ${total_rollouts}/${target_rollouts}"
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
        --model-name "Qwen3-30B-A3B" \
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

