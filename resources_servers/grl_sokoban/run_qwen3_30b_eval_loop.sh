#!/bin/bash
#
# Automatic retry loop for Qwen3 30B-A3B evaluation
# This script runs the evaluation pipeline repeatedly until completion,
# automatically handling Ray crashes and resuming from checkpoints.
#
# Usage:
#   ./run_qwen3_30b_eval_loop.sh
#

# Don't use set -e here, we want to handle errors explicitly in the retry loop
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/run_qwen3_30b_eval.sh"
ROLLOUTS_FILE="${SCRIPT_DIR}/data/qwen3_30b_eval/rollouts.jsonl"
INPUT_JSONL="${SCRIPT_DIR}/data/qwen3_30b_eval/test_examples_200.jsonl"
MAX_ATTEMPTS=20

# Calculate target rollouts from input file and NUM_REPEATS in main script
# Read NUM_REPEATS from the main script (default to 16 if not found, matching 4B setup)
NUM_REPEATS=$(grep -E "^NUM_REPEATS=" "${MAIN_SCRIPT}" | head -1 | cut -d'=' -f2 | tr -d ' ' | sed 's/#.*//' || echo "16")
TARGET_ROLLOUTS=3200  # 200 prompts × 16 repeats = 3,200 rollouts (matching 4B setup)

echo "==================================================================="
echo "Qwen3 30B-A3B Evaluation - Auto-Retry Loop"
echo "==================================================================="
echo ""
echo "Configuration:"
echo "  Input prompts: ${NUM_PROMPTS}"
echo "  Rollouts per prompt: ${NUM_REPEATS}"
echo "  Target rollouts: ${TARGET_ROLLOUTS}"
echo ""
echo "This script will:"
echo "  1. Run the evaluation pipeline"
echo "  2. If it crashes (Ray/timeout), automatically restart"
echo "  3. Continue until all ${TARGET_ROLLOUTS} rollouts are collected"
echo "  4. Stop after ${MAX_ATTEMPTS} attempts or completion"
echo ""
echo "Press Ctrl+C to stop at any time."
echo ""

# Check if input file exists
if [ ! -f "${INPUT_JSONL}" ]; then
    echo "Error: Input file not found: ${INPUT_JSONL}"
    echo "Please create test_examples_200.jsonl by taking the first 200 lines from test_examples.jsonl:"
    echo "  head -n 200 ${SCRIPT_DIR}/data/test_examples.jsonl > ${INPUT_JSONL}"
    exit 1
fi

# Verify NUM_PROMPTS matches expected
NUM_PROMPTS=$(wc -l < "${INPUT_JSONL}" 2>/dev/null || echo "0")
if [ ${NUM_PROMPTS} -ne 200 ]; then
    echo "Warning: Input file has ${NUM_PROMPTS} prompts, expected 200"
fi

# Make sure main script is executable
chmod +x "${MAIN_SCRIPT}"

# Track attempts
attempt=1

while [ ${attempt} -le ${MAX_ATTEMPTS} ]; do
    echo "==================================================================="
    echo "Attempt ${attempt}/${MAX_ATTEMPTS}"
    echo "==================================================================="
    
    # Count current rollouts
    if [ -f "${ROLLOUTS_FILE}" ]; then
        current_count=$(wc -l < "${ROLLOUTS_FILE}" 2>/dev/null || echo "0")
        echo "Current progress: ${current_count}/${TARGET_ROLLOUTS} rollouts"
        
        # Check if we're done
        if [ "${current_count}" -ge "${TARGET_ROLLOUTS}" ]; then
            echo ""
            echo "✓ Success! All ${TARGET_ROLLOUTS} rollouts collected."
            echo ""
            exit 0
        fi
    else
        echo "Starting fresh - no existing rollouts"
    fi
    
    echo "Starting evaluation pipeline..."
    echo ""
    
    # Run the main script
    if "${MAIN_SCRIPT}"; then
        # Script succeeded - check if we're done
        if [ -f "${ROLLOUTS_FILE}" ]; then
            new_count=$(wc -l < "${ROLLOUTS_FILE}" 2>/dev/null || echo "0")
            echo "Progress after attempt ${attempt}: ${new_count}/${TARGET_ROLLOUTS} rollouts"
            
            # If we're done, exit
            if [ "${new_count}" -ge "${TARGET_ROLLOUTS}" ]; then
                echo ""
                echo "✓ Success! All ${TARGET_ROLLOUTS} rollouts collected."
                echo ""
                exit 0
            fi
        fi
        # Script succeeded but not done yet - this shouldn't happen normally
        # but we'll continue to next attempt if it does
    else
        # Script failed - handle retry
        exit_code=$?
        echo ""
        echo "Pipeline exited with code ${exit_code}"
        
        # Check progress
        if [ -f "${ROLLOUTS_FILE}" ]; then
            new_count=$(wc -l < "${ROLLOUTS_FILE}" 2>/dev/null || echo "0")
            echo "Progress after attempt ${attempt}: ${new_count}/${TARGET_ROLLOUTS} rollouts"
            
            # If we're done, exit
            if [ "${new_count}" -ge "${TARGET_ROLLOUTS}" ]; then
                echo ""
                echo "✓ Success! All ${TARGET_ROLLOUTS} rollouts collected."
                echo ""
                exit 0
            fi
        fi
        
        # Wait before retry
        if [ ${attempt} -lt ${MAX_ATTEMPTS} ]; then
            echo ""
            echo "Waiting 10 seconds before retry..."
            sleep 10
        fi
    fi
    
    attempt=$((attempt + 1))
done

echo ""
echo "Reached maximum attempts (${MAX_ATTEMPTS})"
if [ -f "${ROLLOUTS_FILE}" ]; then
    final_count=$(wc -l < "${ROLLOUTS_FILE}" 2>/dev/null || echo "0")
    echo "Final progress: ${final_count}/${TARGET_ROLLOUTS} rollouts"
    
    if [ "${final_count}" -lt "${TARGET_ROLLOUTS}" ]; then
        echo ""
        echo "⚠ Incomplete - still need $((TARGET_ROLLOUTS - final_count)) rollouts"
        echo "Run this script again to continue:"
        echo "  ${SCRIPT_DIR}/run_qwen3_30b_eval_loop.sh"
    fi
fi

exit 1

