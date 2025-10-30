#!/bin/bash
#
# Automatic retry loop for Qwen3 4B evaluation
# This script runs the evaluation pipeline repeatedly until completion,
# automatically handling Ray crashes and resuming from checkpoints.
#
# Usage:
#   ./run_qwen3_4b_eval_loop.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/run_qwen3_4b_eval.sh"
ROLLOUTS_FILE="${SCRIPT_DIR}/data/qwen3_4b_eval/rollouts.jsonl"
TARGET_ROLLOUTS=3200
MAX_ATTEMPTS=20

echo "==================================================================="
echo "Qwen3 4B Evaluation - Auto-Retry Loop"
echo "==================================================================="
echo ""
echo "This script will:"
echo "  1. Run the evaluation pipeline"
echo "  2. If it crashes (Ray/timeout), automatically restart"
echo "  3. Continue until all ${TARGET_ROLLOUTS} rollouts are collected"
echo "  4. Stop after ${MAX_ATTEMPTS} attempts or completion"
echo ""
echo "Press Ctrl+C to stop at any time."
echo ""

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
    "${MAIN_SCRIPT}" || {
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
    }
    
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
        echo "  ${SCRIPT_DIR}/run_qwen3_4b_eval_loop.sh"
    fi
fi

exit 1

