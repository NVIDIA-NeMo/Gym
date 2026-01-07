#!/bin/bash
# Prepare comp-math-24-25 dataset for ns_tools NeMo Gym resource server
#
# Usage:
#   ./prepare_compmath.sh [OPTIONS]
#
# Options:
#   --prompt_config CONFIG    Prompt config (default: generic/math)
#   --limit N                 Limit to N samples
#   --output PATH             Output file path
#   --no-tools                Don't include any tools
#   --verifier_type TYPE      Set verifier type (e.g., math_with_judge)
#
# Examples:
#   # Basic usage with Python tool and math verifier
#   ./prepare_compmath.sh
#
#   # Use llama3 prompt config
#   ./prepare_compmath.sh --prompt_config llama3-instruct/math
#
#   # Limit to 100 samples
#   ./prepare_compmath.sh --limit 100
#
#   # No tools (pure CoT)
#   ./prepare_compmath.sh --no-tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS_TOOLS_DIR="$(dirname "$SCRIPT_DIR")"
GYM_DIR="$(dirname "$(dirname "$NS_TOOLS_DIR")")"

# Default values
PROMPT_CONFIG="generic/math"
LIMIT=""
OUTPUT=""
TOOLS="nemo_skills.mcp.servers.python_tool.PythonTool"
VERIFIER_TYPE="math_with_judge"
SANDBOX_HOST="${SANDBOX_HOST:-localhost}"
SANDBOX_PORT="${SANDBOX_PORT:-6000}"

# Source dataset - adjust path as needed
NEMO_SKILLS_DIR="${NEMO_SKILLS_DIR:-$GYM_DIR/../nemo_skills}"
INPUT_FILE="$NEMO_SKILLS_DIR/dataset/comp-math-24-25/test.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt_config)
            PROMPT_CONFIG="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --no-tools)
            TOOLS=""
            shift
            ;;
        --tools)
            TOOLS="$2"
            shift 2
            ;;
        --verifier_type)
            VERIFIER_TYPE="$2"
            shift 2
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --sandbox_host)
            SANDBOX_HOST="$2"
            shift 2
            ;;
        --sandbox_port)
            SANDBOX_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default output if not specified
if [ -z "$OUTPUT" ]; then
    OUTPUT="$NS_TOOLS_DIR/data/compmath_prepared.jsonl"
fi

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Set NEMO_SKILLS_DIR or use --input to specify the input file."
    exit 1
fi

# Use whatever Python environment the user already has active
# Don't override their environment

# Build command
CMD="python $SCRIPT_DIR/prepare_dataset.py"
CMD="$CMD --input \"$INPUT_FILE\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --prompt_config \"$PROMPT_CONFIG\""
CMD="$CMD --sandbox_host \"$SANDBOX_HOST\""
CMD="$CMD --sandbox_port $SANDBOX_PORT"

if [ -n "$TOOLS" ]; then
    CMD="$CMD --tools $TOOLS"
fi

if [ -n "$VERIFIER_TYPE" ]; then
    CMD="$CMD --verifier_type \"$VERIFIER_TYPE\""
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD $LIMIT"
fi

echo "========================================"
echo "Preparing dataset for ns_tools"
echo "========================================"
echo "Input:         $INPUT_FILE"
echo "Output:        $OUTPUT"
echo "Prompt config: $PROMPT_CONFIG"
echo "Tools:         ${TOOLS:-none}"
echo "Verifier:      $VERIFIER_TYPE"
echo "========================================"
echo ""

# Run the command
eval $CMD

echo ""
echo "Done! Dataset prepared at: $OUTPUT"
echo ""
echo "To run rollouts:"
echo "  cd $GYM_DIR"
echo "  ./resources_servers/ns_tools/run_demo.sh"
