#!/bin/bash

# Use this file to install test dependencies and run the LLM judge evaluation.
# It will be copied to /tests/test.sh and run from the working directory.

# Copy agent output files to verifier directory for preservation and evaluation
cp /app/trace.md /logs/verifier/trace.md 2>/dev/null || echo "Warning: trace.md not found"
cp /app/answer.txt /logs/verifier/answer.txt 2>/dev/null || echo "Warning: answer.txt not found"

echo "DEBUG: Running llm_judge.py..."

# Run the LLM judge evaluation (reads from /logs/verifier/)
uv run /tests/llm_judge.py 2>&1 || echo "ERROR: uv run failed with exit code $?"

echo "DEBUG: Script completed"
