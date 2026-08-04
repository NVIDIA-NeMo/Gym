# Shared BioNMIbench-DA Harbor runtime for --shared-data-mode bind.
# Build: docker build -t biomnibench-da-runtime:smoke -f docker/biomnibench-da-runtime.Dockerfile .
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    bash \
    python3 \
    python3-pip \
    python3-venv \
    r-base \
    r-base-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# LLM judge dependency (tests/llm_judge.py). Pre-install so verifier does not
# rely on a runtime pip install under concurrent load.
RUN python3 -m pip install --break-system-packages --no-cache-dir openai

RUN mkdir -p /app/data
WORKDIR /app
CMD ["bash"]
