ARG PYTHON_IMAGE=python:3.12-bookworm
FROM ${PYTHON_IMAGE}

ARG GYM_REPOSITORY=https://github.com/NVIDIA-NeMo/Gym.git
ARG GYM_REVISION
ARG CLAUDE_CODE_VERSION=2.1.207
ARG UV_VERSION=0.10.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends git nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${GYM_REVISION}" \
    && pip install --no-cache-dir "uv==${UV_VERSION}" \
    && npm install --global "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

RUN git clone "${GYM_REPOSITORY}" /workspace/nemo-gym \
    && cd /workspace/nemo-gym \
    && git checkout "${GYM_REVISION}" \
    && git config user.name "NeMo Gym Skill Evaluation" \
    && git config user.email "nemo-gym-skill-eval@example.invalid" \
    && rm -rf .agents/skills .claude/skills .codex/skills \
    && git add -A \
    && git commit --allow-empty -m "Sanitize project-local skills for evaluation" \
    && uv sync --extra dev

COPY benchmarks/agent_skills/fixtures/create-environment-sql/sql_tasks.jsonl /task_inputs/sql_tasks.jsonl

RUN python - <<'PY'
import sqlite3

connection = sqlite3.connect("/task_inputs/shop.db")
connection.executescript(
    """
    CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
    INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');
    """
)
connection.close()
PY

WORKDIR /workspace/nemo-gym
