# Generic Dockerfile for any NeMo Gym server.
#
# Build:
#   docker build --build-arg SERVER_PATH=resources_servers/math_with_code \
#     -t nemogym/math-with-code:latest -f k8s/base/Dockerfile .

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY nemo_gym/ ./nemo_gym/
COPY pyproject.toml README.md ./
RUN mkdir -p cache && uv pip install --system --no-cache . uvicorn[standard]

ARG SERVER_PATH
COPY ${SERVER_PATH}/ ./server/

RUN if [ -f server/requirements.txt ]; then \
      grep -v "nemo-gym" server/requirements.txt > /tmp/reqs.txt || true; \
      uv pip install --system --no-cache -r /tmp/reqs.txt; \
    fi

WORKDIR /app/server
EXPOSE 8001

CMD ["python", "app.py"]
