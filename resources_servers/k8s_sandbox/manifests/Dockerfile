FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY nemo_gym/ ./nemo_gym/
COPY pyproject.toml ./
COPY README.md ./

RUN pip install --no-cache-dir ".[base]" kubernetes uvicorn[standard]

COPY resources_servers/k8s_sandbox/ ./resources_servers/k8s_sandbox/

WORKDIR /app/resources_servers/k8s_sandbox

EXPOSE 8001

CMD ["python", "app.py"]
