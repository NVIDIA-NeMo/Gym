# Synthetic Dockerfile for NeMo Gym's tests: a build-mode image (the SHELL line needs a build, as the slopcodebench
# Dockerfiles do) with a non-root `agent` account and /app as the working directory. Nothing is copied from the hub.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-lc"]
RUN useradd -m -u 1000 -s /bin/bash agent && mkdir -p /app && chown agent:agent /app

WORKDIR /app
