# Extend mw-code-server with missing dependencies for Software tasks
# Build: docker build -t mw-code-server:latest -f rollout/docker/code-server-deps.Dockerfile .
FROM mw-code-server:latest

USER root

# Install system packages: cmake (for json project tests), pnpm (for tabler tests)
RUN apt-get update \
  && apt-get install -y --no-install-recommends cmake \
  && apt-get clean && rm -rf /var/lib/apt/lists/* \
  && npm install -g pnpm

# Install Python packages: scipy/numpy (for data-analyzer tests)
RUN pip3 install --break-system-packages --no-cache-dir scipy numpy

USER 1000
