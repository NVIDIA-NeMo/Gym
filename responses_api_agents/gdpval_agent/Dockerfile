FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    libreoffice libreoffice-writer fonts-dejavu-core \
    tesseract-ocr \
    pandoc \
    poppler-utils \
    ghostscript \
    ffmpeg \
    graphviz \
    openjdk-21-jre \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    build-essential gcc g++ cmake pkg-config make \
    libcairo2-dev \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace