# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runtime image for visual_browser: one headed Chromium per X display, driven by
# PyAutoGUI. It is the environment only; the Gym harness (agent, model, judge)
# runs wherever Gym runs, and leases browser instances from this image.
#
# Base: Playwright's own Python image for the pinned Playwright release, so the
# browser is upstream's build. Its /ms-playwright/chromium-1187 is byte-identical
# (485 files) to the cache used by the WebVoyager 552 reference runs.
# Added on top: the X tools and fonts of the reference browser image
# (webarena.sqsh), Python 3.13, and Gym's visual_browser component with pinned
# dependencies.
#
# Build with resources_servers/visual_browser/docker/build_visual_browser_image.sh,
# which sends only the files this image needs as the build context.

ARG PLAYWRIGHT_IMAGE=mcr.microsoft.com/playwright/python:v1.55.0-noble@sha256:640d578aae63cfb632461d1b0aecb01414e4e020864ac3dd45a868dc0eff3078
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32

FROM ${UV_IMAGE} AS uv

FROM ${PLAYWRIGHT_IMAGE}

# The reference image's X tools and font set. The base already has Xvfb and the
# fonts Playwright installs; xclip (Unicode typing goes through the clipboard),
# xdpyinfo and the remaining reference fonts are added here.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xvfb \
        xauth \
        x11-utils \
        xclip \
        fontconfig \
        fonts-liberation \
        fonts-noto-color-emoji \
        fonts-ipafont-gothic \
        fonts-wqy-zenhei \
        fonts-tlwg-loma-otf \
        fonts-freefont-ttf \
        fonts-unifont \
        fonts-urw-base35 \
        fonts-droid-fallback \
        fonts-noto-mono \
        xfonts-cyrillic \
        xfonts-scalable \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    XDG_SESSION_TYPE=x11 \
    PYTHONDONTWRITEBYTECODE=1

# Gym requires Python >= 3.13.14; this is the same uv-managed CPython build the
# reference runs used.
RUN uv python install 3.13.14

COPY . /opt/nemo-gym
WORKDIR /opt/nemo-gym/resources_servers/visual_browser
RUN uv venv --python 3.13.14 .venv \
    && uv pip install --python .venv/bin/python \
        -c docker/requirements.lock \
        -e /opt/nemo-gym -e . \
    && xvfb-run -a -s "-screen 0 1920x1080x24" .venv/bin/python -c \
        "import nemo_gym.web.browser_instance, resources_servers.visual_browser.backend, playwright, pyautogui; print(pyautogui.size())" \
    && test -x /ms-playwright/chromium-1187/chrome-linux/chrome

ENV PATH=/opt/nemo-gym/resources_servers/visual_browser/.venv/bin:${PATH}

ARG GYM_COMMIT=unknown
LABEL org.opencontainers.image.title="nemo-gym visual-browser runtime" \
      org.opencontainers.image.revision="${GYM_COMMIT}" \
      org.opencontainers.image.source="https://github.com/NVIDIA-NeMo/Gym"

CMD ["bash"]
