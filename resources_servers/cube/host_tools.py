# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host binary checks for CUBE OSWorld (not installable via pip)."""

from __future__ import annotations

import os
import shutil


_SKIP_ENV = "NEMO_GYM_CUBE_SKIP_QEMU_HOST_CHECK"


def require_qemu_img_if_qemu_backend(vm_backend_class: str) -> None:
    """Raise ``RuntimeError`` if QEMU-style backend is configured but ``qemu-img`` is missing.

    Skipped when ``NEMO_GYM_CUBE_SKIP_QEMU_HOST_CHECK=1`` (e.g. non-QEMU backend mis-detected, or CI).
    """
    if os.environ.get(_SKIP_ENV, "").strip() in ("1", "true", "yes"):
        return
    if "QEMU" not in vm_backend_class.upper():
        return
    if shutil.which("qemu-img"):
        return
    raise RuntimeError(
        "OSWorld QEMU backend requires `qemu-img` on PATH (system package, not pip). "
        "Example — Ubuntu: sudo apt install qemu-system-x86 qemu-utils. "
        f"To skip this check: export {_SKIP_ENV}=1. "
        "See resources_servers/cube/README.md (Host prerequisites)."
    )
